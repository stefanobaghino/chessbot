"""Lichess BOT client: accepts challenges and plays them with a UCI engine.

Configuration comes from environment variables. A .env file is loaded from
DOTENV_PATH (default: <repo>/.env) without overriding variables already set:
  LICHESS_TOKEN  personal API token with bot:play, challenge:read, challenge:write
  ENGINE_PATH    path to the UCI engine binary (default: engine/target/release/chessbot-engine)
  ENGINE_HASH    hash size in MB (default 128)
  ENGINE_THREADS search threads passed as the Threads UCI option (default 1)
  PONDER         1 to keep searching the expected reply on the opponent's time
                 (default 1); costs a core for the whole game, 0 disables it
  MAX_GAMES      concurrent games to accept (default 1)
  SHUTDOWN_TIMEOUT  seconds to wait for games to finish after SIGTERM/SIGINT before
                 resigning them and exiting (default 900)
  STREAM_READ_TIMEOUT  seconds without any byte from the Lichess event stream (it sends
                 keep-alives every few seconds) before the stream is considered wedged
                 and reconnected (default 90)
  HEARTBEAT_INTERVAL  seconds between "alive:" log lines (default 300)
  IDLE_CHALLENGE  1 to challenge an online bot whenever idle (default 0)
  IDLE_CLOCK      clock for those games as seconds+increment (default 300+3), or a
                 weighted list such as 120+1:8,300+3:3,900+10:1: the next game gets the
                 clock furthest below its share of the bot games of the trailing 24 h
  IDLE_RATED      1 for rated (default 1)
  IDLE_MAX_PER_DAY  cap on games finished in the trailing 24 h, incoming ones included
                 (default 80); bot challenges are declined once it is reached
  IDLE_GAP_SECONDS  minimum pause after a game before challenging (default 720)
  IDLE_TICK       seconds between idle checks (default 60)
  IDLE_RATING_RANGE  max rating distance of an opponent (default 500)
  IDLE_MIN_GAMES  minimum blitz games an opponent must have (default 50)
  IDLE_ACCEPT_TIMEOUT  seconds to wait for an opponent before cancelling (default 20)
  IDLE_PAUSE_FILE  while this file exists, no idle challenges are made and incoming
                 challenges are declined (default: unset); SIGUSR1 toggles idle
                 challenging at runtime as well
  BOOK            1 to play the first moves from the Lichess masters opening explorer
                 instead of searching (default 1); 0 disables the book
  BOOK_PLIES      plies after which the book is no longer consulted (default 16)
  BOOK_MIN_GAMES  minimum master games a book move needs (default 50)
SIGHUP re-reads every IDLE_* setting except IDLE_CHALLENGE from the .env file and the
environment (the file wins) on the running process, so ops can retune idle games
without a restart; the new values are logged, a bad value keeps the old ones.

The 24 h game counter is seeded from the Lichess games API at start-up (games against
BOT accounts) and refreshed hourly, so it survives restarts and counts external games.

Liveness: when started by systemd with Type=notify and WatchdogSec, the bot sends
READY=1 after login and WATCHDOG=1 every WatchdogSec/3 seconds while the event stream
is healthy (connected, or reconnected after fewer than 3 consecutive failures).

On SIGTERM or SIGINT the bot drains: it declines new challenges, lets games in
progress finish, then exits 0. A second signal exits immediately.
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import random
import signal
import socket
import sys
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar

import berserk
import chess
import chess.engine
import requests
from dotenv import dotenv_values, load_dotenv

ROOT = Path(__file__).resolve().parent.parent
log = logging.getLogger("chessbot")

ACCEPTED_VARIANTS = {"standard", "fromPosition"}
ACCEPTED_SPEEDS = {"bullet", "blitz", "rapid", "classical"}


def parse_idle_clocks(spec: str) -> list[tuple[tuple[int, int], int]]:
    """Parses IDLE_CLOCK: comma-separated `seconds+increment[:weight]` entries."""
    clocks = []
    for entry in spec.split(","):
        clock, _, weight = entry.strip().partition(":")
        limit, _, inc = clock.partition("+")
        weight = int(weight or 1)
        if weight <= 0:
            raise ValueError(f"bad weight in IDLE_CLOCK entry {entry!r}")
        clocks.append(((int(limit), int(inc or 0)), weight))
    if not clocks:
        raise ValueError("IDLE_CLOCK is empty")
    return clocks


def idle_clock_spec(clocks: list[tuple[tuple[int, int], int]]) -> str:
    return ",".join(f"{limit}+{inc}" + (f":{w}" if len(clocks) > 1 else "") for (limit, inc), w in clocks)


class Config:
    def __init__(self) -> None:
        self.dotenv_path = os.environ.get("DOTENV_PATH", str(ROOT / ".env"))
        load_dotenv(self.dotenv_path)
        self.token = os.environ.get("LICHESS_TOKEN")
        if not self.token:
            sys.exit("LICHESS_TOKEN is not set (put it in .env)")
        self.engine_path = os.environ.get("ENGINE_PATH", str(ROOT / "engine/target/release/chessbot-engine"))
        self.engine_hash = int(os.environ.get("ENGINE_HASH", "128"))
        self.engine_threads = int(os.environ.get("ENGINE_THREADS", "1"))
        self.ponder = os.environ.get("PONDER", "1") == "1"
        self.max_games = int(os.environ.get("MAX_GAMES", "1"))
        self.shutdown_timeout = float(os.environ.get("SHUTDOWN_TIMEOUT", "900"))
        self.stream_read_timeout = float(os.environ.get("STREAM_READ_TIMEOUT", "90"))
        self.login_timeout = float(os.environ.get("LOGIN_TIMEOUT", "300"))
        self.heartbeat_interval = float(os.environ.get("HEARTBEAT_INTERVAL", "300"))
        self.idle_challenge = os.environ.get("IDLE_CHALLENGE", "0") == "1"
        self.book = os.environ.get("BOOK", "1") == "1"
        self.book_plies = int(os.environ.get("BOOK_PLIES", "16"))
        self.book_min_games = int(os.environ.get("BOOK_MIN_GAMES", "50"))
        self.read_idle(os.environ)

    IDLE_KEYS = ("IDLE_CLOCK", "IDLE_RATED", "IDLE_MAX_PER_DAY", "IDLE_GAP_SECONDS", "IDLE_TICK",
                 "IDLE_RATING_RANGE", "IDLE_MIN_GAMES", "IDLE_ACCEPT_TIMEOUT", "IDLE_PAUSE_FILE")

    def read_idle(self, env: Mapping[str, str | None]) -> None:
        """Sets the idle-challenge settings from `env`; on a bad value nothing changes."""
        clocks = parse_idle_clocks(env.get("IDLE_CLOCK") or "300+3")
        values = {
            "idle_clocks": clocks,
            "idle_clock": clocks[0][0],
            "idle_rated": (env.get("IDLE_RATED") or "1") == "1",
            "idle_max_per_day": int(env.get("IDLE_MAX_PER_DAY") or "80"),
            "idle_gap": float(env.get("IDLE_GAP_SECONDS") or "720"),
            "idle_tick": float(env.get("IDLE_TICK") or "60"),
            "idle_rating_range": int(env.get("IDLE_RATING_RANGE") or "500"),
            "idle_min_games": int(env.get("IDLE_MIN_GAMES") or "50"),
            "idle_accept_timeout": float(env.get("IDLE_ACCEPT_TIMEOUT") or "20"),
            "idle_pause_file": env.get("IDLE_PAUSE_FILE") or None,
        }
        self.__dict__.update(values)

    def reload_idle(self) -> None:
        """Re-reads the IDLE_* settings; values in the .env file win over the process
        environment so that editing the file is enough to change them."""
        env: dict[str, str | None] = dict(os.environ)
        env.update(dotenv_values(self.dotenv_path))
        self.read_idle(env)

    def idle_summary(self) -> str:
        return (f"clock={idle_clock_spec(self.idle_clocks)} rated={int(self.idle_rated)} "
                f"max_per_day={self.idle_max_per_day} gap={self.idle_gap:.0f}s tick={self.idle_tick:.0f}s "
                f"rating_range={self.idle_rating_range} min_games={self.idle_min_games} "
                f"accept_timeout={self.idle_accept_timeout:.0f}s pause_file={self.idle_pause_file}")


class TimeoutSession(berserk.TokenSession):
    """Token session with default connect/read timeouts, so a wedged stream raises."""

    def __init__(self, token: str, read_timeout: float) -> None:
        super().__init__(token)
        self.default_timeout = (10, read_timeout)

    def request(self, method, url, **kwargs):
        kwargs.setdefault("timeout", self.default_timeout)
        return super().request(method, url, **kwargs)


def sd_notify(state: str) -> None:
    """Send a systemd notification if NOTIFY_SOCKET is set; silently no-op otherwise."""
    addr = os.environ.get("NOTIFY_SOCKET")
    if not addr:
        return
    if addr.startswith("@"):
        addr = "\0" + addr[1:]
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
            sock.connect(addr)
            sock.sendall(state.encode())
    except OSError as e:
        log.debug("sd_notify failed: %s", e)


def bot_name(entry: dict) -> str:
    """Display name of a /api/bot/online entry (it carries username, not name)."""
    return entry.get("username") or entry.get("name") or entry.get("id") or "?"


def watchdog_period() -> float | None:
    usec = os.environ.get("WATCHDOG_USEC")
    if not usec:
        return None
    return max(1.0, int(usec) / 1_000_000 / 3)


class Book:
    """Opening moves from the Lichess masters explorer: among the moves played in at
    least `min_games` master games that score at least 40% for our side, one is drawn
    with probability proportional to its game count. Anything failing (network, rate
    limit, position not in the database) hands the move back to the engine."""

    URL = "https://explorer.lichess.ovh/masters"
    cache: ClassVar[dict[str, list[dict]]] = {}  # epd -> explorer moves, shared by all games
    disabled_until: ClassVar[float] = 0.0  # after a 429, monotonic time before which the book stays quiet

    def __init__(self, token: str, plies: int = 16, min_games: int = 50, fetch=None) -> None:
        self.token = token
        self.plies = plies
        self.min_games = min_games
        self.fetch = fetch or self.fetch_explorer
        self.out = False  # left the book in this game: stop asking

    def fetch_explorer(self, fen: str) -> list[dict]:
        r = requests.get(self.URL, params={"fen": fen, "moves": 12, "topGames": 0},
                         headers={"Authorization": f"Bearer {self.token}"}, timeout=3)
        if r.status_code == 429:
            Book.disabled_until = time.monotonic() + 600
            log.warning("book: rate limited by the explorer, off for 10 min")
            return []
        r.raise_for_status()
        return r.json().get("moves", [])

    def move(self, board: chess.Board) -> chess.Move | None:
        if self.out or len(board.move_stack) >= self.plies or time.monotonic() < Book.disabled_until:
            return None
        key = board.epd()
        moves = Book.cache.get(key)
        if moves is None:
            try:
                moves = self.fetch(board.fen())
            except Exception as e:  # noqa: BLE001
                log.warning("book: lookup failed (%s: %s), using the engine", type(e).__name__, e)
                self.out = True
                return None
            Book.cache[key] = moves
        candidates = []
        for m in moves:
            total = m.get("white", 0) + m.get("draws", 0) + m.get("black", 0)
            if total < self.min_games:
                continue
            ours = m.get("white", 0) if board.turn == chess.WHITE else m.get("black", 0)
            if (ours + 0.5 * m.get("draws", 0)) / total < 0.4:
                continue
            try:
                candidates.append((board.parse_uci(m["uci"]), total))
            except (KeyError, ValueError):
                continue
        if not candidates:
            self.out = True
            return None
        return random.choices([c[0] for c in candidates], weights=[c[1] for c in candidates])[0]


class Game(threading.Thread):
    def __init__(self, client: berserk.Client, game_id: str, my_id: str, cfg: Config, on_done) -> None:
        super().__init__(daemon=True, name=f"game-{game_id}")
        self.client = client
        self.game_id = game_id
        self.my_id = my_id
        self.cfg = cfg
        self.on_done = on_done
        self.clock: tuple[int, int] | None = None  # (seconds, increment) once gameFull arrived
        self.book = Book(cfg.token, cfg.book_plies, cfg.book_min_games) if cfg.book else None

    def run(self) -> None:
        outcome = "finished"
        try:
            self.play()
        except Exception:
            outcome = "crashed"
            log.exception("game %s crashed", self.game_id)
        finally:
            self.on_done(self.game_id, outcome)

    def new_engine(self) -> chess.engine.SimpleEngine:
        # Own process group: a signal sent to the bot's group (or cgroup by a service
        # manager configured that way) must not kill the engine mid-search.
        engine = chess.engine.SimpleEngine.popen_uci(self.cfg.engine_path, setpgrp=True)
        engine.configure({"Hash": self.cfg.engine_hash, "Threads": self.cfg.engine_threads})
        return engine

    @staticmethod
    def quit_engine(engine: chess.engine.SimpleEngine | None) -> None:
        if engine is None:
            return
        try:
            engine.quit()
        except chess.engine.EngineTerminatedError:
            pass
        except Exception:
            log.exception("engine quit failed")

    TRANSIENT = (berserk.exceptions.BerserkError, requests.exceptions.RequestException)

    def play(self) -> None:
        engine = self.new_engine()
        failures = 0
        try:
            while True:
                try:
                    engine = self.play_stream(engine)
                    return
                except self.TRANSIENT as e:
                    failures += 1
                    if failures > 5:
                        raise
                    log.warning("game %s: stream error (%s: %s), reconnecting (%d/5)", self.game_id, type(e).__name__, e, failures)
                    time.sleep(min(2 * failures, 10))
        finally:
            self.quit_engine(engine)

    def play_stream(self, engine: chess.engine.SimpleEngine) -> chess.engine.SimpleEngine:
        """Follows the game stream until the game ends. Raises on transport errors."""
        stream = self.client.bots.stream_game_state(self.game_id)
        first = next(stream)
        if first.get("type") != "gameFull":
            log.warning("game %s: unexpected first event %s", self.game_id, first.get("type"))
            return engine
        board = self.board_from(first)
        clock = first.get("clock") or {}
        if "initial" in clock:
            self.clock = (self.ms(clock["initial"]) // 1000, self.ms(clock.get("increment")) // 1000)
        my_color = chess.WHITE if first["white"].get("id") == self.my_id else chess.BLACK
        opponent = (first["black"] if my_color else first["white"]).get("name", "?")
        log.info("game %s: playing %s vs %s", self.game_id, "white" if my_color else "black", opponent)
        self.apply_state(board, first["state"])
        if first["state"].get("status", "started") != "started":
            log.info("game %s: already over (%s)", self.game_id, first["state"].get("status"))
            return engine
        engine = self.maybe_move(engine, board, my_color, first["state"])
        for event in stream:
            kind = event.get("type")
            if kind == "gameState":
                board = self.board_from(first)
                self.apply_state(board, event)
                if event.get("status") != "started":
                    winner = event.get("winner")
                    result = "draw" if not winner else ("win" if (winner == "white") == my_color else "loss")
                    log.info("game %s: over (%s) result=%s vs %s", self.game_id, event.get("status"), result, opponent)
                    break
                engine = self.maybe_move(engine, board, my_color, event)
            elif kind == "chatLine" or kind == "opponentGone":
                continue
        return engine

    @staticmethod
    def board_from(game_full: dict) -> chess.Board:
        fen = game_full.get("initialFen", "startpos")
        return chess.Board() if fen == "startpos" else chess.Board(fen)

    @staticmethod
    def apply_state(board: chess.Board, state: dict) -> None:
        moves = state.get("moves", "").split()
        for uci in moves:
            board.push_uci(uci)

    def maybe_move(self, engine: chess.engine.SimpleEngine, board: chess.Board, my_color: chess.Color, state: dict) -> chess.engine.SimpleEngine:
        """Plays our move if it is our turn. Returns the engine, which is re-spawned if it died."""
        if board.turn != my_color or board.is_game_over():
            return engine
        wtime = self.ms(state.get("wtime"))
        btime = self.ms(state.get("btime"))
        winc = self.ms(state.get("winc"))
        binc = self.ms(state.get("binc"))
        if self.book is not None:
            book_move = self.book.move(board)
            if book_move is not None:
                log.info("game %s: book move %s", self.game_id, book_move.uci())
                self.send_move(book_move.uci())
                return engine
        limit = chess.engine.Limit(white_clock=wtime / 1000, black_clock=btime / 1000,
                                   white_inc=winc / 1000, black_inc=binc / 1000)
        # First move: play fast, clocks are not running yet. No pondering from a fixed-time
        # limit either: python-chess reuses the limit for the ponder, and a ponderhit would
        # then not know the clocks.
        first = len(board.move_stack) < 2
        if first:
            limit = chess.engine.Limit(time=0.5)
        result = None
        for attempt in range(3):
            try:
                # With ponder=True python-chess keeps the engine searching the expected reply
                # after bestmove, sends ponderhit if the opponent plays it, stop otherwise.
                result = engine.play(board, limit, ponder=self.cfg.ponder and not first, game=self.game_id)
                break
            except chess.engine.EngineTerminatedError:
                log.warning("game %s: engine died during search, re-spawning (attempt %d)", self.game_id, attempt + 1)
                self.quit_engine(engine)
                engine = self.new_engine()
        if result is None or result.move is None:
            return engine
        self.send_move(result.move.uci())
        return engine

    def send_move(self, uci: str) -> None:
        """Posts the move, retrying transport errors; a lost response whose move was
        accepted shows up as 'Not your turn' on the retry and counts as success."""
        last: Exception | None = None
        for attempt in range(1, 5):
            try:
                self.client.bots.make_move(self.game_id, uci)
                return
            except berserk.exceptions.ResponseError as e:
                text = str(e).lower()
                if "not your turn" in text or "already" in text:
                    # Lichess answers "Not your turn, or game already over" both when a
                    # retry follows a lost response (the move was accepted) and when the
                    # game ended while we were searching, typically an automatic
                    # threefold-repetition draw after the opponent's move (issue #13).
                    # Either way there is nothing left to send.
                    if attempt > 1:
                        log.info("game %s: move %s was accepted by an earlier attempt", self.game_id, uci)
                    else:
                        log.info("game %s: move %s not sent, the game is over or it is not our turn (%s)",
                                 self.game_id, uci, e)
                    return
                last = e
                log.warning("game %s: move %s rejected (%s), attempt %d", self.game_id, uci, e, attempt)
            except self.TRANSIENT as e:
                last = e
                log.warning("game %s: move %s transport error (%s: %s), attempt %d", self.game_id, uci, type(e).__name__, e, attempt)
            time.sleep(attempt)
        raise RuntimeError(f"game {self.game_id}: giving up on move {uci}: {last}")

    @staticmethod
    def ms(value) -> int:
        if value is None:
            return 60_000
        if isinstance(value, datetime.timedelta):
            # berserk 0.14 parses clock fields as timedeltas
            return int(value.total_seconds() * 1000)
        if hasattr(value, "timestamp"):
            # older berserk versions parsed them as datetimes since the epoch
            return int(value.timestamp() * 1000)
        return int(value)


class Bot:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        session = TimeoutSession(cfg.token, cfg.stream_read_timeout)
        self.session = session
        self.client = berserk.Client(session=session)
        account = self.login(cfg.login_timeout)
        self.my_id = account["id"]
        if account.get("title") != "BOT":
            sys.exit(f"account {self.my_id} is not a BOT account; run scripts/upgrade_to_bot.py first")
        self.games: dict[str, Game] = {}
        self.lock = threading.Lock()
        self.draining = threading.Event()
        self.drain_outcomes: dict[str, int] = {}
        self.stream_failures = 0
        self.stream_ok = False
        self.last_line_at: float | None = None
        self.finished_at: list[float] = []
        self.clock_history: list[tuple[float, tuple[int, int]]] = []  # (monotonic end, clock) of bot games, 24 h
        self.pending_challenge: str | None = None
        self.idle_paused = False
        self.idle_pause_logged: str | None = None
        self.skip_until: dict[str, float] = {}
        self.my_rating = account.get("perfs", {}).get("blitz", {}).get("rating", 1500)
        self.signals_received = 0
        self.exit = os._exit  # replaced in tests
        log.info("logged in as %s", account.get("username"))

    def login(self, timeout: float, sleep=time.sleep) -> dict:
        """Fetches the account, retrying transport errors (DNS not up yet after a reboot) for up to `timeout` s."""
        deadline = time.monotonic() + timeout
        delay = 2.0
        for attempt in range(1, 1000):
            try:
                return self.client.account.get()
            except Game.TRANSIENT as e:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise
                log.warning("login attempt %d failed (%s: %s); retrying in %.0fs", attempt, type(e).__name__, e, min(delay, left))
                sleep(min(delay, left))
                delay = min(delay * 2, 30.0)
        raise RuntimeError("unreachable")

    def game_done(self, game_id: str, outcome: str = "finished") -> None:
        with self.lock:
            g = self.games.pop(game_id, None)
            clock = getattr(g, "clock", None)
            if clock:
                self.clock_history.append((time.monotonic(), clock))
            if self.draining.is_set():
                self.drain_outcomes[outcome] = self.drain_outcomes.get(outcome, 0) + 1
        if outcome != "finished":
            log.error("game %s ended by %s", game_id, outcome)
            if not self.draining.is_set():
                threading.Thread(target=self.reattach, args=(game_id,), name=f"reattach-{game_id}", daemon=True).start()

    def reattach(self, game_id: str, delay: float = 3.0) -> None:
        """Re-attaches to any game still in progress after a game thread crashed."""
        time.sleep(delay)
        try:
            ongoing = self.client.games.get_ongoing(count=20)
        except Exception as e:  # noqa: BLE001
            log.warning("reattach: could not list ongoing games (%s)", e)
            return
        for g in ongoing:
            gid = g.get("gameId") or g.get("id")
            if not gid:
                continue
            with self.lock:
                if gid in self.games:
                    continue
                game = Game(self.client, gid, self.my_id, self.cfg, self.game_done)
                self.games[gid] = game
            log.warning("reattach: game %s is still in progress, resuming it", gid)
            game.start()
        if game_id not in [g.get("gameId") or g.get("id") for g in ongoing]:
            log.info("reattach: game %s is no longer in progress", game_id)

    def install_signal_handlers(self) -> None:
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self.on_signal)
        signal.signal(signal.SIGUSR1, self.on_toggle_idle)
        signal.signal(signal.SIGHUP, self.on_reload_idle)

    def on_toggle_idle(self, _signum, _frame) -> None:
        self.idle_paused = not self.idle_paused
        log.info("idle: %s by SIGUSR1", "paused" if self.idle_paused else "resumed")

    def on_reload_idle(self, _signum, _frame) -> None:
        try:
            self.cfg.reload_idle()
        except (ValueError, OSError) as e:
            log.error("idle: settings NOT reloaded on SIGHUP (%s); keeping %s", e, self.cfg.idle_summary())
            return
        log.info("idle: settings reloaded on SIGHUP: %s", self.cfg.idle_summary())

    def seed_game_counter(self) -> int:
        """Rebuilds the 24 h counter from finished games against BOT accounts on Lichess."""
        now_wall = time.time()
        now_mono = time.monotonic()
        since_ms = int((now_wall - 86400) * 1000)
        stamps = []
        clocks: list[tuple[float, tuple[int, int]]] = []
        for g in self.client.games.export_by_player(self.my_id, since=since_ms, max=200, moves=False, finished=True):
            players = g.get("players", {})
            opp = players.get("black" if players.get("white", {}).get("user", {}).get("id") == self.my_id else "white", {})
            if opp.get("user", {}).get("title") != "BOT":
                continue
            clock = g.get("clock") or {}
            ended = g.get("lastMoveAt") or g.get("createdAt")
            if hasattr(ended, "timestamp"):
                ended = ended.timestamp()
            elif isinstance(ended, (int, float)):
                ended = ended / 1000 if ended > 1e11 else ended
            else:
                continue
            stamps.append(now_mono - (now_wall - ended))
            if "initial" in clock:
                clocks.append((stamps[-1], (int(clock["initial"]), int(clock.get("increment") or 0))))
        with self.lock:
            self.finished_at = sorted(stamps)
            self.clock_history = sorted(clocks)
        return len(stamps)

    def refresh_game_counter(self) -> None:
        try:
            n = self.seed_game_counter()
            log.info("idle: %d games vs bots in the last 24h (from Lichess)", n)
        except Exception as e:  # noqa: BLE001
            log.warning("idle: could not refresh the game counter (%s)", e)

    def on_signal(self, signum, _frame) -> None:
        self.signals_received += 1
        name = signal.Signals(signum).name
        if self.signals_received > 1:
            log.warning("received %s again, exiting immediately", name)
            self.exit(1)
            return
        with self.lock:
            n = len(self.games)
        log.info("received %s, draining: %d game(s) in progress", name, n)
        self.draining.set()
        threading.Thread(target=self.drain, name="drain", daemon=True).start()

    def drain(self) -> None:
        deadline = time.monotonic() + self.cfg.shutdown_timeout
        while True:
            with self.lock:
                remaining = list(self.games)
            if not remaining:
                break
            if time.monotonic() >= deadline:
                log.warning("shutdown timeout reached, resigning %d game(s)", len(remaining))
                for game_id in remaining:
                    try:
                        self.client.bots.resign_game(game_id)
                    except Exception:
                        log.exception("failed to resign game %s", game_id)
                with self.lock:
                    threads = list(self.games.values())
                for t in threads:
                    if t.is_alive():
                        t.join(timeout=10)
                break
            time.sleep(1)
        crashed = self.drain_outcomes.get("crashed", 0)
        log.info("drained, exiting (%d finished, %d crashed)", self.drain_outcomes.get("finished", 0), crashed)
        self.exit(0)

    def should_accept(self, challenge: dict) -> str | None:
        if self.draining.is_set() or self.idle_pause_reason():
            # Paused (smoke test or maintenance, #22): nothing may take the game slot.
            return "later"
        if challenge.get("challenger", {}).get("title") == "BOT" and self.games_last_24h() >= self.cfg.idle_max_per_day:
            return "later"
        if challenge.get("variant", {}).get("key") not in ACCEPTED_VARIANTS:
            return "variant"
        if challenge.get("speed") not in ACCEPTED_SPEEDS:
            return "timeControl"
        with self.lock:
            if len(self.games) >= self.cfg.max_games:
                return "later"
        return None

    def games_last_24h(self) -> int:
        cutoff = time.monotonic() - 86400
        with self.lock:
            self.finished_at = [t for t in self.finished_at if t >= cutoff]
            return len(self.finished_at)

    def event_stream(self):
        """Yields incoming events; blank keep-alive lines refresh last_line_at without
        producing an event, so idle and wedged streams can be told apart."""
        with self.session.get("https://lichess.org/api/stream/event", stream=True) as resp:
            resp.raise_for_status()
            self.stream_ok = True
            self.stream_failures = 0
            self.last_line_at = time.monotonic()
            for line in resp.iter_lines():
                self.last_line_at = time.monotonic()
                if line:
                    yield json.loads(line)

    def stream_age(self) -> float | None:
        return None if self.last_line_at is None else time.monotonic() - self.last_line_at

    def healthy(self) -> bool:
        age = self.stream_age()
        if self.stream_ok and age is not None and age > 3 * self.cfg.stream_read_timeout:
            return False
        return self.stream_ok or self.stream_failures < 3

    def heartbeat(self) -> None:
        period = watchdog_period()
        last_log = 0.0
        last_refresh = time.monotonic()
        while True:
            if self.cfg.idle_challenge and time.monotonic() - last_refresh >= 3600:
                self.refresh_game_counter()
                last_refresh = time.monotonic()
            if self.healthy():
                sd_notify("WATCHDOG=1")
            now = time.monotonic()
            if now - last_log >= self.cfg.heartbeat_interval:
                with self.lock:
                    n = len(self.games)
                age = self.stream_age()
                stream = f"ok (last line {age:.0f}s ago)" if self.stream_ok else f"down ({self.stream_failures} failures)"
                log.info("alive: %d game(s), stream %s, %d games in 24h%s", n, stream,
                         self.games_last_24h(), ", draining" if self.draining.is_set() else "")
                last_log = now
            time.sleep(min(period or 60.0, 60.0))

    def pick_opponent(self, bots) -> dict | None:
        now = time.monotonic()
        candidates = []
        for b in bots:
            bid = b.get("id")
            if not bid or bid == self.my_id or self.skip_until.get(bid, 0) > now:
                continue
            blitz = b.get("perfs", {}).get("blitz", {})
            if blitz.get("games", 0) < self.cfg.idle_min_games or blitz.get("prov"):
                continue
            distance = abs(blitz.get("rating", 1500) - self.my_rating)
            if distance > self.cfg.idle_rating_range:
                continue
            candidates.append((distance, b))
        if not candidates:
            return None
        candidates.sort(key=lambda c: c[0])
        # Prefer close ratings but vary the opponent.
        return random.choice(candidates[:8])[1]

    def next_idle_clock(self) -> tuple[int, int]:
        """The configured clock furthest below its weighted share of the bot games of the
        trailing 24 h (deficit selection); with one clock configured, that clock."""
        clocks = self.cfg.idle_clocks
        if len(clocks) == 1:
            return clocks[0][0]
        cutoff = time.monotonic() - 86400
        with self.lock:
            self.clock_history = [h for h in self.clock_history if h[0] >= cutoff]
            counts = {clock: 0 for clock, _ in clocks}
            for _, clock in self.clock_history:
                if clock in counts:
                    counts[clock] += 1
        total_weight = sum(w for _, w in clocks)
        played = sum(counts.values()) + 1  # including the game about to be played
        deficit = {clock: w / total_weight * played - counts[clock] for clock, w in clocks}
        limit, inc = max(clocks, key=lambda c: deficit[c[0]])[0]
        log.info("idle: next clock %s+%s (24h counts %s)", limit, inc,
                 " ".join(f"{c[0]}+{c[1]}={n}" for c, n in counts.items()))
        return limit, inc

    def idle_pause_reason(self) -> str | None:
        if self.idle_paused:
            return "SIGUSR1"
        if self.cfg.idle_pause_file and os.path.exists(self.cfg.idle_pause_file):
            return self.cfg.idle_pause_file
        return None

    def idle_ready(self) -> bool:
        reason = self.idle_pause_reason()
        if reason != self.idle_pause_logged:
            log.info("idle: %s", f"paused by {reason}" if reason else "resumed")
            self.idle_pause_logged = reason
        if reason:
            return False
        with self.lock:
            busy = bool(self.games) or self.pending_challenge is not None
            last = self.finished_at[-1] if self.finished_at else None
        if busy or self.draining.is_set() or not self.stream_ok:
            return False
        if self.games_last_24h() >= self.cfg.idle_max_per_day:
            return False
        return last is None or time.monotonic() - last >= self.cfg.idle_gap

    def challenge_once(self) -> bool:
        """Challenges one online bot and waits for the game to start. Returns True if it did."""
        try:
            bots = list(self.client.bots.get_online_bots(limit=300))
        except Exception as e:  # noqa: BLE001
            log.warning("idle: listing online bots failed (%s)", e)
            return False
        opp = self.pick_opponent(bots)
        if opp is None:
            log.info("idle: no suitable opponent among %d online bots", len(bots))
            return False
        limit, inc = self.next_idle_clock()
        try:
            ch = self.client.challenges.create(opp["id"], rated=self.cfg.idle_rated, clock_limit=limit, clock_increment=inc)
        except Exception as e:  # noqa: BLE001
            log.warning("idle: challenge to %s failed (%s)", bot_name(opp), e)
            self.skip_until[opp["id"]] = time.monotonic() + 3600
            return False
        cid = ch.get("id") or ch.get("challenge", {}).get("id")
        with self.lock:
            self.pending_challenge = cid
        log.info("idle: challenged %s (%s, %s+%s, %s) id=%s", bot_name(opp), opp.get("perfs", {}).get("blitz", {}).get("rating"),
                 limit, inc, "rated" if self.cfg.idle_rated else "casual", cid)
        deadline = time.monotonic() + self.cfg.idle_accept_timeout
        while time.monotonic() < deadline:
            with self.lock:
                if self.games or self.pending_challenge is None:
                    started = bool(self.games)
                    self.pending_challenge = None
                    if not started:
                        self.skip_until[opp["id"]] = time.monotonic() + 3600
                    return started
            time.sleep(0.5)
        with self.lock:
            self.pending_challenge = None
        try:
            self.client.challenges.cancel(cid)
        except Exception as e:  # noqa: BLE001
            log.debug("idle: cancel %s failed (%s)", cid, e)
        log.info("idle: %s did not accept within %.0fs, cancelled", bot_name(opp), self.cfg.idle_accept_timeout)
        self.skip_until[opp["id"]] = time.monotonic() + 3600
        return False

    def idle_loop(self) -> None:
        while True:
            time.sleep(self.cfg.idle_tick)
            try:
                if self.idle_ready():
                    self.challenge_once()
            except Exception:
                log.exception("idle: challenge attempt failed")

    def run(self) -> None:
        log.info("waiting for challenges (max %d concurrent games)", self.cfg.max_games)
        if self.cfg.idle_challenge:
            log.info("idle challenges enabled: %s %s, max %d/day, gap %.0fs", idle_clock_spec(self.cfg.idle_clocks),
                     "rated" if self.cfg.idle_rated else "casual", self.cfg.idle_max_per_day, self.cfg.idle_gap)
            self.refresh_game_counter()
            threading.Thread(target=self.idle_loop, name="idle", daemon=True).start()
        sd_notify("READY=1")
        threading.Thread(target=self.heartbeat, name="heartbeat", daemon=True).start()
        while not self.draining.is_set():
            try:
                for event in self.event_stream():
                    self.handle(event)
                # Stream ended without error (server closed it): reconnect quietly.
                self.stream_ok = False
            except Exception as e:  # noqa: BLE001
                self.stream_ok = False
                if self.draining.is_set():
                    break
                self.stream_failures += 1
                log.warning("event stream failed (%s: %s), reconnecting in 5s (failure %d)",
                            type(e).__name__, e, self.stream_failures)
                time.sleep(5)
        # Draining: the drain thread exits the process once the games are over.
        while True:
            time.sleep(60)

    def handle(self, event: dict) -> None:
        kind = event.get("type")
        if kind == "challenge":
            ch = event["challenge"]
            if ch.get("challenger", {}).get("id") == self.my_id:
                log.info("outgoing challenge %s to %s, waiting for the opponent", ch["id"], ch.get("destUser", {}).get("name"))
                return
            reason = self.should_accept(ch)
            try:
                if reason is None:
                    log.info("accepting challenge %s from %s (%s)", ch["id"], ch["challenger"].get("name"), ch.get("speed"))
                    self.client.bots.accept_challenge(ch["id"])
                else:
                    log.info("declining challenge %s (%s)", ch["id"], reason)
                    try:
                        self.client.bots.decline_challenge(ch["id"], reason=reason)
                    except TypeError:
                        self.client.bots.decline_challenge(ch["id"])
            except berserk.exceptions.ResponseError as e:
                log.warning("challenge %s: request failed (%s)", ch["id"], e)
        elif kind == "gameStart":
            game_id = event["game"]["id"] if "game" in event else event["id"]
            if self.draining.is_set():
                log.info("draining, not starting game %s", game_id)
                return
            with self.lock:
                if game_id in self.games:
                    return
                g = Game(self.client, game_id, self.my_id, self.cfg, self.game_done)
                self.games[game_id] = g
            g.start()
        elif kind == "gameFinish":
            with self.lock:
                self.finished_at.append(time.monotonic())
        elif kind in ("challengeDeclined", "challengeCanceled"):
            ch = event.get("challenge", {})
            with self.lock:
                if ch.get("id") == self.pending_challenge:
                    self.pending_challenge = None
            if ch.get("challenger", {}).get("id") == self.my_id:
                log.info("idle: challenge %s to %s %s (%s)", ch.get("id"), ch.get("destUser", {}).get("name"),
                         "declined" if kind == "challengeDeclined" else "cancelled", ch.get("declineReason", ""))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    bot = Bot(Config())
    bot.install_signal_handlers()
    bot.run()


if __name__ == "__main__":
    main()
