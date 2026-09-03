"""Lichess BOT client: accepts challenges and plays them with a UCI engine.

Configuration comes from environment variables (a .env file is loaded if present):
  LICHESS_TOKEN  personal API token with bot:play, challenge:read, challenge:write
  ENGINE_PATH    path to the UCI engine binary (default: engine/target/release/chessbot-engine)
  ENGINE_HASH    hash size in MB (default 128)
  MAX_GAMES      concurrent games to accept (default 1)
  SHUTDOWN_TIMEOUT  seconds to wait for games to finish after SIGTERM/SIGINT before
                 resigning them and exiting (default 900)

On SIGTERM or SIGINT the bot drains: it declines new challenges, lets games in
progress finish, then exits 0. A second signal exits immediately.
"""
from __future__ import annotations

import datetime
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path

import berserk
import chess
import chess.engine
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
log = logging.getLogger("chessbot")

ACCEPTED_VARIANTS = {"standard", "fromPosition"}
ACCEPTED_SPEEDS = {"bullet", "blitz", "rapid", "classical"}


class Config:
    def __init__(self) -> None:
        load_dotenv(ROOT / ".env")
        self.token = os.environ.get("LICHESS_TOKEN")
        if not self.token:
            sys.exit("LICHESS_TOKEN is not set (put it in .env)")
        self.engine_path = os.environ.get("ENGINE_PATH", str(ROOT / "engine/target/release/chessbot-engine"))
        self.engine_hash = int(os.environ.get("ENGINE_HASH", "128"))
        self.max_games = int(os.environ.get("MAX_GAMES", "1"))
        self.shutdown_timeout = float(os.environ.get("SHUTDOWN_TIMEOUT", "900"))


class Game(threading.Thread):
    def __init__(self, client: berserk.Client, game_id: str, my_id: str, cfg: Config, on_done) -> None:
        super().__init__(daemon=True, name=f"game-{game_id}")
        self.client = client
        self.game_id = game_id
        self.my_id = my_id
        self.cfg = cfg
        self.on_done = on_done

    def run(self) -> None:
        try:
            self.play()
        except Exception:
            log.exception("game %s crashed", self.game_id)
        finally:
            self.on_done(self.game_id)

    def play(self) -> None:
        engine = chess.engine.SimpleEngine.popen_uci(self.cfg.engine_path)
        engine.configure({"Hash": self.cfg.engine_hash})
        try:
            stream = self.client.bots.stream_game_state(self.game_id)
            first = next(stream)
            if first.get("type") != "gameFull":
                log.warning("game %s: unexpected first event %s", self.game_id, first.get("type"))
                return
            board = self.board_from(first)
            my_color = chess.WHITE if first["white"].get("id") == self.my_id else chess.BLACK
            log.info("game %s: playing %s vs %s", self.game_id, "white" if my_color else "black",
                     (first["black"] if my_color else first["white"]).get("name", "?"))
            self.apply_state(board, first["state"])
            self.maybe_move(engine, board, my_color, first["state"])
            for event in stream:
                kind = event.get("type")
                if kind == "gameState":
                    board = self.board_from(first)
                    self.apply_state(board, event)
                    if event.get("status") != "started":
                        log.info("game %s: over (%s)", self.game_id, event.get("status"))
                        break
                    self.maybe_move(engine, board, my_color, event)
                elif kind == "chatLine" or kind == "opponentGone":
                    continue
        finally:
            engine.quit()

    @staticmethod
    def board_from(game_full: dict) -> chess.Board:
        fen = game_full.get("initialFen", "startpos")
        return chess.Board() if fen == "startpos" else chess.Board(fen)

    @staticmethod
    def apply_state(board: chess.Board, state: dict) -> None:
        moves = state.get("moves", "").split()
        for uci in moves:
            board.push_uci(uci)

    def maybe_move(self, engine: chess.engine.SimpleEngine, board: chess.Board, my_color: chess.Color, state: dict) -> None:
        if board.turn != my_color or board.is_game_over():
            return
        wtime = self.ms(state.get("wtime"))
        btime = self.ms(state.get("btime"))
        winc = self.ms(state.get("winc"))
        binc = self.ms(state.get("binc"))
        limit = chess.engine.Limit(white_clock=wtime / 1000, black_clock=btime / 1000,
                                   white_inc=winc / 1000, black_inc=binc / 1000)
        # First move: play fast, clocks are not running yet.
        if len(board.move_stack) < 2:
            limit = chess.engine.Limit(time=0.5)
        result = engine.play(board, limit)
        if result.move is None:
            return
        for attempt in range(3):
            try:
                self.client.bots.make_move(self.game_id, result.move.uci())
                return
            except berserk.exceptions.ResponseError as e:
                log.warning("game %s: move %s rejected (%s), attempt %d", self.game_id, result.move, e, attempt)
                time.sleep(1)

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
        session = berserk.TokenSession(cfg.token)
        self.client = berserk.Client(session=session)
        account = self.client.account.get()
        self.my_id = account["id"]
        if account.get("title") != "BOT":
            sys.exit(f"account {self.my_id} is not a BOT account; run scripts/upgrade_to_bot.py first")
        self.games: dict[str, Game] = {}
        self.lock = threading.Lock()
        self.draining = threading.Event()
        self.signals_received = 0
        self.exit = os._exit  # replaced in tests
        log.info("logged in as %s", account.get("username"))

    def game_done(self, game_id: str) -> None:
        with self.lock:
            self.games.pop(game_id, None)

    def install_signal_handlers(self) -> None:
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self.on_signal)

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
        log.info("drained, exiting")
        self.exit(0)

    def should_accept(self, challenge: dict) -> str | None:
        if self.draining.is_set():
            return "later"
        if challenge.get("variant", {}).get("key") not in ACCEPTED_VARIANTS:
            return "variant"
        if challenge.get("speed") not in ACCEPTED_SPEEDS:
            return "timeControl"
        with self.lock:
            if len(self.games) >= self.cfg.max_games:
                return "later"
        return None

    def run(self) -> None:
        log.info("waiting for challenges (max %d concurrent games)", self.cfg.max_games)
        while not self.draining.is_set():
            try:
                for event in self.client.bots.stream_incoming_events():
                    self.handle(event)
            except Exception:
                if self.draining.is_set():
                    break
                log.exception("event stream failed, reconnecting in 5s")
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
            pass


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    bot = Bot(Config())
    bot.install_signal_handlers()
    bot.run()


if __name__ == "__main__":
    main()
