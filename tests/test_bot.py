import datetime
import logging
import signal
import threading
import time

from bot.lichess_bot import Bot, Config, Game


class FakeBots:
    def __init__(self):
        self.calls = []

    def accept_challenge(self, cid):
        self.calls.append(("accept", cid))

    def decline_challenge(self, cid, reason=None):
        self.calls.append(("decline", cid, reason))

    def resign_game(self, gid):
        self.calls.append(("resign", gid))


def make_bot(timeout=30.0):
    b = Bot.__new__(Bot)
    b.my_id = "me"
    b.client = type("Client", (), {})()
    b.client.bots = FakeBots()
    b.games = {}
    b.lock = threading.Lock()
    b.draining = threading.Event()
    b.drain_outcomes = {}
    b.stream_ok = True
    b.stream_failures = 0
    b.last_line_at = None
    b.finished_at = []
    b.clock_history = []
    b.pending_challenge = None
    b.skip_until = {}
    b.my_rating = 2000
    b.signals_received = 0
    b.exits = []
    b.exit = b.exits.append
    b.idle_paused = False
    b.idle_pause_logged = None
    b.cfg = type("Cfg", (), {"shutdown_timeout": timeout, "max_games": 1, "idle_pause_file": None, "book": False, "tablebase": False})()
    return b


def test_ms_accepts_clock_types():
    assert Game.ms(datetime.timedelta(seconds=299.5)) == 299500
    assert Game.ms(None) == 60000
    assert Game.ms(1234) == 1234
    dt = datetime.datetime.fromtimestamp(12.5, tz=datetime.timezone.utc)
    assert Game.ms(dt) == 12500


def test_own_challenge_is_ignored():
    b = make_bot()
    b.handle({"type": "challenge", "challenge": {"id": "c1", "challenger": {"id": "me"}, "destUser": {"name": "x"}}})
    assert b.client.bots.calls == []


def test_challenge_accepted_when_idle():
    b = make_bot()
    ch = {"id": "c1", "challenger": {"id": "x", "name": "x"}, "variant": {"key": "standard"}, "speed": "blitz"}
    b.handle({"type": "challenge", "challenge": ch})
    assert b.client.bots.calls == [("accept", "c1")]


def test_drain_declines_and_exits_when_games_end():
    b = make_bot()
    b.games["g1"] = threading.Thread()
    b.on_signal(signal.SIGTERM, None)
    assert b.draining.is_set()
    ch = {"id": "c1", "challenger": {"id": "x", "name": "x"}, "variant": {"key": "standard"}, "speed": "blitz"}
    b.handle({"type": "challenge", "challenge": ch})
    assert b.client.bots.calls[-1] == ("decline", "c1", "later")
    b.handle({"type": "gameStart", "game": {"id": "g2"}})
    assert "g2" not in b.games
    time.sleep(1.5)
    assert b.exits == []
    b.game_done("g1")
    time.sleep(1.5)
    assert b.exits == [0]
    b.on_signal(signal.SIGINT, None)
    assert b.exits == [0, 1]


def test_drain_timeout_resigns():
    b = make_bot(timeout=2)
    t = threading.Thread(target=lambda: time.sleep(1))
    t.start()
    b.games["g9"] = t
    b.on_signal(signal.SIGTERM, None)
    time.sleep(3.5)
    assert ("resign", "g9") in b.client.bots.calls
    assert b.exits == [0]


class DeadEngine:
    def play(self, board, limit, **kwargs):
        import chess.engine

        raise chess.engine.EngineTerminatedError("dead")

    def quit(self):
        import chess.engine

        raise chess.engine.EngineTerminatedError("dead")


class LiveEngine:
    def __init__(self):
        self.played = 0
        self.calls = []

    def play(self, board, limit, **kwargs):
        import chess

        self.played += 1
        self.calls.append((limit, kwargs))
        return chess.engine.PlayResult(next(iter(board.legal_moves)), None)

    def quit(self):
        pass


def test_engine_is_respawned_mid_game():
    import chess

    g = Game.__new__(Game)
    g.game_id = "g1"
    g.book = None
    g.tablebase = None
    g.client = type("Client", (), {})()
    g.client.bots = FakeBots()
    g.client.bots.make_move = lambda gid, uci: None
    g.cfg = type("Cfg", (), {"ponder": True})()
    live = LiveEngine()
    g.new_engine = lambda: live
    board = chess.Board()
    engine = g.maybe_move(DeadEngine(), board, chess.WHITE, {"wtime": 60000, "btime": 60000})
    assert engine is live
    assert live.played == 1


def test_maybe_move_ponders_after_the_first_move():
    import chess

    g = Game.__new__(Game)
    g.game_id = "g1"
    g.book = None
    g.tablebase = None
    g.client = type("Client", (), {})()
    g.client.bots = FakeBots()
    g.client.bots.make_move = lambda gid, uci: None
    g.cfg = type("Cfg", (), {"ponder": True})()
    live = LiveEngine()
    board = chess.Board()
    g.maybe_move(live, board, chess.WHITE, {"wtime": 60000, "btime": 60000})
    board.push_uci("e2e4")
    board.push_uci("e7e5")
    g.maybe_move(live, board, chess.WHITE, {"wtime": 60000, "btime": 60000})
    assert [c[1] for c in live.calls] == [{"ponder": False, "game": "g1"}, {"ponder": True, "game": "g1"}]
    assert live.calls[0][0].time == 0.5 and live.calls[1][0].white_clock == 60.0
    g.cfg.ponder = False
    g.maybe_move(live, board, chess.WHITE, {"wtime": 60000, "btime": 60000})
    assert live.calls[-1][1] == {"ponder": False, "game": "g1"}


def test_quit_engine_tolerates_dead_engine():
    Game.quit_engine(DeadEngine())
    Game.quit_engine(None)


def test_game_done_records_drain_outcomes():
    b = make_bot()
    b.drain_outcomes = {}
    b.games["g1"] = threading.Thread()
    b.games["g2"] = threading.Thread()
    b.draining.set()
    b.game_done("g1", "crashed")
    b.game_done("g2")
    assert b.drain_outcomes == {"crashed": 1, "finished": 1}


def test_healthy_tracks_stream_failures():
    b = make_bot()
    b.stream_ok = False
    b.stream_failures = 0
    b.finished_at = []
    assert b.healthy()
    b.stream_failures = 3
    assert not b.healthy()
    b.stream_ok = True
    assert b.healthy()


def test_games_last_24h_counts_and_prunes():
    b = make_bot()
    b.finished_at = [time.monotonic() - 90000, time.monotonic() - 10, time.monotonic()]
    assert b.games_last_24h() == 2
    b.handle({"type": "gameFinish", "game": {"id": "g1"}})
    assert b.games_last_24h() == 3


def test_sd_notify_without_socket_is_noop(monkeypatch):
    from bot.lichess_bot import sd_notify, watchdog_period

    monkeypatch.delenv("NOTIFY_SOCKET", raising=False)
    sd_notify("READY=1")
    monkeypatch.delenv("WATCHDOG_USEC", raising=False)
    assert watchdog_period() is None
    monkeypatch.setenv("WATCHDOG_USEC", "30000000")
    assert watchdog_period() == 10.0


def test_sd_notify_sends_to_unix_socket(monkeypatch, tmp_path):
    import socket

    from bot.lichess_bot import sd_notify

    path = str(tmp_path / "notify.sock")
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    srv.bind(path)
    srv.settimeout(2)
    monkeypatch.setenv("NOTIFY_SOCKET", path)
    sd_notify("WATCHDOG=1")
    assert srv.recv(64) == b"WATCHDOG=1"
    srv.close()


def idle_bot(**cfg):
    b = make_bot()
    b.pending_challenge = None
    b.skip_until = {}
    b.my_rating = 2000
    b.stream_ok = True
    defaults = {"idle_clock": (300, 3), "idle_clocks": [((300, 3), 1)], "idle_rated": True, "idle_max_per_day": 80, "idle_gap": 720.0,
                "idle_rating_range": 500, "idle_min_games": 50, "idle_accept_timeout": 0.5, "idle_pause_file": None,
                "idle_challenge": True}
    b.idle_paused = False
    b.idle_pause_logged = None
    defaults.update(cfg)
    for k, v in defaults.items():
        setattr(b.cfg, k, v)
    return b


def bot_entry(bid, rating, games=100, prov=False):
    return {"id": bid, "name": bid, "perfs": {"blitz": {"rating": rating, "games": games, "prov": prov}}}


def test_pick_opponent_filters_candidates():
    b = idle_bot()
    b.skip_until["skipped"] = time.monotonic() + 100
    bots = [bot_entry("me", 2000), bot_entry("far", 2600), bot_entry("new", 2000, games=3), bot_entry("skipped", 2000),
            bot_entry("prov", 2000, prov=True), bot_entry("good", 2100)]
    assert b.pick_opponent(bots)["id"] == "good"
    assert b.pick_opponent([]) is None


def test_idle_ready_respects_pacing():
    b = idle_bot()
    assert b.idle_ready()
    b.finished_at = [time.monotonic()]
    assert not b.idle_ready()
    b.finished_at = [time.monotonic() - 1000]
    assert b.idle_ready()
    b.finished_at = [time.monotonic() - 1000] * 80
    assert not b.idle_ready()
    b.finished_at = []
    b.games["g"] = threading.Thread()
    assert not b.idle_ready()
    b.games.clear()
    b.stream_ok = False
    assert not b.idle_ready()


def test_bot_challenges_declined_at_daily_cap():
    b = idle_bot(idle_max_per_day=1)
    b.finished_at = [time.monotonic()]
    ch = {"id": "c", "challenger": {"id": "x", "title": "BOT"}, "variant": {"key": "standard"}, "speed": "blitz"}
    assert b.should_accept(ch) == "later"
    ch["challenger"]["title"] = None
    assert b.should_accept(ch) is None


def test_challenge_once_cancels_when_not_accepted():
    b = idle_bot()
    calls = []
    b.client.bots.get_online_bots = lambda limit=None: iter([bot_entry("opp", 2050)])
    b.client.challenges = type("Ch", (), {})()
    b.client.challenges.create = lambda *a, **k: calls.append(("create", a, k)) or {"id": "c1"}
    b.client.challenges.cancel = lambda cid: calls.append(("cancel", cid))
    assert b.challenge_once() is False
    assert calls[0][0] == "create" and calls[0][2] == {"rated": True, "clock_limit": 300, "clock_increment": 3}
    assert ("cancel", "c1") in calls
    assert b.pending_challenge is None
    assert "opp" in b.skip_until


def test_challenge_once_returns_true_when_game_starts():
    b = idle_bot(idle_accept_timeout=5)
    b.client.bots.get_online_bots = lambda limit=None: iter([bot_entry("opp", 2050)])
    b.client.challenges = type("Ch", (), {})()
    b.client.challenges.create = lambda *a, **k: {"id": "c1"}
    b.client.challenges.cancel = lambda cid: (_ for _ in ()).throw(AssertionError("should not cancel"))

    def start_game():
        time.sleep(0.3)
        with b.lock:
            b.games["g1"] = threading.Thread()

    threading.Thread(target=start_game).start()
    assert b.challenge_once() is True
    assert b.pending_challenge is None


def test_declined_event_clears_pending_challenge():
    b = idle_bot()
    b.pending_challenge = "c1"
    b.handle({"type": "challengeDeclined", "challenge": {"id": "c1", "challenger": {"id": "me"}, "destUser": {"name": "x"}}})
    assert b.pending_challenge is None


def test_seed_game_counter_counts_recent_bot_games():
    import datetime as dt

    b = idle_bot()
    now = dt.datetime.now(dt.timezone.utc)
    games = [
        {"players": {"white": {"user": {"id": "me"}}, "black": {"user": {"id": "b1", "title": "BOT"}}},
         "lastMoveAt": now - dt.timedelta(hours=1)},
        {"players": {"white": {"user": {"id": "h1"}}, "black": {"user": {"id": "me"}}},
         "lastMoveAt": now - dt.timedelta(hours=2)},
        {"players": {"white": {"user": {"id": "b2", "title": "BOT"}}, "black": {"user": {"id": "me"}}},
         "lastMoveAt": int((now - dt.timedelta(hours=3)).timestamp() * 1000)},
    ]
    b.client.games = type("G", (), {})()
    b.client.games.export_by_player = lambda *a, **k: iter(games)
    assert b.seed_game_counter() == 2
    assert b.games_last_24h() == 2
    assert b.idle_ready()
    b.cfg.idle_gap = 7200
    assert not b.idle_ready()


def test_pause_file_and_sigusr1_block_idle(tmp_path):
    b = idle_bot()
    b.idle_paused = False
    b.idle_pause_logged = None
    b.cfg.idle_pause_file = str(tmp_path / "pause")
    assert b.idle_ready()
    (tmp_path / "pause").write_text("")
    assert not b.idle_ready()
    (tmp_path / "pause").unlink()
    assert b.idle_ready()
    b.on_toggle_idle(signal.SIGUSR1, None)
    assert not b.idle_ready()
    b.on_toggle_idle(signal.SIGUSR1, None)
    assert b.idle_ready()


def make_game():
    g = Game.__new__(Game)
    g.game_id = "g1"
    g.my_id = "me"
    g.book = None
    g.tablebase = None
    g.client = type("Client", (), {})()
    g.client.bots = FakeBots()
    return g


def test_send_move_retries_transport_errors(monkeypatch):
    import berserk

    monkeypatch.setattr(time, "sleep", lambda s: None)
    g = make_game()
    calls = []

    def make_move(gid, uci):
        calls.append(uci)
        if len(calls) < 3:
            raise berserk.exceptions.ApiError(ConnectionError("Remote end closed connection"))

    g.client.bots.make_move = make_move
    g.send_move("e2e4")
    assert calls == ["e2e4"] * 3


def test_send_move_treats_not_your_turn_as_accepted(monkeypatch):
    import berserk

    monkeypatch.setattr(time, "sleep", lambda s: None)
    g = make_game()
    resp = type("R", (), {"status_code": 400, "reason": "Bad Request", "text": '{"error":"Not your turn, or game already over"}',
                          "json": lambda self: {"error": "Not your turn, or game already over"}})()
    g.client.bots.make_move = lambda gid, uci: (_ for _ in ()).throw(berserk.exceptions.ResponseError(resp))
    g.send_move("e2e4")


def test_send_move_distinguishes_game_over_from_lost_response(monkeypatch, caplog):
    import berserk

    monkeypatch.setattr(time, "sleep", lambda s: None)
    resp = type("R", (), {"status_code": 400, "reason": "Bad Request", "text": '{"error":"Not your turn, or game already over"}',
                          "json": lambda self: {"error": "Not your turn, or game already over"}})()
    # First attempt rejected: the game ended (e.g. threefold repetition) while we searched.
    g = make_game()
    g.client.bots.make_move = lambda gid, uci: (_ for _ in ()).throw(berserk.exceptions.ResponseError(resp))
    with caplog.at_level(logging.INFO, logger="chessbot"):
        g.send_move("e2e4")
    assert "the game is over or it is not our turn" in caplog.text
    caplog.clear()
    # Transport error, then the retry finds the move already on the board: it was accepted.
    g = make_game()
    calls = []

    def make_move(gid, uci):
        calls.append(uci)
        if len(calls) == 1:
            raise berserk.exceptions.ApiError(ConnectionError("Remote end closed connection"))
        raise berserk.exceptions.ResponseError(resp)

    g.client.bots.make_move = make_move
    with caplog.at_level(logging.INFO, logger="chessbot"):
        g.send_move("e2e4")
    assert calls == ["e2e4"] * 2
    assert "accepted by an earlier attempt" in caplog.text


def test_play_reconnects_stream_after_transport_error(monkeypatch):
    import berserk

    monkeypatch.setattr(time, "sleep", lambda s: None)
    g = make_game()
    g.cfg = type("Cfg", (), {"engine_path": "x", "engine_hash": 16, "ponder": True, "tablebase": False})()
    g.new_engine = lambda: LiveEngine()
    g.quit_engine = lambda e: None
    opened = []

    def stream(gid):
        opened.append(gid)
        full = {"type": "gameFull", "white": {"id": "me"}, "black": {"id": "opp", "name": "opp"},
                "state": {"moves": "", "status": "started", "wtime": 60000, "btime": 60000}}
        yield full
        if len(opened) == 1:
            raise berserk.exceptions.ApiError(ConnectionError("dropped"))
        yield {"type": "gameState", "moves": "e2e4 e7e5", "status": "mate", "winner": "white"}

    g.client.bots.stream_game_state = stream
    g.client.bots.make_move = lambda gid, uci: None
    g.play()
    assert len(opened) == 2


def test_reattach_resumes_ongoing_game(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    b = make_bot()
    b.cfg.engine_path = "x"
    b.client.games = type("G", (), {})()
    b.client.games.get_ongoing = lambda count=10: [{"gameId": "g1"}, {"gameId": "g2"}]
    b.games["g2"] = threading.Thread()
    started = []
    monkeypatch.setattr(Game, "start", lambda self: started.append(self.game_id))
    b.reattach("g1", delay=0)
    assert started == ["g1"]
    assert "g1" in b.games


def test_event_stream_counts_keepalives_and_yields_events():
    b = make_bot()
    b.last_line_at = None
    b.stream_ok = False
    b.stream_failures = 2

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def raise_for_status(self):
            pass

        def iter_lines(self):
            yield b""
            yield b""
            yield b'{"type": "gameFinish", "game": {"id": "g1"}}'

    b.session = type("S", (), {"get": lambda self, url, stream=True: Resp()})()
    b.cfg.stream_read_timeout = 90
    events = list(b.event_stream())
    assert [e["type"] for e in events] == ["gameFinish"]
    assert b.stream_ok and b.stream_failures == 0
    assert b.stream_age() is not None and b.stream_age() < 1
    assert b.healthy()
    b.last_line_at = time.monotonic() - 1000
    assert not b.healthy()


def test_bot_name_prefers_username():
    from bot.lichess_bot import bot_name

    assert bot_name({"id": "pi0w", "username": "Pi0w"}) == "Pi0w"
    assert bot_name({"id": "pi0w", "name": "Pi0w"}) == "Pi0w"
    assert bot_name({"id": "pi0w"}) == "pi0w"
    assert bot_name({}) == "?"


def test_challenge_log_uses_username(caplog):
    b = idle_bot()
    entry = {"id": "pi0w", "username": "Pi0w", "perfs": {"blitz": {"rating": 1976, "games": 500}}}
    b.client.bots.get_online_bots = lambda limit=None: iter([entry])
    b.client.challenges = type("Ch", (), {})()
    b.client.challenges.create = lambda *a, **k: {"id": "c1"}
    b.client.challenges.cancel = lambda cid: None
    with caplog.at_level("INFO", logger="chessbot"):
        b.challenge_once()
    assert "idle: challenged Pi0w (1976" in caplog.text


def test_engine_threads_option_is_configured(monkeypatch):
    import chess.engine

    configured = {}

    class FakeEngine:
        def configure(self, opts):
            configured.update(opts)

    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", staticmethod(lambda path, setpgrp=False: FakeEngine()))
    g = make_game()
    g.cfg = type("Cfg", (), {"engine_path": "x", "engine_hash": 64, "engine_threads": 3})()
    g.new_engine()
    assert configured == {"Hash": 64, "Threads": 3}


def test_sighup_reloads_idle_settings_from_dotenv(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("IDLE_CLOCK=300+3\n")
    monkeypatch.setenv("DOTENV_PATH", str(env))
    monkeypatch.setenv("LICHESS_TOKEN", "x")
    monkeypatch.delenv("IDLE_CLOCK", raising=False)
    cfg = Config()
    assert cfg.idle_clock == (300, 3) and cfg.idle_rated
    b = make_bot()
    b.cfg = cfg
    env.write_text("IDLE_CLOCK=120+1\nIDLE_RATED=0\nIDLE_GAP_SECONDS=30\n")
    b.on_reload_idle(signal.SIGHUP, None)
    assert cfg.idle_clock == (120, 1) and not cfg.idle_rated and cfg.idle_gap == 30
    # A bad value keeps the previous settings.
    env.write_text("IDLE_CLOCK=bogus\n")
    b.on_reload_idle(signal.SIGHUP, None)
    assert cfg.idle_clock == (120, 1)
    # A key removed from the file falls back to the environment, then the default.
    env.write_text("")
    monkeypatch.setenv("IDLE_MAX_PER_DAY", "10")
    b.on_reload_idle(signal.SIGHUP, None)
    assert cfg.idle_clock == (300, 3) and cfg.idle_max_per_day == 10


def test_login_retries_transport_errors_until_the_deadline(caplog):
    import requests

    b = make_bot()
    calls = []

    class Account:
        def get(self):
            calls.append(1)
            if len(calls) < 3:
                raise requests.exceptions.ConnectionError("Failed to resolve 'lichess.org'")
            return {"id": "me", "title": "BOT"}

    b.client.account = Account()
    slept = []
    with caplog.at_level(logging.WARNING):
        account = b.login(60.0, sleep=slept.append)
    assert account["id"] == "me"
    assert len(calls) == 3
    assert slept == [2.0, 4.0]
    assert [r.levelname for r in caplog.records] == ["WARNING", "WARNING"]
    assert "login attempt 1 failed (ConnectionError" in caplog.text


def test_login_gives_up_after_the_deadline(monkeypatch):
    import pytest
    import requests

    b = make_bot()

    class Account:
        def get(self):
            raise requests.exceptions.ConnectionError("still down")

    b.client.account = Account()
    now = [0.0]
    monkeypatch.setattr("bot.lichess_bot.time.monotonic", lambda: now[0])

    def sleep(s):
        now[0] += s

    with pytest.raises(requests.exceptions.ConnectionError):
        b.login(10.0, sleep=sleep)
    assert now[0] == 10.0


def test_parse_idle_clocks():
    from bot.lichess_bot import idle_clock_spec, parse_idle_clocks

    assert parse_idle_clocks("300+3") == [((300, 3), 1)]
    assert parse_idle_clocks("120+1:8, 300+3:3,900+10:1") == [((120, 1), 8), ((300, 3), 3), ((900, 10), 1)]
    assert idle_clock_spec(parse_idle_clocks("120+1:8,300+3:3")) == "120+1:8,300+3:3"
    assert idle_clock_spec(parse_idle_clocks("120+1")) == "120+1"
    for bad in ("", "120+1:0", "abc", "120+1:x"):
        try:
            parse_idle_clocks(bad)
        except ValueError:
            continue
        raise AssertionError(bad)


def test_next_idle_clock_follows_the_weights():
    b = idle_bot(idle_clocks=[((120, 1), 8), ((300, 3), 3), ((900, 10), 1)])
    picks = []
    for _ in range(12):
        clock = b.next_idle_clock()
        picks.append(clock)
        b.clock_history.append((time.monotonic(), clock))
    assert picks[0] == (120, 1)
    assert sorted(picks.count(c) for c in {(120, 1), (300, 3), (900, 10)}) == [1, 3, 8]
    # Games older than 24 h and clocks outside the list do not count.
    b.clock_history = [(time.monotonic() - 90000, (300, 3))] * 5 + [(time.monotonic(), (60, 0))] * 5
    assert b.next_idle_clock() == (120, 1)
    # Incoming games with a listed clock count towards its share.
    b.clock_history = [(time.monotonic(), (120, 1))] * 8
    assert b.next_idle_clock() == (300, 3)
    assert idle_bot().next_idle_clock() == (300, 3)


def test_game_done_records_the_clock():
    b = make_bot()
    g = Game.__new__(Game)
    g.clock = (120, 1)
    b.games["g1"] = g
    b.game_done("g1")
    assert [c for _, c in b.clock_history] == [(120, 1)]
    b.games["g2"] = threading.Thread()
    b.game_done("g2")
    assert len(b.clock_history) == 1


def test_game_reads_the_clock_from_gamefull():
    g = Game.__new__(Game)
    g.my_id = "me"
    g.game_id = "g"
    g.clock = None
    calls = []

    class Bots:
        def stream_game_state(self, gid):
            yield {"type": "gameFull", "clock": {"initial": datetime.timedelta(seconds=120), "increment": 1000},
                   "white": {"id": "me"}, "black": {"name": "x"}, "initialFen": "startpos",
                   "state": {"moves": "", "status": "mate"}}

    g.client = type("C", (), {"bots": Bots()})()
    g.apply_state = lambda board, state: calls.append(state)
    g.board_from = lambda first: __import__("chess").Board()
    g.play_stream(None)
    assert g.clock == (120, 1)


def test_incoming_challenges_declined_while_paused(tmp_path):
    b = idle_bot()
    ch = {"id": "c", "challenger": {"id": "x"}, "variant": {"key": "standard"}, "speed": "blitz"}
    assert b.should_accept(ch) is None
    b.idle_paused = True
    assert b.should_accept(ch) == "later"
    b.idle_paused = False
    pause = tmp_path / "pause"
    pause.touch()
    b.cfg.idle_pause_file = str(pause)
    assert b.should_accept(ch) == "later"
    pause.unlink()
    assert b.should_accept(ch) is None


def explorer(*rows):
    return [{"uci": u, "white": w, "draws": d, "black": bl} for u, w, d, bl in rows]


def test_book_picks_well_tried_moves_and_falls_back():
    from bot.lichess_bot import Book

    Book.cache.clear()
    Book.disabled_until = 0.0
    import chess

    calls = []

    def fetch(fen):
        calls.append(fen)
        if fen.split()[1] == "b":
            return explorer(("e7e5", 300, 300, 300), ("c7c5", 300, 300, 300))
        return explorer(("e2e4", 400, 300, 300), ("d2d4", 300, 300, 300), ("h2h4", 30, 10, 60), ("a2a3", 10, 100, 500))

    book = Book("tok", plies=4, min_games=50, fetch=fetch)
    board = chess.Board()
    picks = {book.move(board).uci() for _ in range(50)}
    assert picks == {"e2e4", "d2d4"}  # h4 too rare, a3 scores under 40% for white
    assert len(calls) == 1  # cached
    board.push_uci("e2e4")
    assert book.move(board).uci() in {"e7e5", "c7c5"}
    assert len(calls) == 2
    # Beyond BOOK_PLIES the book is silent.
    for uci in ("e7e5", "g1f3", "b8c6"):
        board.push_uci(uci)
    assert book.move(board) is None
    # Out of the database: engine takes over for the rest of the game.
    Book.cache.clear()
    empty = Book("tok", plies=40, min_games=50, fetch=lambda fen: [])
    b2 = chess.Board()
    assert empty.move(b2) is None
    b2.push_uci("e2e4")
    assert empty.move(b2) is None and empty.out
    # Errors do not propagate.
    def boom(fen):
        raise OSError("down")
    Book.cache.clear()
    broken = Book("tok", fetch=boom)
    assert broken.move(chess.Board()) is None and broken.out
    # A rate limit silences every book for a while.
    Book.cache.clear()
    Book.disabled_until = time.monotonic() + 100
    assert Book("tok", fetch=fetch).move(chess.Board()) is None
    Book.disabled_until = 0.0


def test_book_scores_for_black():
    from bot.lichess_bot import Book

    Book.cache.clear()
    import chess

    board = chess.Board()
    board.push_uci("e2e4")
    def fetch(fen):
        return explorer(("e7e5", 600, 200, 200), ("c7c5", 300, 300, 400))

    picks = {Book("tok", fetch=fetch).move(board).uci() for _ in range(40)}
    assert picks == {"c7c5"}  # e5 scores only 30% for black here


def tb_data(category, *moves):
    return {"category": category, "dtz": 3, "moves": [{"uci": u, "category": c} for u, c in moves]}


def tb_game(fetch, engine_move):
    import chess

    g = Game.__new__(Game)
    g.game_id = "g1"
    g.book = None
    g.tablebase = None
    g.cfg = type("Cfg", (), {"ponder": False})()
    g.client = type("Client", (), {})()
    g.client.bots = FakeBots()
    sent = []
    g.client.bots.make_move = lambda gid, uci: sent.append(uci)
    from bot.lichess_bot import Tablebase

    g.tablebase = Tablebase(fetch=fetch)

    class Engine:
        def __init__(self):
            self.played = 0

        def play(self, board, limit, **kwargs):
            self.played += 1
            return chess.engine.PlayResult(chess.Move.from_uci(engine_move), None)

    return g, sent, Engine()


def test_tablebase_converts_wins_without_the_engine():
    import chess

    from bot.lichess_bot import Tablebase

    Tablebase.disabled_until = 0.0
    board = chess.Board("8/8/8/8/8/4k3/4p3/4K3 b - - 0 1")
    calls = []

    def fetch(fen):
        calls.append(fen)
        return tb_data("win", ("e3d3", "loss"), ("e3d4", "draw"))

    g, sent, engine = tb_game(fetch, "e3d4")
    g.maybe_move(engine, board, chess.BLACK, {"wtime": 60000, "btime": 60000})
    assert sent == ["e3d3"] and engine.played == 0 and calls == [board.fen()]
    # Lost: the tablebase's first move (longest resistance) is played too.
    g, sent, engine = tb_game(lambda fen: tb_data("loss", ("e1f2", "win")), "e1f2")
    g.maybe_move(engine, chess.Board("8/8/8/8/8/3k4/4p3/4K3 w - - 0 1"), chess.WHITE, {"wtime": 60000, "btime": 60000})
    assert sent == ["e1f2"] and engine.played == 0


def test_tablebase_lets_the_engine_play_draws_but_not_lose_them():
    import chess

    board = chess.Board("8/8/8/8/3k4/8/3P4/3K4 w - - 0 1")
    data = tb_data("draw", ("d2d3", "draw"), ("d1c1", "draw"), ("d1e2", "win"))
    g, sent, engine = tb_game(lambda fen: data, "d1c1")
    g.maybe_move(engine, board, chess.WHITE, {"wtime": 60000, "btime": 60000})
    assert sent == ["d1c1"] and engine.played == 1
    g, sent, engine = tb_game(lambda fen: data, "d1e2")
    g.maybe_move(engine, board, chess.WHITE, {"wtime": 60000, "btime": 60000})
    assert sent == ["d2d3"] and engine.played == 1


def test_tablebase_is_skipped_when_it_does_not_apply():
    import chess

    calls = []

    def fetch(fen):
        calls.append(fen)
        return tb_data("win", ("e2e4", "loss"))

    g, sent, engine = tb_game(fetch, "e2e4")
    g.maybe_move(engine, chess.Board(), chess.WHITE, {"wtime": 60000, "btime": 60000})  # 32 pieces
    g.maybe_move(engine, chess.Board("4k3/8/8/8/8/8/8/4K2R w K - 0 1"), chess.WHITE, {"wtime": 60000, "btime": 60000})  # castling
    assert calls == [] and engine.played == 2

    def boom(fen):
        raise OSError("down")

    g, sent, engine = tb_game(boom, "e1f2")
    g.maybe_move(engine, chess.Board("8/8/8/8/8/3k4/4p3/4K3 w - - 0 1"), chess.WHITE, {"wtime": 60000, "btime": 60000})
    assert sent == ["e1f2"] and engine.played == 1
    from bot.lichess_bot import Tablebase

    Tablebase.disabled_until = time.monotonic() + 100
    g, sent, engine = tb_game(fetch, "e1f2")
    g.maybe_move(engine, chess.Board("8/8/8/8/8/3k4/4p3/4K3 w - - 0 1"), chess.WHITE, {"wtime": 60000, "btime": 60000})
    assert calls == [] and engine.played == 1
    Tablebase.disabled_until = 0.0
