import datetime
import signal
import threading
import time

from bot.lichess_bot import Bot, Game


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
    b.finished_at = []
    b.pending_challenge = None
    b.skip_until = {}
    b.my_rating = 2000
    b.signals_received = 0
    b.exits = []
    b.exit = b.exits.append
    b.cfg = type("Cfg", (), {"shutdown_timeout": timeout, "max_games": 1})()
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
    def play(self, board, limit):
        import chess.engine

        raise chess.engine.EngineTerminatedError("dead")

    def quit(self):
        import chess.engine

        raise chess.engine.EngineTerminatedError("dead")


class LiveEngine:
    def __init__(self):
        self.played = 0

    def play(self, board, limit):
        import chess

        self.played += 1
        return chess.engine.PlayResult(next(iter(board.legal_moves)), None)

    def quit(self):
        pass


def test_engine_is_respawned_mid_game():
    import chess

    g = Game.__new__(Game)
    g.game_id = "g1"
    g.client = type("Client", (), {})()
    g.client.bots = FakeBots()
    g.client.bots.make_move = lambda gid, uci: None
    live = LiveEngine()
    g.new_engine = lambda: live
    board = chess.Board()
    engine = g.maybe_move(DeadEngine(), board, chess.WHITE, {"wtime": 60000, "btime": 60000})
    assert engine is live
    assert live.played == 1


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
    defaults = {"idle_clock": (300, 3), "idle_rated": True, "idle_max_per_day": 80, "idle_gap": 720.0,
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


def test_play_reconnects_stream_after_transport_error(monkeypatch):
    import berserk

    monkeypatch.setattr(time, "sleep", lambda s: None)
    g = make_game()
    g.cfg = type("Cfg", (), {"engine_path": "x", "engine_hash": 16})()
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
