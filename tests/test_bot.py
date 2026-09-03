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
