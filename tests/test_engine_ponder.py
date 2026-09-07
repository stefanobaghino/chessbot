"""End-to-end check of the engine's pondering through python-chess, the way the bot drives it."""

import os
import time
from pathlib import Path

import chess
import chess.engine
import pytest

ENGINE = Path(os.environ.get("ENGINE_PATH", Path(__file__).resolve().parents[1] / "engine/target/release/chessbot-engine"))
pytestmark = pytest.mark.skipif(not ENGINE.exists(), reason="engine binary not built")


def test_ponderhit_and_ponder_miss():
    engine = chess.engine.SimpleEngine.popen_uci(str(ENGINE))
    try:
        assert "Ponder" in engine.options
        limit = chess.engine.Limit(white_clock=3, black_clock=3, white_inc=0.1, black_inc=0.1)
        board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
        r1 = engine.play(board, limit, ponder=True, game="g")
        assert r1.move in board.legal_moves and r1.ponder is not None
        board.push(r1.move)
        assert r1.ponder in board.legal_moves
        # Opponent plays the expected reply: ponderhit, the engine answers from its ponder search.
        board.push(r1.ponder)
        t0 = time.monotonic()
        r2 = engine.play(board, limit, ponder=True, game="g")
        assert r2.move in board.legal_moves
        assert time.monotonic() - t0 < 2.5
        board.push(r2.move)
        # Opponent plays something else: the ponder is stopped and a fresh search runs.
        other = next(m for m in board.legal_moves if m != r2.ponder)
        board.push(other)
        t0 = time.monotonic()
        r3 = engine.play(board, limit, ponder=True, game="g")
        assert r3.move in board.legal_moves
        assert time.monotonic() - t0 < 2.5
    finally:
        engine.quit()


def test_stop_right_after_go_ponder_is_not_lost():
    """A "stop" sent immediately after "go ponder" (what python-chess does when the opponent
    replies at once with a move other than the pondered one) must still produce a bestmove.
    The engine used to clear its stop flag after printing the previous bestmove, which could
    wipe out that stop and leave the ponder search running forever."""
    import subprocess

    # Sharing one core with the engine makes the race likely: the engine's search thread is
    # preempted between printing bestmove and its next step while we react to the bestmove.
    try:
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    except (AttributeError, OSError):
        pass
    p = subprocess.Popen([str(ENGINE)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)

    def send(s):
        p.stdin.write(s + "\n")
        p.stdin.flush()

    def wait_for(prefix, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            line = p.stdout.readline()
            if not line or line.startswith(prefix):
                return line
        return None

    try:
        send("uci")
        assert wait_for("uciok", 5)
        moves = "e2e4 e7e5"
        for _ in range(8):
            send(f"position startpos moves {moves}")
            send("go wtime 200 btime 200 winc 10 binc 10")
            bm = wait_for("bestmove", 5)
            assert bm and bm.split()[1] != "0000"
            # The pondered position, then "stop" in the very same write as "go ponder".
            send(f"position startpos moves {moves} {bm.split()[1]} {bm.split()[3]}\ngo ponder wtime 200 btime 200 winc 10 binc 10\nstop")
            assert wait_for("bestmove", 3), "engine kept pondering after stop"
        send("quit")
        assert p.wait(timeout=5) == 0
    finally:
        p.kill()
