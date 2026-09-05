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
