"""Find the worst move per lost game using Stockfish at fixed depth."""
import sys

import chess
import chess.engine
import chess.pgn

pgn_path = sys.argv[1]
us = sys.argv[2] if len(sys.argv) > 2 else "chessbot"
depth = int(sys.argv[3]) if len(sys.argv) > 3 else 14
engine = chess.engine.SimpleEngine.popen_uci("stockfish")
with open(pgn_path) as f:
    n = 0
    while True:
        game = chess.pgn.read_game(f)
        if game is None:
            break
        n += 1
        white, black = game.headers["White"], game.headers["Black"]
        result = game.headers["Result"]
        our_color = chess.WHITE if white == us else chess.BLACK
        lost = (result == "0-1" and our_color == chess.WHITE) or (result == "1-0" and our_color == chess.BLACK)
        if not lost:
            continue
        board = game.board()
        prev = None
        worst = (0, None, None, None)
        drops = []
        for mv in game.mainline_moves():
            if board.turn == our_color:
                info = engine.analyse(board, chess.engine.Limit(depth=depth))
                before = info["score"].pov(our_color).score(mate_score=10000)
                board.push(mv)
                info = engine.analyse(board, chess.engine.Limit(depth=depth))
                after = info["score"].pov(our_color).score(mate_score=10000)
                drop = before - after
                if drop > worst[0]:
                    worst = (drop, board.fullmove_number, mv.uci(), (before, after))
                if drop > 120 and len(drops) < 4:
                    drops.append((board.fullmove_number, mv.uci(), before, after, board.fen()))
            else:
                board.push(mv)
        plies = len(list(game.mainline_moves()))
        print(f"game {n} ({'W' if our_color else 'B'}) plies={plies}")
        for d in drops:
            print("   ", d)
engine.quit()
