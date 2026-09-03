"""Extract training positions from PGN games: one FEN per line, skipping the
first plies, positions in check, and the last plies before a decisive end."""
import sys
import chess.pgn

src, dst = sys.argv[1], sys.argv[2]
skip = int(sys.argv[3]) if len(sys.argv) > 3 else 8
n = 0
with open(src) as f, open(dst, "w") as out:
    while True:
        g = chess.pgn.read_game(f)
        if g is None:
            break
        b = g.board()
        moves = list(g.mainline_moves())
        for i, mv in enumerate(moves):
            b.push(mv)
            if i < skip or i >= len(moves) - 2:
                continue
            if b.is_check() or b.is_game_over():
                continue
            out.write(b.fen() + "\n")
            n += 1
print(f"wrote {n} positions to {dst}")
