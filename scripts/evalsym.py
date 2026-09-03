"""Check static eval symmetry (colour flip) on positions from a PGN."""
import subprocess
import sys

import chess
import chess.pgn

eng = sys.argv[1]; pgn = sys.argv[2]
def ev(fen):
    out = subprocess.run([eng], input=f"position fen {fen}\neval\nquit\n", capture_output=True, text=True, check=True).stdout
    return int(next(l for l in out.splitlines() if l.startswith("eval ")).split()[1])
bad = 0; n = 0
with open(pgn) as f:
    while n < 300:
        g = chess.pgn.read_game(f)
        if g is None: break
        b = g.board()
        for i, mv in enumerate(g.mainline_moves()):
            b.push(mv)
            if i % 17 != 5: continue
            n += 1
            a = ev(b.fen()); m = ev(b.mirror().fen())
            if a != m:
                bad += 1
                if bad <= 5: print("asym", a, m, b.fen())
print(f"checked {n} positions, asymmetric {bad}")
