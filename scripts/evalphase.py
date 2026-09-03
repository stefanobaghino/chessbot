"""Eval correlation with Stockfish split by number of pieces on the board."""
import subprocess, sys, random
import numpy as np, chess, chess.pgn, chess.engine
eng, pgn = sys.argv[1], sys.argv[2]
random.seed(1)
fens = []
with open(pgn) as f:
    while len(fens) < 6000:
        g = chess.pgn.read_game(f)
        if g is None: break
        b = g.board()
        for i, mv in enumerate(g.mainline_moves()):
            b.push(mv)
            if i >= 10 and not b.is_check() and random.random() < 0.2:
                fens.append(b.fen())
random.shuffle(fens); fens = fens[:600]
def evals(use):
    cmds = f"setoption name UseNNUE value {str(use).lower()}\n" + "".join(f"position fen {x}\neval\n" for x in fens) + "quit\n"
    out = subprocess.run([eng], input=cmds, capture_output=True, text=True).stdout
    return np.array([int(l.split()[1]) for l in out.splitlines() if l.startswith("eval ")], float)
sf = chess.engine.SimpleEngine.popen_uci("stockfish")
ref = np.array([sf.analyse(chess.Board(x), chess.engine.Limit(depth=8))["score"].pov(chess.Board(x).turn).score(mate_score=3000) for x in fens], float)
sf.quit()
ref = np.clip(ref, -1500, 1500)
pieces = np.array([len(chess.Board(x).piece_map()) for x in fens])
nn = np.clip(evals(True), -1500, 1500); hc = np.clip(evals(False), -1500, 1500)
for lo, hi in ((2, 10), (11, 18), (19, 26), (27, 33)):
    sel = (pieces >= lo) & (pieces <= hi)
    if sel.sum() < 20: continue
    print(f"pieces {lo}-{hi} n={sel.sum():3d}: nnue mae {np.abs(nn[sel]-ref[sel]).mean():4.0f} corr {np.corrcoef(nn[sel],ref[sel])[0,1]:.2f} | hce mae {np.abs(hc[sel]-ref[sel]).mean():4.0f} corr {np.corrcoef(hc[sel],ref[sel])[0,1]:.2f}")
