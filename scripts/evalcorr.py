"""Correlate engine static evals (NNUE and HCE) with Stockfish search evals on PGN positions."""
import subprocess, sys, random
import numpy as np, chess, chess.pgn, chess.engine
eng, pgn = sys.argv[1], sys.argv[2]
depth = int(sys.argv[3]) if len(sys.argv) > 3 else 10
random.seed(0)
fens = []
with open(pgn) as f:
    while len(fens) < 4000:
        g = chess.pgn.read_game(f)
        if g is None: break
        b = g.board()
        for i, mv in enumerate(g.mainline_moves()):
            b.push(mv)
            if i >= 10 and not b.is_check() and random.random() < 0.15:
                fens.append(b.fen())
random.shuffle(fens); fens = fens[:400]
def evals(use_nnue):
    cmds = f"setoption name UseNNUE value {str(use_nnue).lower()}\n" + "".join(f"position fen {x}\neval\n" for x in fens) + "quit\n"
    out = subprocess.run([eng], input=cmds, capture_output=True, text=True).stdout
    return np.array([int(l.split()[1]) for l in out.splitlines() if l.startswith("eval ")], dtype=float)
sf = chess.engine.SimpleEngine.popen_uci("stockfish")
ref = []
for x in fens:
    b = chess.Board(x)
    info = sf.analyse(b, chess.engine.Limit(depth=depth))
    ref.append(info["score"].pov(b.turn).score(mate_score=3000))
sf.quit()
ref = np.clip(np.array(ref, dtype=float), -1500, 1500)
for name, use in (("nnue", True), ("hce", False)):
    e = np.clip(evals(use), -1500, 1500)
    r = np.corrcoef(e, ref)[0, 1]
    mae = np.abs(e - ref).mean()
    sign = ((np.sign(e) == np.sign(ref)) | (np.abs(ref) < 30)).mean()
    print(f"{name}: corr {r:.3f} mae {mae:.0f} sign-agree {sign:.2f} (n={len(e)})")
