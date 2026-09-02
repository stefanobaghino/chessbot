"""Fast Stockfish relabeller using raw UCI pipes (no python-chess engine layer).

Usage: relabel_fast.py src.jsonl dst.npz [--workers 4] [--depth 5] [--limit N] [--skip N]
Drops positions in check or whose best move is a capture/promotion.
"""
import argparse
import itertools
import subprocess
from multiprocessing import Pool

import chess
import numpy as np

PIECE_CODE = {c: i + 1 for i, c in enumerate("PNBRQK")}
PIECE_CODE.update({c: i + 7 for i, c in enumerate("pnbrqk")})
CLAMP = 3000
_sf = None
_depth = 5


def init(depth):
    global _sf, _depth
    _depth = depth
    _sf = subprocess.Popen(["stockfish"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    _sf.stdin.write("uci\nsetoption name Hash value 16\nisready\n")
    _sf.stdin.flush()
    while _sf.stdout.readline().strip() != "readyok":
        pass


def analyse(fen):
    _sf.stdin.write(f"position fen {fen}\ngo depth {_depth}\n")
    _sf.stdin.flush()
    score = None
    pv = None
    while True:
        line = _sf.stdout.readline()
        if not line:
            return None, None
        if line.startswith("bestmove"):
            return score, pv
        if line.startswith("info") and " score " in line and " pv " in line and " multipv 2" not in line:
            t = line.split()
            i = t.index("score")
            if t[i + 1] == "cp":
                score = int(t[i + 2])
            else:
                score = CLAMP if int(t[i + 2]) > 0 else -CLAMP
            pv = t[t.index("pv") + 1]


def label(lines):
    out = []
    for line in lines:
        i = line.find('"fen":"')
        if i < 0:
            continue
        j = line.find('"', i + 7)
        fen = line[i + 7 : j]
        try:
            board = chess.Board(fen + " 0 1")
        except ValueError:
            continue
        if board.is_check() or not board.is_valid():
            continue
        score, first = analyse(fen + " 0 1")
        if score is None or first is None:
            continue
        try:
            mv = chess.Move.from_uci(first)
        except ValueError:
            continue
        if board.is_capture(mv) or mv.promotion is not None:
            continue
        cp = max(-CLAMP, min(CLAMP, score))
        pieces = np.zeros(64, dtype=np.uint8)
        for sq, pc in board.piece_map().items():
            pieces[sq] = PIECE_CODE[pc.symbol()]
        out.append((pieces, 0 if board.turn == chess.WHITE else 1, cp))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=500)
    args = ap.parse_args()
    P, S, C = [], [], []
    with open(args.src) as f, Pool(args.workers, initializer=init, initargs=(args.depth,)) as pool:
        it = itertools.islice(f, args.skip, args.skip + args.limit if args.limit else None)
        chunks = iter(lambda: list(itertools.islice(it, args.chunk)), [])
        done = 0
        for res in pool.imap_unordered(label, chunks):
            for p, s, c in res:
                P.append(p)
                S.append(s)
                C.append(c)
            done += args.chunk
            if done % 100000 == 0:
                print(f"{done} lines read, {len(C)} labelled", flush=True)
    P = np.stack(P) if P else np.zeros((0, 64), dtype=np.uint8)
    np.savez(args.dst, pieces=P, stm=np.array(S, dtype=np.uint8), score=np.array(C, dtype=np.int16))
    print(f"wrote {len(C)} positions to {args.dst}; mean |cp| {np.abs(np.array(C)).mean():.0f}", flush=True)


if __name__ == "__main__":
    main()
