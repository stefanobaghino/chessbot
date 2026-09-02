"""Relabel positions from Lichess eval-DB JSONL with a shallow Stockfish search.

Usage: relabel.py src.jsonl dst.npz [--workers 4] [--depth 6] [--limit N] [--skip N]
Positions in check, or whose best move is a capture/promotion, are dropped.
Output matches prepare.py: pieces uint8 [N,64], stm uint8 [N], score int16 (stm pov).
"""
import argparse
import itertools
import sys
from multiprocessing import Pool

import chess
import chess.engine
import numpy as np

PIECE_CODE = {c: i + 1 for i, c in enumerate("PNBRQK")}
PIECE_CODE.update({c: i + 7 for i, c in enumerate("pnbrqk")})
CLAMP = 3000
_engine = None
_depth = 6


def init(depth):
    global _engine, _depth
    _depth = depth
    _engine = chess.engine.SimpleEngine.popen_uci("stockfish")
    _engine.configure({"Hash": 16, "Threads": 1})


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
        try:
            info = _engine.analyse(board, chess.engine.Limit(depth=_depth))
        except chess.engine.EngineError:
            continue
        pv = info.get("pv")
        if not pv:
            continue
        mv = pv[0]
        if board.is_capture(mv) or mv.promotion is not None:
            continue
        cp = info["score"].pov(board.turn).score(mate_score=CLAMP)
        cp = max(-CLAMP, min(CLAMP, cp))
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
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=200)
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
    print(f"wrote {len(C)} positions to {args.dst}; mean |cp| {np.abs(np.array(C)).mean():.0f}")


if __name__ == "__main__":
    main()
