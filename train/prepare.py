"""Convert Lichess evaluation-DB JSONL into a compact binary dataset.

Output (numpy .npz): pieces uint8 [N,64] (0 empty, 1-6 white PNBRQK, 7-12 black),
stm uint8 [N] (0 white, 1 black), score int16 [N] centipawns from side to move.
"""
import json
import sys
from multiprocessing import Pool

import numpy as np

PIECE_CODE = {c: i + 1 for i, c in enumerate("PNBRQK")}
PIECE_CODE.update({c: i + 7 for i, c in enumerate("pnbrqk")})
CLAMP = 3000


def parse_line(line: str):
    try:
        d = json.loads(line)
    except json.JSONDecodeError:
        return None
    fen = d["fen"].split()
    board_str, stm = fen[0], fen[1]
    best = max(d["evals"], key=lambda e: e["depth"])
    pv = best["pvs"][0]
    if "cp" in pv:
        cp = max(-CLAMP, min(CLAMP, pv["cp"]))
    else:
        mate = pv["mate"]
        cp = CLAMP if mate > 0 else -CLAMP
    if stm == "b":
        cp = -cp
    pieces = np.zeros(64, dtype=np.uint8)
    rank = 7
    file = 0
    for ch in board_str:
        if ch == "/":
            rank -= 1
            file = 0
        elif ch.isdigit():
            file += int(ch)
        else:
            pieces[rank * 8 + file] = PIECE_CODE[ch]
            file += 1
    return pieces, 1 if stm == "b" else 0, cp


def main():
    src, dst = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    with open(src, "rb") as f:
        total = sum(1 for _ in f)
    P = np.zeros((total, 64), dtype=np.uint8)
    S = np.zeros(total, dtype=np.uint8)
    C = np.zeros(total, dtype=np.int16)
    n = 0
    with open(src) as f, Pool(workers) as pool:
        for i, r in enumerate(pool.imap(parse_line, f, chunksize=4096)):
            if r is None:
                continue
            P[n] = r[0]
            S[n] = r[1]
            C[n] = r[2]
            n += 1
            if i % 1_000_000 == 0:
                print(f"{i} lines", flush=True)
    P, S, C = P[:n], S[:n], C[:n]
    np.savez(dst, pieces=P, stm=S, score=C)
    print(f"wrote {len(C)} positions to {dst}; mean |cp| {np.abs(C).mean():.0f}")


if __name__ == "__main__":
    main()
