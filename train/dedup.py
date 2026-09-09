"""Merge npz datasets and drop duplicate positions (same pieces and side to move).

Usage: dedup.py out.npz in1.npz [in2.npz ...]

The first occurrence of a position is kept, so list the best-labelled files first. Training
on a list that repeats files makes the random validation split leak into the training set
and hides overfitting; the merged file has each position once.
"""
import sys

import numpy as np


def dedup(pieces: np.ndarray, stm: np.ndarray, score: np.ndarray):
    key = np.concatenate([pieces, stm[:, None]], axis=1).view(np.dtype((np.void, 65))).reshape(-1)
    _, first = np.unique(key, return_index=True)
    keep = np.sort(first)
    return pieces[keep], stm[keep], score[keep]


def main(out: str, paths: list[str]) -> None:
    parts = [np.load(p) for p in paths]
    pieces = np.concatenate([d["pieces"] for d in parts])
    stm = np.concatenate([d["stm"] for d in parts])
    score = np.concatenate([d["score"] for d in parts])
    n = len(score)
    pieces, stm, score = dedup(pieces, stm, score)
    np.savez(out, pieces=pieces, stm=stm, score=score)
    print(f"{out}: {len(score)} unique of {n} positions")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
