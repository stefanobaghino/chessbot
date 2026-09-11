"""Merge npz datasets and drop duplicate positions (same pieces and side to move).

Usage: dedup.py out.npz in1.npz [in2.npz ...] [--exclude held.npz]

The first occurrence of a position is kept, so list the best-labelled files first. Training
on a list that repeats files makes the random validation split leak into the training set
and hides overfitting; the merged file has each position once. Positions that also occur in
the --exclude file (a held-out set, see train.py --holdout) are dropped, so the held-out loss
is measured on positions the net never trained on.
"""
import argparse

import numpy as np


def keys(pieces: np.ndarray, stm: np.ndarray) -> np.ndarray:
    """One 65-byte key per position: the 64 piece codes and the side to move."""
    return np.concatenate([pieces, stm[:, None]], axis=1).view(np.dtype((np.void, 65))).reshape(-1)


def dedup(pieces: np.ndarray, stm: np.ndarray, score: np.ndarray, exclude: tuple[np.ndarray, np.ndarray] | None = None):
    key = keys(pieces, stm)
    _, first = np.unique(key, return_index=True)
    keep = np.sort(first)
    if exclude is not None:
        keep = keep[~np.isin(key[keep], keys(*exclude))]
    return pieces[keep], stm[keep], score[keep]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--exclude", default=None, help="npz whose positions are removed from the output")
    args = ap.parse_args()
    parts = [np.load(p) for p in args.inputs]
    pieces = np.concatenate([d["pieces"] for d in parts])
    stm = np.concatenate([d["stm"] for d in parts])
    score = np.concatenate([d["score"] for d in parts])
    del parts
    n = len(score)
    exclude = None
    if args.exclude:
        held = np.load(args.exclude)
        exclude = (held["pieces"], held["stm"])
    pieces, stm, score = dedup(pieces, stm, score, exclude)
    np.savez(args.out, pieces=pieces, stm=stm, score=score)
    print(f"{args.out}: {len(score)} unique of {n} positions" + (f", {args.exclude} excluded" if args.exclude else ""))


if __name__ == "__main__":
    main()
