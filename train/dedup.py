"""Merge npz datasets and drop duplicate positions (same pieces and side to move).

Usage: dedup.py out.npz in1.npz [in2.npz ...] [--exclude held.npz]

The first occurrence of a position is kept, so list the best-labelled files first. Training
on a list that repeats files makes the random validation split leak into the training set
and hides overfitting; the merged file has each position once. Positions that also occur in
the --exclude file (a held-out set, see train.py --holdout) are dropped, so the held-out loss
is measured on positions the net never trained on.

Positions are compared through a 64-bit hash of their 64 piece codes and side to move, so
the merge needs about 150 bytes per input position instead of the 500 that sorting the raw
65-byte keys took (3 GB for 6.3M positions, see #63). Two distinct positions share a hash
with probability 2^-64, so a merge of ten million positions loses about one in a million
runs one position to a collision.
"""
import argparse

import numpy as np

SEED = np.uint64(0x9E3779B97F4A7C15)


def mix(h: np.ndarray) -> np.ndarray:
    """splitmix64's finaliser, a bijection that spreads every input bit over the output."""
    h ^= h >> np.uint64(30)
    h *= np.uint64(0xBF58476D1CE4E5B9)
    h ^= h >> np.uint64(27)
    h *= np.uint64(0x94D049BB133111EB)
    h ^= h >> np.uint64(31)
    return h


def keys(pieces: np.ndarray, stm: np.ndarray) -> np.ndarray:
    """One 64-bit key per position from its 64 piece codes and the side to move.

    Each of the eight 8-square words is folded into an already mixed state, so two positions
    only share a key by chance: a word xored into a raw seed would let a piece on a1 and the
    side to move cancel out.
    """
    words = np.ascontiguousarray(pieces, dtype=np.uint8).reshape(-1, 64).view(np.uint64)
    h = mix(np.asarray(stm, dtype=np.uint64) + SEED)
    for i in range(8):
        h = mix(h ^ words[:, i])
    return h


def unique_first(key: np.ndarray, exclude: np.ndarray | None = None) -> np.ndarray:
    """Sorted indices of the first occurrence of each key, minus those found in exclude."""
    _, first = np.unique(key, return_index=True)
    keep = np.sort(first)
    if exclude is not None:
        keep = keep[~np.isin(key[keep], exclude)]
    return keep


def dedup(pieces: np.ndarray, stm: np.ndarray, score: np.ndarray, exclude: tuple[np.ndarray, np.ndarray] | None = None):
    keep = unique_first(keys(pieces, stm), None if exclude is None else keys(*exclude))
    return pieces[keep], stm[keep], score[keep]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--exclude", default=None, help="npz whose positions are removed from the output")
    args = ap.parse_args()
    # Only the keys of every input are held at once; the rows are gathered file by file.
    parts = [np.load(p) for p in args.inputs]
    key = np.concatenate([keys(d["pieces"], d["stm"]) for d in parts])
    n = len(key)
    exclude = None
    if args.exclude:
        held = np.load(args.exclude)
        exclude = keys(held["pieces"], held["stm"])
    keep = unique_first(key, exclude)
    del key, exclude
    pieces, stm, score = [], [], []
    start = 0
    for d in parts:
        s = d["stm"]
        rows = keep[(keep >= start) & (keep < start + len(s))] - start
        pieces.append(d["pieces"][rows])
        stm.append(s[rows])
        score.append(d["score"][rows])
        start += len(s)
    del parts, keep
    np.savez(args.out, pieces=np.concatenate(pieces), stm=np.concatenate(stm), score=np.concatenate(score))
    print(f"{args.out}: {sum(len(s) for s in score)} unique of {n} positions" + (f", {args.exclude} excluded" if args.exclude else ""))


if __name__ == "__main__":
    main()
