#!/usr/bin/env python3
"""Combine the final results of several fastchess logs into one Elo estimate.

Usage: scripts/sparsum.py matches/a.log [matches/b.log ...]

Each log's last "Games: N, Wins: W, Losses: L, Draws: D" line is summed; the output is the
combined score, the Elo difference of "new" over "old" and its 95 percent error.
"""
import math
import re
import sys

LINE = re.compile(r"Games: (\d+), Wins: (\d+), Losses: (\d+), Draws: (\d+)")


def final_wld(text: str) -> tuple[int, int, int]:
    """W, L, D of the last result line in a fastchess log (0, 0, 0 if none)."""
    found = LINE.findall(text)
    if not found:
        return 0, 0, 0
    _, w, l, d = map(int, found[-1])
    return w, l, d


def elo(w: int, l: int, d: int) -> tuple[float, float, float]:
    """Score, Elo and 95 percent error for the given W/L/D counts."""
    n = w + l + d
    if n == 0:
        return 0.5, 0.0, float("inf")
    s = (w + d / 2) / n
    if s <= 0 or s >= 1:
        return s, math.copysign(float("inf"), s - 0.5), float("inf")
    var = (w * (1 - s) ** 2 + l * s**2 + d * (0.5 - s) ** 2) / n
    se = math.sqrt(var / n)
    to_elo = 400 / math.log(10) / (s * (1 - s))
    return s, -400 * math.log10(1 / s - 1), 1.96 * se * to_elo


def main(paths: list[str]) -> None:
    if not paths:
        sys.exit(__doc__)
    w = l = d = 0
    for path in paths:
        with open(path) as f:
            pw, pl, pd = final_wld(f.read())
        print(f"{path}: {pw + pl + pd} games W{pw} L{pl} D{pd}")
        w, l, d = w + pw, l + pl, d + pd
    s, e, err = elo(w, l, d)
    print(f"combined: {w + l + d} games W{w} L{l} D{d} score {100 * s:.2f}% Elo {e + 0.0:+.1f} +/- {err:.1f}")


if __name__ == "__main__":
    main(sys.argv[1:])
