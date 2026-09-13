#!/usr/bin/env python3
"""Summarise the bot's results ledger (the RESULTS_LOG file written per finished game, see #64).

Usage: results.py [results.tsv] [--by version|day|speed] [--since 2026-09-12T18:37]

One row per group in first-seen order: games, wins-losses-draws, score, the average opponent
rating and the bot's own rating, first and last game of the group (pre-game ratings, so the
last one shows the effect of the games before it).
"""
import argparse
import os
from pathlib import Path

COLS = ("time", "version", "game", "speed", "clock", "rated", "color", "opponent", "opp_rating", "my_rating",
        "result", "status", "plies")
POINTS = {"win": 1.0, "draw": 0.5, "loss": 0.0}


def default_path() -> str:
    home = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local/state")
    return os.environ.get("RESULTS_LOG") or str(Path(home) / "chessbot/results.tsv")


def read(path: str, since: str | None = None) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = dict(zip(COLS, line.rstrip("\n").split("\t")))
            if since is None or row["time"] >= since:
                rows.append(row)
    return rows


def summarise(rows: list[dict], by: str = "version") -> list[dict]:
    key = {"version": lambda r: r["version"], "day": lambda r: r["time"][:10], "speed": lambda r: r["speed"]}[by]
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(key(r), []).append(r)
    out = []
    for name, rs in groups.items():
        counted = [r for r in rs if r["result"] in POINTS]
        opp = [int(r["opp_rating"]) for r in rs if r["opp_rating"].lstrip("-").isdigit()]
        mine = [int(r["my_rating"]) for r in rs if r["my_rating"].lstrip("-").isdigit()]
        out.append({
            by: name, "games": len(rs),
            "wins": sum(r["result"] == "win" for r in rs), "losses": sum(r["result"] == "loss" for r in rs),
            "draws": sum(r["result"] == "draw" for r in rs),
            "score": sum(POINTS[r["result"]] for r in counted) / len(counted) if counted else None,
            "avg_opp": sum(opp) / len(opp) if opp else None,
            "rating_first": mine[0] if mine else None, "rating_last": mine[-1] if mine else None,
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default=default_path())
    ap.add_argument("--by", choices=("version", "day", "speed"), default="version")
    ap.add_argument("--since", default=None, help="ISO time; earlier games are left out")
    args = ap.parse_args()
    table = summarise(read(args.path, args.since), args.by)
    print(f"{args.by:16s} {'games':>5s} {'W-L-D':>11s} {'score':>6s} {'avg opp':>8s} {'own rating':>13s}")
    for t in table:
        score = "-" if t["score"] is None else f"{100 * t['score']:.0f}%"
        opp = "-" if t["avg_opp"] is None else f"{t['avg_opp']:.0f}"
        own = "-" if t["rating_first"] is None else f"{t['rating_first']} -> {t['rating_last']}"
        print(f"{t[args.by]:16s} {t['games']:5d} {t['wins']:3d}-{t['losses']:3d}-{t['draws']:3d} {score:>6s} {opp:>8s} {own:>13s}")


if __name__ == "__main__":
    main()
