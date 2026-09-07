#!/usr/bin/env python3
"""Resumable SPSA tuning of the engine's search margins (#39).

Usage: scripts/spsa.py <tune-build> <state.json> [--tune a,b,c] [--iterations N]
       [--games 8] [--nodes 20000] [--window 9-21] [--cpus 2-3]

The engine must be built with `--features tune`; its `uci` output lists every parameter as
`info string tunable <name> <default> <min> <max> <step>`. Each iteration perturbs the
current values by +/- c_k * step per parameter, plays a paired fixed-node match between the
two perturbed sets with fastchess, and moves the values along the score difference. The
state (values, iteration, history) is written to <state.json> after every iteration; run
the same command again to resume. With --window H1-H2 an iteration is only started when it
is expected to finish inside that daily window (estimate: the previous iteration's
duration); the process exits with status 3 otherwise. Exit 0 when --iterations is reached.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import re
import subprocess
import sys
import time

BOOK = os.path.expanduser("~/tools/books/UHO_Lichess_4852_v1.epd")
ALPHA, GAMMA, A_FRAC = 0.602, 0.101, 0.1  # Spall's standard exponents; A = A_FRAC * iterations


def read_tunables(engine: str) -> dict[str, dict]:
    out = subprocess.run([engine], input="uci\nquit\n", capture_output=True, text=True, timeout=30, check=False).stdout
    table = {}
    for m in re.finditer(r"info string tunable (\S+) (-?\d+) (-?\d+) (-?\d+) (\d+)", out):
        name, default, lo, hi, step = m.group(1), *map(int, m.groups()[1:])
        table[name] = {"default": default, "min": lo, "max": hi, "step": step}
    if not table:
        sys.exit(f"{engine} lists no tunables; build it with --features tune")
    return table


def coefficients(k: int, iterations: int) -> tuple[float, float]:
    """(a_k, c_k) multipliers for iteration k (0-based): c_k scales the step, a_k the move."""
    c_k = 1.0 / (k + 1) ** GAMMA
    a_k = 2.0 / (A_FRAC * iterations + k + 1) ** ALPHA
    return a_k, c_k


def perturbed(values: dict[str, float], table: dict[str, dict], delta: dict[str, int], c_k: float, sign: int) -> dict[str, int]:
    out = {}
    for name, v in values.items():
        t = table[name]
        out[name] = round(min(t["max"], max(t["min"], v + sign * c_k * t["step"] * delta[name])))
    return out


def parse_points(log: str) -> tuple[float, int] | None:
    """Points of the first engine and number of games from fastchess's final report."""
    m = None
    for m in re.finditer(r"Games: (\d+), Wins: \d+, Losses: \d+, Draws: \d+, Points: ([\d.]+)", log):
        pass
    return (float(m.group(2)), int(m.group(1))) if m else None


def update(values: dict[str, float], table: dict[str, dict], delta: dict[str, int], a_k: float, c_k: float, diff: float) -> dict[str, float]:
    """Move every value along its perturbation sign by a_k * step * diff, diff in [-1, 1]."""
    out = {}
    for name, v in values.items():
        t = table[name]
        out[name] = min(t["max"], max(t["min"], v + a_k * t["step"] * diff * delta[name]))
    return out


def play(engine: str, plus: dict[str, int], minus: dict[str, int], games: int, nodes: int, cpus: str, log_path: str) -> tuple[float, int]:
    def opts(p: dict[str, int]) -> list[str]:
        return [f"option.{k}={v}" for k, v in p.items()]
    cmd = ["nice", "taskset", "-c", cpus, "fastchess",
           "-engine", f"cmd={engine}", "name=plus", *opts(plus),
           "-engine", f"cmd={engine}", "name=minus", *opts(minus),
           "-each", "tc=inf", f"nodes={nodes}", "option.Hash=16",
           "-openings", f"file={BOOK}", "format=epd", "order=random",
           "-rounds", str((games + 1) // 2), "-games", "2", "-repeat", "-concurrency", "2",
           "-report", "penta=false"]
    env = dict(os.environ, PATH=os.path.expanduser("~/.local/bin") + ":" + os.environ.get("PATH", ""))
    out = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False).stdout
    with open(log_path, "a") as f:
        f.write(out)
    r = parse_points(out)
    if r is None:
        sys.exit(f"fastchess produced no result; see {log_path}")
    return r


def window_allows(window: tuple[int, int] | None, est: float, now: dt.datetime | None = None) -> bool:
    if window is None:
        return True
    now = now or dt.datetime.now().astimezone()
    start = now.replace(hour=window[0], minute=0, second=0, microsecond=0)
    end = now.replace(hour=window[1], minute=0, second=0, microsecond=0)
    return start <= now and now + dt.timedelta(seconds=est) <= end


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("engine")
    ap.add_argument("state")
    ap.add_argument("--tune", default="", help="comma-separated parameter names (default: all)")
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--nodes", type=int, default=20000)
    ap.add_argument("--window", default=None)
    ap.add_argument("--cpus", default=os.environ.get("SPAR_CPUS", "2-3"))
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    window = tuple(int(h) for h in args.window.split("-")) if args.window else None
    table = read_tunables(args.engine)
    names = [n for n in args.tune.split(",") if n] or list(table)
    unknown = [n for n in names if n not in table]
    if unknown:
        sys.exit(f"unknown parameters: {', '.join(unknown)}")
    table = {n: table[n] for n in names}
    if os.path.exists(args.state):
        with open(args.state) as f:
            state = json.load(f)
        print(f"resumed from {args.state} at iteration {state['iteration']}", flush=True)
    else:
        state = {"iteration": 0, "values": {n: float(table[n]["default"]) for n in names}, "history": [], "last_seconds": None}
    rng = random.Random(args.seed if args.seed is not None else state["iteration"])
    log_path = os.path.splitext(args.state)[0] + ".fastchess.log"
    while state["iteration"] < args.iterations:
        k = state["iteration"]
        est = state["last_seconds"] or 0.0
        if not window_allows(window, est):
            print(f"iteration {k + 1}/{args.iterations} would not finish inside {args.window} (est {est / 60:.0f} min); stopping", flush=True)
            return 3
        a_k, c_k = coefficients(k, args.iterations)
        delta = {n: rng.choice((-1, 1)) for n in names}
        plus = perturbed(state["values"], table, delta, c_k, +1)
        minus = perturbed(state["values"], table, delta, c_k, -1)
        t0 = time.monotonic()
        points, games = play(args.engine, plus, minus, args.games, args.nodes, args.cpus, log_path)
        diff = (points - (games - points)) / games  # +1: plus won every game, -1: lost every game
        state["values"] = update(state["values"], table, delta, a_k, c_k, diff)
        state["iteration"] = k + 1
        state["last_seconds"] = time.monotonic() - t0
        state["history"].append({"iteration": k + 1, "diff": diff, "games": games})
        tmp = args.state + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f, indent=1)
        os.replace(tmp, args.state)
        current = " ".join(f"{n}={v:.1f}" for n, v in state["values"].items())
        print(f"iteration {k + 1}/{args.iterations} diff {diff:+.3f} {state['last_seconds']:.0f}s {current}", flush=True)
    print("done: " + " ".join(f"{n}={round(v)}" for n, v in state["values"].items()), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
