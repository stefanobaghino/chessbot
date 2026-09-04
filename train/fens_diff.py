"""Write the FENs of the input files that are not in the excluded files, shuffled.

Usage: fens_diff.py out.fens [--exclude x.fens ...] [--seed 0] in1.fens [in2.fens ...]
Used to pick the positions of a self-play batch that have not been relabelled yet.
"""
import argparse
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    seen = set()
    for f in args.exclude:
        with open(f) as fh:
            seen.update(line.strip() for line in fh)
    excluded = len(seen)
    lines = []
    for f in args.inputs:
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line and line not in seen:
                    seen.add(line)
                    lines.append(line)
    random.Random(args.seed).shuffle(lines)
    with open(args.out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {len(lines)} positions to {args.out} ({excluded} excluded, duplicates dropped)")


if __name__ == "__main__":
    main()
