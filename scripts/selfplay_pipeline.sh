#!/usr/bin/env bash
# Generate self-play games in batches and relabel each batch with Stockfish.
# Usage: scripts/selfplay_pipeline.sh <batches> <games_per_batch> <nodes> <prefix> [concurrency]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BATCHES="$1"; GAMES="$2"; NODES="$3"; PREFIX="$4"; CONC="${5:-2}"
for i in $(seq 1 "$BATCHES"); do
  PGN="$ROOT/data/${PREFIX}_$i.pgn"
  "$ROOT/scripts/selfplay_gen.sh" "$GAMES" "$NODES" "$PGN" "$CONC"
  "$ROOT/.venv/bin/python" "$ROOT/train/pgn_to_fens.py" "$PGN" "${PGN%.pgn}.fens" 8
  nice "$ROOT/.venv/bin/python" "$ROOT/train/relabel_fast.py" "${PGN%.pgn}.fens" "${PGN%.pgn}.npz" --workers "$CONC" --depth 6 | tail -1
  echo "batch $i done: $(date +%H:%M)"
done
