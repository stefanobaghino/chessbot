#!/usr/bin/env bash
# Relabel a FEN file with Stockfish in resumable chunks, only inside the 09:00-21:00
# window reserved for resource-intensive work.
# Usage: scripts/relabel_chunks.sh <src.fens> <out_prefix> [depth=10] [chunk=50000] [workers=2]
#
# Chunk i (0-based, `chunk` lines each) is written to <out_prefix>_<i>.npz; chunks that
# already exist are skipped, so re-running the same command resumes where it stopped.
# A chunk is only started when it is expected to finish before 21:00 (the estimate is
# the duration of the previous chunk of this run, 40 min before one has completed).
# Exit 0: all chunks done. Exit 3: stopped for the window; run again after 09:00.
# Train on the chunks with train.py's comma-separated list, e.g. $(ls prefix_*.npz | paste -sd,).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$1"; PREFIX="$2"; DEPTH="${3:-10}"; CHUNK="${4:-50000}"; WORKERS="${5:-2}"
CPUS="${SPAR_CPUS:-2-3}"
export PATH="$PATH:/usr/games"  # stockfish lives in /usr/games, which systemd user units do not have on PATH
START_H="${WINDOW_START:-9}"; END_H="${WINDOW_END:-21}"
total=$(wc -l < "$SRC")
chunks=$(( (total + CHUNK - 1) / CHUNK ))
est=2400
echo "$(date +%FT%T) relabel_chunks: $total lines in $chunks chunks of $CHUNK at depth $DEPTH, $WORKERS workers on cpus $CPUS"
for ((i = 0; i < chunks; i++)); do
  out="${PREFIX}_$i.npz"
  [ -s "$out" ] && continue
  now=$(date +%s)
  window_start=$(date -d "today ${START_H}:00" +%s)
  window_end=$(date -d "today ${END_H}:00" +%s)
  if (( now < window_start || now + est > window_end )); then
    echo "$(date +%FT%T) relabel_chunks: chunk $i would not finish inside ${START_H}:00-${END_H}:00 (est $((est / 60)) min); stopping, $((chunks - i)) chunks left"
    exit 3
  fi
  echo "$(date +%FT%T) relabel_chunks: chunk $i/$chunks (lines $((i * CHUNK))-$(( (i + 1) * CHUNK - 1 )))"
  t0=$now
  nice taskset -c "$CPUS" "$ROOT/.venv/bin/python" "$ROOT/train/relabel_fast.py" "$SRC" "$out.tmp.npz" \
    --workers "$WORKERS" --depth "$DEPTH" --skip $((i * CHUNK)) --limit "$CHUNK" | tail -1
  mv "$out.tmp.npz" "$out"
  est=$(( $(date +%s) - t0 ))
  echo "$(date +%FT%T) relabel_chunks: chunk $i done in $((est / 60)) min"
done
echo "$(date +%FT%T) relabel_chunks: all $chunks chunks done"
