#!/usr/bin/env bash
# Self-play A/B match at fixed nodes per move (immune to CPU contention), or timed when
# TC is set (TC=10+0.1 ...; nodes are then ignored and background cgroups are frozen).
# Usage: scripts/spar.sh <new_binary> <old_binary> [games] [nodes] [name] [concurrency] [extra -each options]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NEW="$1"; OLD="$2"
GAMES="${3:-200}"
NODES="${4:-40000}"
NAME="${5:-spar}"
CONC="${6:-2}"
EXTRA="${7:-}"
BOOK="$HOME/tools/books/UHO_Lichess_4852_v1.epd"
OUT="$ROOT/matches/${NAME}"
ROUNDS=$(( (GAMES + 1) / 2 ))
export PATH="$HOME/.local/bin:$PATH"
# Keep off the live bot's cores (0-1); see README "Sharing the machine with the live bot".
SPAR_CPUS="${SPAR_CPUS:-2-3}"
if [ -n "${TC:-}" ]; then
  LIMIT="tc=$TC"
  # Timed games are sensitive to contention: pause background jobs sharing the sparring cores.
  FROZEN=()
  for g in quiet train; do
    if [ -w /sys/fs/cgroup/$g/cgroup.freeze ] && [ "$(cat /sys/fs/cgroup/$g/cgroup.freeze)" = 0 ]; then
      echo 1 > /sys/fs/cgroup/$g/cgroup.freeze && FROZEN+=("$g")
    fi
  done
  thaw() { for g in "${FROZEN[@]}"; do echo 0 > /sys/fs/cgroup/$g/cgroup.freeze; done; }
  trap thaw EXIT
else
  LIMIT="tc=inf nodes=$NODES"
fi
nice taskset -c "$SPAR_CPUS" fastchess \
  -engine cmd="$NEW" name=new \
  -engine cmd="$OLD" name=old \
  -each $LIMIT option.Hash=${HASH:-32} option.Threads=1 $EXTRA \
  -openings file="$BOOK" format=epd order=random \
  -rounds "$ROUNDS" -games 2 -repeat -concurrency "$CONC" \
  -pgnout file="$OUT.pgn" -ratinginterval 20 -report penta=false \
  2>&1 | tee "$OUT.log"
