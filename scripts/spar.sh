#!/usr/bin/env bash
# Self-play A/B match at fixed nodes per move (immune to CPU contention).
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
nice taskset -c "$SPAR_CPUS" fastchess \
  -engine cmd="$NEW" name=new \
  -engine cmd="$OLD" name=old \
  -each tc=inf nodes="$NODES" option.Hash=32 $EXTRA \
  -openings file="$BOOK" format=epd order=random \
  -rounds "$ROUNDS" -games 2 -repeat -concurrency "$CONC" \
  -pgnout file="$OUT.pgn" -ratinginterval 20 -report penta=false \
  2>&1 | tee "$OUT.log"
