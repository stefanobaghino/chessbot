#!/usr/bin/env bash
# Generate self-play games for NNUE training data. By default the fast HCE of the
# nnue2 build; set ENGINE to another binary and ENGINE_OPTS to its fastchess options
# (empty for the current NNUE engine, see #51). Games are pinned to SPAR_CPUS.
# Usage: scripts/selfplay_gen.sh <games> <nodes> <out.pgn> [concurrency]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GAMES="$1"; NODES="$2"; OUT="$3"; CONC="${4:-2}"
ENGINE="${ENGINE:-$ROOT/matches/bin/nnue2}"
ENGINE_OPTS="${ENGINE_OPTS-option.UseNNUE=false}"
SPAR_CPUS="${SPAR_CPUS:-2-3}"
export PATH="$HOME/.local/bin:$PATH"
nice taskset -c "$SPAR_CPUS" fastchess \
  -engine cmd="$ENGINE" name=a $ENGINE_OPTS \
  -engine cmd="$ENGINE" name=b $ENGINE_OPTS \
  -each tc=inf nodes="$NODES" option.Hash=16 \
  -openings file="$HOME/tools/books/UHO_Lichess_4852_v1.epd" format=epd order=random \
  -rounds "$GAMES" -games 1 -concurrency "$CONC" -pgnout file="$OUT" -report penta=false \
  > "${OUT%.pgn}.log" 2>&1
