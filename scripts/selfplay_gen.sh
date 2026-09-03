#!/usr/bin/env bash
# Generate fast HCE self-play games for NNUE training data.
# Usage: scripts/selfplay_gen.sh <games> <nodes> <out.pgn> [concurrency]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GAMES="$1"; NODES="$2"; OUT="$3"; CONC="${4:-2}"
ENGINE="$ROOT/matches/bin/nnue2"
export PATH="$HOME/.local/bin:$PATH"
nice fastchess \
  -engine cmd="$ENGINE" name=a option.UseNNUE=false \
  -engine cmd="$ENGINE" name=b option.UseNNUE=false \
  -each tc=inf nodes="$NODES" option.Hash=16 \
  -openings file="$HOME/tools/books/UHO_Lichess_4852_v1.epd" format=epd order=random \
  -rounds "$GAMES" -games 1 -concurrency "$CONC" -pgnout file="$OUT" -report penta=false \
  > "${OUT%.pgn}.log" 2>&1
