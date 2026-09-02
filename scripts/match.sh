#!/usr/bin/env bash
# Run a fastchess match between the engine and Elo-limited Stockfish.
# Usage: scripts/match.sh [elo] [games] [tc] [concurrency] [name]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ELO="${1:-2000}"
GAMES="${2:-20}"
TC="${3:-10+0.1}"
CONC="${4:-3}"
NAME="${5:-run}"
ENGINE="$ROOT/engine/target/release/chessbot-engine"
BOOK="$HOME/tools/books/UHO_Lichess_4852_v1.epd"
OUT="$ROOT/matches/${NAME}_sf${ELO}_${GAMES}g"
ROUNDS=$(( (GAMES + 1) / 2 ))
export PATH="$HOME/.local/bin:$PATH"
fastchess \
  -engine cmd="$ENGINE" name=chessbot option.Hash=64 \
  -engine cmd=stockfish name="SF_$ELO" option.UCI_LimitStrength=true option.UCI_Elo="$ELO" option.Hash=64 \
  -each tc="$TC" option.Threads=1 \
  -openings file="$BOOK" format=epd order=random \
  -rounds "$ROUNDS" -games 2 -repeat -concurrency "$CONC" \
  -pgnout file="$OUT.pgn" \
  -ratinginterval 10 -report penta=false \
  2>&1 | tee "$OUT.log"
