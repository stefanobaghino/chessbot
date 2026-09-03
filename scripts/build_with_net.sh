#!/usr/bin/env bash
# Build the engine with a given net embedded and copy the binary to matches/bin/<name>.
# Usage: scripts/build_with_net.sh <net.bin> <name>
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NET="$1"; NAME="$2"
export PATH="$HOME/.cargo/bin:$PATH"
cp "$NET" "$ROOT/engine/nets/default.bin"
(cd "$ROOT/engine" && cargo build --release 2>&1 | grep -E "^error" -A 8 || true)
mkdir -p "$ROOT/matches/bin"
cp "$ROOT/engine/target/release/chessbot-engine" "$ROOT/matches/bin/$NAME"
echo "built matches/bin/$NAME with $(basename "$NET")"
