#!/usr/bin/env bash
# Validate a source tree: engine build + unit tests + bench + NNUE self-check,
# Python lint + tests. Usage: scripts/validate.sh <srcdir> <version>
set -euo pipefail
SRC="$1"; VERSION="$2"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
export PATH="$HOME/.cargo/bin:$PATH"
export CHESSBOT_VERSION="$VERSION"
cd "$SRC/engine"
cargo build --release 2>&1 | tail -1
cargo test --release 2>&1 | grep -E "^test result" | grep -q " 0 failed" || { cargo test --release 2>&1 | tail -20; exit 1; }
BIN="$SRC/engine/target/release/chessbot-engine"
out=$("$BIN" bench 10); echo "$out" | grep -q "^bench: [1-9]" || { echo "bench failed: $out"; exit 1; }
"$BIN" selfcheck | tee /dev/stderr | grep -qE "selfcheck: .* 0 mismatches|no network loaded"
banner=$(printf 'uci\nquit\n' | "$BIN")
echo "$banner" | grep -q "^id name chessbot-engine $VERSION " || { echo "bad banner: $banner"; exit 1; }
cd "$SRC"
"$ROOT/.venv/bin/ruff" check bot tests scripts train
PYTHONPATH="$SRC" "$ROOT/.venv/bin/pytest" -q -p no:cacheprovider 2>&1 | tail -1
echo "validate: ok ($VERSION)"
