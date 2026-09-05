#!/usr/bin/env bash
# Validate a source tree: engine build + unit tests + bench + NNUE self-check,
# Python lint + tests. Usage: scripts/validate.sh <srcdir> <version>
set -euo pipefail
SRC="$1"; VERSION="$2"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
export PATH="$HOME/.cargo/bin:$PATH"
export CHESSBOT_VERSION="$VERSION"
# Do not pile onto a memory-starved host (it wedged twice on 2026-09-05, #20): wait up to
# 10 min for VALIDATE_MIN_MEM_MB of MemAvailable, then give up.
min_mb="${VALIDATE_MIN_MEM_MB:-1500}"
for ((i = 0; i < 60; i++)); do
  avail=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1024 ))
  (( avail >= min_mb )) && break
  (( i == 0 )) && echo "validate: waiting for memory ($avail MB available, need $min_mb MB)"
  sleep 10
done
(( avail >= min_mb )) || { echo "validate: only $avail MB available after 10 min; not starting"; exit 1; }
# One target dir for every validated tree, so dependencies are compiled once, not per run.
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/ci/target}"
cd "$SRC/engine"
cargo build --release 2>&1 | tail -1
mkdir -p target/release && cp "$CARGO_TARGET_DIR/release/chessbot-engine" target/release/  # where package.sh looks
BIN="$SRC/engine/target/release/chessbot-engine"
# VALIDATE_SKIP_TESTS=1 (set by release.sh when the pre-commit hook already validated this
# exact tree) keeps the build, bench, self-check, banner and lint but skips the test suites.
skip_tests="${VALIDATE_SKIP_TESTS:-0}"
[ "$skip_tests" = 1 ] && echo "validate: tests skipped, this tree passed the pre-commit hook"
[ "$skip_tests" = 1 ] || cargo test --release 2>&1 | grep -E "^test result" | grep -q " 0 failed" || { cargo test --release 2>&1 | tail -20; exit 1; }
out=$("$BIN" bench 10); echo "$out" | grep -q "^bench: [1-9]" || { echo "bench failed: $out"; exit 1; }
# No `tee /dev/stderr` here: when stderr is a log file, tee reopens it with truncation and
# wipes everything written before this point (the v0.1.24-v0.1.26 release logs).
check=$("$BIN" selfcheck); echo "$check"; echo "$check" | grep -qE "selfcheck: .* 0 mismatches|no network loaded"
banner=$(printf 'uci\nquit\n' | "$BIN")
echo "$banner" | grep -q "^id name chessbot-engine $VERSION " || { echo "bad banner: $banner"; exit 1; }
cd "$SRC"
"$ROOT/.venv/bin/ruff" check bot tests scripts train
[ "$skip_tests" = 1 ] || PYTHONPATH="$SRC" "$ROOT/.venv/bin/pytest" -q -p no:cacheprovider 2>&1 | tail -1
echo "validate: ok ($VERSION)"
