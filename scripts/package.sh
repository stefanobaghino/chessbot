#!/usr/bin/env bash
# Package a validated source tree (engine already built by validate.sh) as a release tarball.
# Usage: scripts/package.sh <srcdir> <version> <outdir>
# Produces <outdir>/chessbot-<version>-linux-aarch64.tar.gz and <outdir>/SHA256SUMS.
set -euo pipefail
SRC="$1"; VERSION="$2"; OUT="$3"
PKG="chessbot-$VERSION"
STAGE="$OUT/$PKG"
rm -rf "$STAGE"; mkdir -p "$STAGE/bin" "$STAGE/bot"
cp "$SRC/engine/target/release/chessbot-engine" "$STAGE/bin/"
cp "$SRC/bot/__init__.py" "$SRC/bot/lichess_bot.py" "$STAGE/bot/"
cp "$SRC/requirements.txt" "$STAGE/"
echo "$VERSION" > "$STAGE/VERSION"
TARBALL="$PKG-linux-aarch64.tar.gz"
tar -C "$OUT" -czf "$OUT/$TARBALL" "$PKG"
(cd "$OUT" && sha256sum "$TARBALL" > SHA256SUMS && sha256sum -c SHA256SUMS >/dev/null)
rm -rf "$STAGE"
echo "$OUT/$TARBALL"
