#!/usr/bin/env bash
# Build a release tarball from a git tag and publish it as a GitHub Release.
# Usage: scripts/release.sh vX.Y.Z releases/vX.Y.Z.md
# The tag must exist, be pushed, and match engine/Cargo.toml's version.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TAG="$1"; NOTES="$2"
VERSION="${TAG#v}"
export PATH="$HOME/.cargo/bin:$PATH"
git -C "$ROOT" rev-parse "refs/tags/$TAG" >/dev/null 2>&1 || { echo "tag $TAG not found"; exit 1; }
CRATE_VERSION=$(git -C "$ROOT" show "$TAG:engine/Cargo.toml" | sed -n 's/^version = "\(.*\)"/\1/p' | head -1)
[ "$CRATE_VERSION" = "$VERSION" ] || { echo "Cargo.toml version $CRATE_VERSION != tag $VERSION"; exit 1; }
WORK=$(mktemp -d "${TMPDIR:-/tmp}/chessbot-release.XXXXXX")
trap 'rm -rf "$WORK"' EXIT
SRC="$WORK/src"; mkdir -p "$SRC"
git -C "$ROOT" archive "$TAG" | tar -x -C "$SRC"
(cd "$SRC/engine" && cargo build --release 2>&1 | tail -1)
PKG="chessbot-$TAG"; OUT="$WORK/$PKG"
mkdir -p "$OUT/bin" "$OUT/bot"
cp "$SRC/engine/target/release/chessbot-engine" "$OUT/bin/"
cp "$SRC/bot/__init__.py" "$SRC/bot/lichess_bot.py" "$OUT/bot/"
cp "$SRC/requirements.txt" "$OUT/"
echo "$TAG" > "$OUT/VERSION"
"$OUT/bin/chessbot-engine" bench 10 | tail -1
printf 'uci\nquit\n' | "$OUT/bin/chessbot-engine" | grep "^id name"
TARBALL="$PKG-linux-aarch64.tar.gz"
tar -C "$WORK" -czf "$WORK/$TARBALL" "$PKG"
(cd "$WORK" && sha256sum "$TARBALL" > SHA256SUMS && sha256sum -c SHA256SUMS)
gh release create "$TAG" "$WORK/$TARBALL" "$WORK/SHA256SUMS" --repo stefanobaghino/chessbot --title "$TAG" --notes-file "$NOTES"
echo "published $TAG"
