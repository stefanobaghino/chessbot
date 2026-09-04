#!/usr/bin/env bash
# Release the current main HEAD on purpose: validate it, sign a tag with the next
# patch version, push main and the tag, and publish a GitHub Release whose notes
# cover every commit since the previous release.
# Usage: scripts/release.sh
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="stefanobaghino/chessbot"
LOGDIR="$ROOT/ci/logs"; mkdir -p "$LOGDIR"
exec 9>"$ROOT/ci/lock"
flock -n 9 || { echo "release: another release is in progress"; exit 1; }
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
cd "$ROOT"
branch=$(git rev-parse --abbrev-ref HEAD)
[ "$branch" = "main" ] || { echo "release: not on main ($branch)"; exit 1; }
git diff --quiet && git diff --cached --quiet || { echo "release: uncommitted changes in the working tree"; exit 1; }
last_tag=$(git tag --list 'v*' --sort=-v:refname | head -1)
[ -n "$last_tag" ] || { echo "release: no release tag found"; exit 1; }
c=$(git rev-parse HEAD)
existing=$(git tag --points-at "$c" 'v*')
[ -z "$existing" ] || { echo "release: HEAD is already released as $existing"; exit 0; }
[ -n "$(git rev-list "$last_tag..$c")" ] || { echo "release: nothing new since $last_tag"; exit 0; }
IFS='.' read -r major minor patch <<< "${last_tag#v}"
version="v$major.$minor.$((patch + 1))"
log="$LOGDIR/$version-${c:0:8}.log"
echo "release: $c -> $version (log: $log)"
work=$(mktemp -d "${TMPDIR:-/tmp}/chessbot-release.XXXXXX")
{
  echo "commit $c"; git log --oneline "$last_tag..$c"; echo
  mkdir -p "$work/src" && git archive "$c" | tar -x -C "$work/src"
  "$ROOT/scripts/validate.sh" "$work/src" "$version" \
    && tarball=$("$ROOT/scripts/package.sh" "$work/src" "$version" "$work") \
    && git tag -s "$version" -m "$version" "$c" \
    && git push origin "$c:refs/heads/main" "refs/tags/$version" \
    && notes="$work/notes.md" && "$ROOT/scripts/release_notes.sh" "$last_tag..$c" > "$notes" \
    && gh release create "$version" "$tarball" "$work/SHA256SUMS" --repo "$REPO" --title "$version" --notes-file "$notes" \
    && echo "release: published $version"
} > "$log" 2>&1
status=$?
rm -rf "$work"
if [ $status -ne 0 ]; then
  echo "release: FAILED $version for $c (see $log)"; echo "$version $c FAILED" > "$ROOT/ci/last_result"; exit 1
fi
echo "$version $c ok" > "$ROOT/ci/last_result"
echo "release: published $version"
