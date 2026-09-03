#!/usr/bin/env bash
# Local CI/CD: validate every commit on main since the last release tag, in order,
# and turn each passing commit into a signed tag, a push and a GitHub Release.
# Stops at the first failing commit (fix forward with a new commit).
# Usage: scripts/ci.sh            (normally launched by scripts/hooks/post-commit)
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="stefanobaghino/chessbot"
LOGDIR="$ROOT/ci/logs"; mkdir -p "$LOGDIR"
exec 9>"$ROOT/ci/lock"
flock 9
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
cd "$ROOT"
branch=$(git rev-parse --abbrev-ref HEAD)
[ "$branch" = "main" ] || { echo "ci: not on main ($branch), nothing to do"; exit 0; }
last_tag=$(git tag --list 'v*' --sort=-v:refname | head -1)
[ -n "$last_tag" ] || { echo "ci: no release tag found"; exit 1; }
commits=$(git rev-list --reverse "$last_tag..HEAD")
[ -n "$commits" ] || { echo "ci: nothing new since $last_tag"; exit 0; }
IFS='.' read -r major minor patch <<< "${last_tag#v}"
for c in $commits; do
  patch=$((patch + 1)); version="v$major.$minor.$patch"
  log="$LOGDIR/$version-${c:0:8}.log"
  echo "ci: $c -> $version (log: $log)"
  work=$(mktemp -d "${TMPDIR:-/tmp}/chessbot-ci.XXXXXX")
  {
    echo "commit $c"; git log -1 --format=%B "$c"; echo
    mkdir -p "$work/src" && git archive "$c" | tar -x -C "$work/src"
    "$ROOT/scripts/validate.sh" "$work/src" "$version" \
      && tarball=$("$ROOT/scripts/package.sh" "$work/src" "$version" "$work") \
      && git tag -s "$version" -m "$version" "$c" \
      && git push origin "$c:refs/heads/main" "refs/tags/$version" \
      && notes="$work/notes.md" && "$ROOT/scripts/release_notes.sh" "$c" > "$notes" \
      && gh release create "$version" "$tarball" "$work/SHA256SUMS" --repo "$REPO" --title "$version" --notes-file "$notes" \
      && echo "ci: released $version"
  } > "$log" 2>&1
  status=$?
  rm -rf "$work"
  if [ $status -ne 0 ]; then
    echo "ci: FAILED $version for $c (see $log)"; echo "$version $c FAILED" > "$ROOT/ci/last_result"; exit 1
  fi
  echo "$version $c ok" > "$ROOT/ci/last_result"
  echo "ci: released $version"
done
