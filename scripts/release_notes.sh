#!/usr/bin/env bash
# Release notes for a commit range: any "MANUAL:" lines from the commit messages
# hoisted to the top, then each commit's message, oldest first.
# Usage: scripts/release_notes.sh <range>   (e.g. v0.1.19..HEAD)
set -euo pipefail
msgs=$(git log --reverse --format='### %s%n%n%b' "$1")
manual=$(printf '%s\n' "$msgs" | grep -E '^MANUAL:' || true)
rest=$(printf '%s\n' "$msgs" | grep -vE '^MANUAL:' || true)
[ -n "$manual" ] && printf '%s\n\n' "$manual"
printf '%s\n' "$rest"
