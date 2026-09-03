#!/usr/bin/env bash
# Release notes for one commit: the commit message, with any "MANUAL:" lines hoisted to the top.
set -euo pipefail
msg=$(git log -1 --format=%B "$1")
manual=$(printf '%s\n' "$msg" | grep -E '^MANUAL:' || true)
rest=$(printf '%s\n' "$msg" | grep -vE '^MANUAL:' || true)
[ -n "$manual" ] && printf '%s\n\n' "$manual"
printf '%s\n' "$rest"
