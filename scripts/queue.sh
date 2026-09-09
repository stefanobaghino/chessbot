#!/usr/bin/env bash
# Run one resource-heavy job on the sparring cores (2-3) behind a lock, optionally
# inside the 09:00-21:00 window. Jobs queued from several shells serialise instead
# of overlapping; a timed match started while another runs would be worthless.
#
# Usage: scripts/queue.sh [--window] [--est MINUTES] [--name NAME] -- <command> [args...]
#   --window       wait for the next 09:00 start such that the job (of --est minutes,
#                  default 60) finishes before 21:00; re-checked after the lock is taken
#   --est MINUTES  estimated duration, used only by --window
#   --name NAME    label in matches/queue.log (default: first word of the command)
#   --next-start   print the epoch second the job would start at and exit (for tests)
# Env: QUEUE_NOW (epoch seconds) overrides the clock, for tests. Child processes see
# CORES_LOCKED=1 so spar.sh and match.sh do not try to take the lock again.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOCK="${QUEUE_LOCK:-$ROOT/matches/.cores23.lock}"
LOG="${QUEUE_LOG:-$ROOT/matches/queue.log}"
WINDOW=0; EST=60; NAME=""; PRINT_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --window) WINDOW=1 ;;
    --est) EST="$2"; shift ;;
    --name) NAME="$2"; shift ;;
    --next-start) PRINT_ONLY=1 ;;
    --) shift; break ;;
    *) echo "queue: unknown option $1" >&2; exit 2 ;;
  esac
  shift
done
[ $# -gt 0 ] || { echo "queue: no command given" >&2; exit 2; }
NAME="${NAME:-$(basename "$1")}"

now() { echo "${QUEUE_NOW:-$(date +%s)}"; }
# Earliest start at or after $1 (epoch) from which EST minutes end before 21:00.
next_start() {
  local t="$1" day start end
  day=$(date -d "@$t" +%F)
  start=$(date -d "$day 09:00" +%s)
  end=$(date -d "$day 21:00" +%s)
  if [ "$t" -lt "$start" ]; then t=$start; fi
  if [ $((t + EST * 60)) -gt "$end" ]; then
    t=$(date -d "$(date -d "@$t" +%F) + 1 day 09:00" +%s)
  fi
  echo "$t"
}
wait_window() {
  local t target
  t=$(now); target=$(next_start "$t")
  if [ "$target" -gt "$t" ]; then
    echo "queue: $NAME waits until $(date -d "@$target" '+%F %R') ($EST min estimated)"
    [ -n "${QUEUE_NOW:-}" ] || sleep $((target - t))
  fi
}

if [ "$PRINT_ONLY" = 1 ]; then
  if [ "$WINDOW" = 1 ]; then next_start "$(now)"; else now; fi
  exit 0
fi
[ "$WINDOW" = 1 ] && wait_window
mkdir -p "$(dirname "$LOCK")"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "queue: $NAME waits for the cores lock"
  flock 9
  [ "$WINDOW" = 1 ] && wait_window   # the wait may have pushed us past the window
fi
export CORES_LOCKED=1
echo "$(date '+%F %T') start $NAME: $*" >> "$LOG"
set +e
"$@"
RC=$?
set -e
echo "$(date '+%F %T') end $NAME rc=$RC" >> "$LOG"
exit $RC
