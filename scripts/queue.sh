#!/usr/bin/env bash
# Run one resource-heavy job on the sparring cores (2-3) behind a lock, optionally
# inside the 09:00-21:00 window. Jobs queued from several shells serialise instead
# of overlapping and start in the order they were queued; a timed match started
# while another runs would be worthless.
#
# Usage: scripts/queue.sh [--window] [--est MINUTES] [--name NAME] -- <command> [args...]
#   --window       wait for the next 09:00 start such that the job (of --est minutes,
#                  default 60) finishes before 21:00; re-checked after the lock is taken,
#                  and a job pushed past 21:00 gives the lock back until the morning
#   --est MINUTES  estimated duration, used only by --window
#   --name NAME    label in matches/queue.log (default: first word of the command)
#   --next-start   print the epoch second the job would start at and exit (for tests)
# Order (see #54): each job takes a ticket, a file named by submission time and pid
# under <lock>.d/, and waits until no older ticket of a live process is ready before
# taking the lock. A job still waiting for its window is not ready and does not hold
# the line; tickets of dead processes are removed by the next waiter. spar.sh and
# match.sh run directly take no ticket and compete at the lock only.
# Env: QUEUE_NOW (epoch seconds) overrides the clock and QUEUE_POLL (seconds) the turn
# poll, for tests. Child processes see CORES_LOCKED=1 so spar.sh and match.sh do not
# try to take the lock again.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOCK="${QUEUE_LOCK:-$ROOT/matches/.cores23.lock}"
LOG="${QUEUE_LOG:-$ROOT/matches/queue.log}"
TICKETS="$LOCK.d"
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

mkdir -p "$(dirname "$LOCK")" "$TICKETS"
TICKET="$TICKETS/$(printf '%019d' "$(date +%s%N)")-$$"
: > "$TICKET"
trap 'rm -f "$TICKET" "$TICKET.ready"' EXIT
# True while an older ticket belongs to a live process that is ready to run.
turn_blocked() {
  local t
  for t in "$TICKETS"/*-*; do
    case "$t" in *.ready) continue ;; esac
    [ -e "$t" ] && [[ "$t" < "$TICKET" ]] || continue
    if kill -0 "${t##*-}" 2>/dev/null; then
      [ -e "$t.ready" ] && return 0
    else
      rm -f "$t" "$t.ready"
    fi
  done
  return 1
}
wait_turn() {
  local said=0
  while turn_blocked; do
    [ "$said" = 1 ] || { echo "queue: $NAME waits for its turn"; said=1; }
    sleep "${QUEUE_POLL:-5}"
  done
}
exec 9>"$LOCK"
while :; do
  [ "$WINDOW" = 1 ] && wait_window
  : > "$TICKET.ready"
  wait_turn
  if ! flock -n 9; then
    echo "queue: $NAME waits for the cores lock"
    flock 9
  fi
  # The waits may have pushed a windowed job past 21:00: give the lock back until morning.
  if [ "$WINDOW" = 1 ] && [ -z "${QUEUE_NOW:-}" ] && [ "$(next_start "$(now)")" -gt "$(now)" ]; then
    flock -u 9; rm -f "$TICKET.ready"; continue
  fi
  break
done
export CORES_LOCKED=1
echo "$(date '+%F %T') start $NAME: $*" >> "$LOG"
set +e
"$@"
RC=$?
set -e
echo "$(date '+%F %T') end $NAME rc=$RC" >> "$LOG"
exit $RC
