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
#   --list         show the queued jobs in order (running, ready, waiting for the
#                  window, or stale) with their age, pid and command, and exit (see #55)
#   --now [NAME]   release the job(s) waiting for their window (all without NAME): each
#                  starts as soon as its turn and the lock allow, with WINDOW_START=0
#                  exported so a window-aware command starts too, and exit (see #58)
# Order (see #54): each job takes a ticket, a file named by submission time and pid
# under <lock>.d/ holding its name and command, and waits until no older ticket of a
# live process is ready before taking the lock. A job still waiting for its window is
# not ready and does not hold the line; tickets of dead processes are removed by the
# next waiter. spar.sh and match.sh run directly take no ticket and compete at the
# lock only.
# Env: QUEUE_NOW (epoch seconds) overrides the clock and QUEUE_POLL (seconds) the turn
# poll, for tests. Child processes see CORES_LOCKED=1 so spar.sh and match.sh do not
# try to take the lock again.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOCK="${QUEUE_LOCK:-$ROOT/matches/.cores23.lock}"
LOG="${QUEUE_LOG:-$ROOT/matches/queue.log}"
TICKETS="$LOCK.d"
WINDOW=0; EST=60; NAME=""; PRINT_ONLY=0; LIST=0; RELEASE=0; RELEASE_NAME=""; EARLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --window) WINDOW=1 ;;
    --est) EST="$2"; shift ;;
    --name) NAME="$2"; shift ;;
    --next-start) PRINT_ONLY=1 ;;
    --list) LIST=1 ;;
    --now) RELEASE=1; if [ $# -gt 1 ] && [ "${2#-}" = "$2" ]; then RELEASE_NAME="$2"; shift; fi ;;
    --) shift; break ;;
    *) echo "queue: unknown option $1" >&2; exit 2 ;;
  esac
  shift
done
if [ "$LIST" = 1 ]; then
  n=0
  for t in "$TICKETS"/*-*; do
    case "$t" in *.ready|*.running|*.now) continue ;; esac
    [ -e "$t" ] || continue
    n=$((n + 1))
    pid="${t##*-}"; stamp="${t##*/}"; stamp="${stamp%-*}"
    age=$(( $(date +%s) - 10#${stamp:0:10} ))
    job=$(cat "$t" 2>/dev/null || true)
    if ! kill -0 "$pid" 2>/dev/null; then state=stale; rm -f "$t" "$t.ready" "$t.running" "$t.now"
    elif [ -e "$t.running" ]; then state=running
    elif [ -e "$t.ready" ]; then state=ready
    else state=waiting; fi
    printf '%-8s %6ss  pid %-8s %s\n' "$state" "$age" "$pid" "$job"
  done
  [ "$n" -gt 0 ] || echo "queue: empty"
  exit 0
fi
if [ "$RELEASE" = 1 ]; then
  n=0
  for t in "$TICKETS"/*-*; do
    case "$t" in *.ready|*.running|*.now) continue ;; esac
    [ -e "$t" ] || continue
    job=$(cat "$t" 2>/dev/null || true)
    [ -z "$RELEASE_NAME" ] || [ "${job%%:*}" = "$RELEASE_NAME" ] || continue
    kill -0 "${t##*-}" 2>/dev/null || continue
    if [ -e "$t.ready" ] || [ -e "$t.running" ]; then echo "queue: ${job%%:*} is not waiting for the window"; continue; fi
    : > "$t.now"
    n=$((n + 1))
    echo "queue: released ${job%%:*} (pid ${t##*-})"
  done
  [ "$n" -gt 0 ] || { echo "queue: nothing to release"; exit 1; }
  exit 0
fi
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
# Waits for the window start, or for the --now mark, which also exports WINDOW_START=0
# to the command so that a window-aware one (relabel_chunks.sh) starts early as well.
wait_window() {
  local t target
  t=$(now); target=$(next_start "$t")
  [ "$target" -gt "$t" ] || return 0
  echo "queue: $NAME waits until $(date -d "@$target" '+%F %R') ($EST min estimated; --now $NAME starts it early)"
  [ -z "${QUEUE_NOW:-}" ] || return 0
  while [ "$(now)" -lt "$target" ]; do
    if [ -e "$TICKET.now" ]; then
      EARLY=1
      echo "queue: $NAME released early"
      echo "$(date '+%F %T') release $NAME" >> "$LOG"
      return 0
    fi
    sleep "${QUEUE_POLL:-5}"
  done
}

if [ "$PRINT_ONLY" = 1 ]; then
  if [ "$WINDOW" = 1 ]; then next_start "$(now)"; else now; fi
  exit 0
fi

mkdir -p "$(dirname "$LOCK")" "$TICKETS"
TICKET="$TICKETS/$(printf '%019d' "$(date +%s%N)")-$$"
echo "$NAME: $*" > "$TICKET"
trap 'rm -f "$TICKET" "$TICKET.ready" "$TICKET.running" "$TICKET.now"' EXIT
# True while an older ticket belongs to a live process that is ready to run.
turn_blocked() {
  local t
  for t in "$TICKETS"/*-*; do
    case "$t" in *.ready|*.running|*.now) continue ;; esac
    [ -e "$t" ] && [[ "$t" < "$TICKET" ]] || continue
    if kill -0 "${t##*-}" 2>/dev/null; then
      [ -e "$t.ready" ] && return 0
    else
      rm -f "$t" "$t.ready" "$t.running" "$t.now"
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
  if [ "$WINDOW" = 1 ] && [ "$EARLY" = 0 ] && [ -z "${QUEUE_NOW:-}" ] && [ "$(next_start "$(now)")" -gt "$(now)" ]; then
    flock -u 9; rm -f "$TICKET.ready"; continue
  fi
  break
done
: > "$TICKET.running"
export CORES_LOCKED=1
if [ "$EARLY" = 1 ]; then export WINDOW_START=0; fi
echo "$(date '+%F %T') start $NAME: $*" >> "$LOG"
set +e
"$@"
RC=$?
set -e
echo "$(date '+%F %T') end $NAME rc=$RC" >> "$LOG"
exit $RC
