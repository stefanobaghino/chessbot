#!/usr/bin/env bash
# Creates the cgroups that keep background work off the cores used by the live Lichess bot.
# Run as root after boot (cgroups do not persist): sudo scripts/cgroups.sh [user]
#
#   quiet  cores 2-3, 1 core cap   background jobs (relabelling, data generation)
#   train  cores 2-3, 2 core cap   training runs
#
# Moving a process in still needs root (sudo sh -c "echo $$ > /sys/fs/cgroup/quiet/cgroup.procs")
# but the owner can freeze/thaw the groups. Matches use taskset instead (scripts/spar.sh).
# The live bot is pinned to cores 0-1 by CPUAffinity in its systemd unit.
set -euo pipefail
OWNER="${1:-${SUDO_USER:-$USER}}"
ROOT=/sys/fs/cgroup
grep -qw cpuset "$ROOT/cgroup.subtree_control" || echo +cpuset > "$ROOT/cgroup.subtree_control"
grep -qw cpu "$ROOT/cgroup.subtree_control" || echo +cpu > "$ROOT/cgroup.subtree_control"
mk() {
  local name="$1" cpus="$2" max="$3"
  mkdir -p "$ROOT/$name"
  echo "$cpus" > "$ROOT/$name/cpuset.cpus"
  echo "$max" > "$ROOT/$name/cpu.max"
  chown "$OWNER" "$ROOT/$name/cgroup.procs" "$ROOT/$name/cgroup.freeze"
  printf '%-6s cpus=%-4s cpu.max=%s\n' "$name" "$(cat "$ROOT/$name/cpuset.cpus.effective")" "$(cat "$ROOT/$name/cpu.max")"
}
mk quiet 2-3 "100000 100000"
mk train 2-3 "200000 100000"
