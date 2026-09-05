#!/usr/bin/env bash
# Install the daily training-pipeline timers as persistent systemd user units.
# Transient `systemd-run --on-calendar` timers vanish on reboot (they did on 2026-09-05);
# these survive it, and with Persistent=true a run missed while the machine was down
# starts at boot. Both jobs check the 09:00-21:00 window themselves, and are capped to
# 50% of the machine (CPUQuota=200% on 4 cores, pinned to cores 2-3 by the scripts).
# The relabel job chains into the training job, so training starts the moment the last
# chunk is done instead of waiting for the next 09:05.
# Both also run 2 min after boot (OnBootSec), so a reboot inside the window only costs
# the chunk or epoch that was in progress.
# Usage: scripts/install_timers.sh [--uninstall]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UNITS="$HOME/.config/systemd/user"
mkdir -p "$UNITS"

if [ "${1:-}" = "--uninstall" ]; then
  systemctl --user disable --now chessbot-relabel-d10b-daily.timer chessbot-train-net6-daily.timer 2>/dev/null || true
  rm -f "$UNITS"/chessbot-relabel-d10b-daily.* "$UNITS"/chessbot-train-net6-daily.*
  systemctl --user daemon-reload
  echo "timers removed"
  exit 0
fi

unit() {  # name description calendar command
  cat > "$UNITS/$1.service" <<UNIT
[Unit]
Description=$2

[Service]
Type=oneshot
WorkingDirectory=$ROOT
Environment=PATH=/usr/local/bin:/usr/bin:/bin:/usr/games
Nice=10
CPUQuota=200%
SuccessExitStatus=3
ExecStart=/usr/bin/bash -c '$4'
UNIT
  cat > "$UNITS/$1.timer" <<UNIT
[Unit]
Description=$2 (daily)

[Timer]
OnCalendar=$3
OnBootSec=2min
Persistent=true

[Install]
WantedBy=timers.target
UNIT
}

unit chessbot-relabel-d10b-daily "chessbot: depth-10 relabel of self-play batches 1-3" "*-*-* 09:00:00" \
  'scripts/relabel_chunks.sh data/selfplay3_rest.fens data/selfplay3_d10b 10 50000 2 >> data/selfplay3_d10b.log 2>&1 && scripts/train_net6.sh >> data/nets/net6.log 2>&1'
unit chessbot-train-net6-daily "chessbot: net6 training" "*-*-* 09:05:00" \
  'scripts/train_net6.sh >> data/nets/net6.log 2>&1'

systemctl --user daemon-reload
systemctl --user enable --now chessbot-relabel-d10b-daily.timer chessbot-train-net6-daily.timer
systemctl --user list-timers --no-pager 'chessbot-relabel*' 'chessbot-train*'
