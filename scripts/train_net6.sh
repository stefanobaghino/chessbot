#!/usr/bin/env bash
# Train net6 (issue #12) once every depth-10 chunk of data/selfplay3_rest.fens exists,
# inside the 09:00-21:00 window, resuming from data/nets/net6.bin.ckpt across days.
# Meant to run from a daily 09:00 timer; it is a no-op while the relabel job is still
# running or its chunks are incomplete. Exit 0: trained or nothing to do yet. Exit 3:
# stopped for the window; the next timer run resumes.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT="${OUT:-data/nets/net6.bin}"
CPUS="${SPAR_CPUS:-2-3}"
log() { echo "$(date +%FT%T) train_net6: $*"; }
if [ -f "$OUT.done" ]; then log "$OUT already trained"; exit 0; fi
if systemctl --user is-active --quiet chessbot-relabel-d10b-resume.service chessbot-relabel-d10b-daily.service 2>/dev/null; then
  log "relabel job still running; not starting"; exit 0
fi
chunks=$(( ($(wc -l < data/selfplay3_rest.fens) + 49999) / 50000 ))
have=$(ls data/selfplay3_d10b_*.npz 2>/dev/null | wc -l)
if [ "$have" -lt "$chunks" ]; then log "$have/$chunks relabel chunks present; not starting"; exit 0; fi
S1=data/selfplay3_1.npz; S2=data/selfplay3_2.npz; S3=data/selfplay3_3.npz
P=data/selfplay2_2_partial.npz; D=data/selfplay3_d10.npz
B=$(ls data/selfplay3_d10b_*.npz | sort -V | paste -sd,)
# net5 mix (self-play x2, depth-10 slice x3) plus the new depth-10 chunks x3.
DATA="data/sf6_a.npz,$S1,$S1,$S2,$S2,$S3,$S3,$P,$P,$D,$D,$D,$B,$B,$B"
log "training $OUT on cpus $CPUS"
set +e
nice taskset -c "$CPUS" .venv/bin/python train/train.py "$DATA" "$OUT" \
  --epochs 20 --hidden 384 --lr 1e-3 --threads 2 --window "${WINDOW:-9-21}"
rc=$?
set -e
if [ "$rc" -eq 0 ]; then touch "$OUT.done"; log "done"; fi
exit "$rc"
