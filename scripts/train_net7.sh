#!/usr/bin/env bash
# Train net7, the first king-bucketed net (see #44), on the net6 data inside the 09:00-21:00
# window, resuming from data/nets/net7.bin.ckpt across days. TRAIN_PY points at the bucketed
# train/train.py so the job can run while another branch is checked out.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT="${OUT:-data/nets/net7.bin}"
CPUS="${SPAR_CPUS:-2-3}"
TRAIN_PY="${TRAIN_PY:-train/train.py}"
log() { echo "$(date +%FT%T) train_net7: $*"; }
if [ -f "$OUT.done" ]; then log "$OUT already trained"; exit 0; fi
grep -q "KING_BUCKETS" "$TRAIN_PY" || { log "$TRAIN_PY is not the bucketed trainer"; exit 1; }
S1=data/selfplay3_1.npz; S2=data/selfplay3_2.npz; S3=data/selfplay3_3.npz
P=data/selfplay2_2_partial.npz; D=data/selfplay3_d10.npz
B=$(ls data/selfplay3_d10b_*.npz | sort -V | paste -sd,)
DATA="data/sf6_a.npz,$S1,$S1,$S2,$S2,$S3,$S3,$P,$P,$D,$D,$D,$B,$B,$B"
log "training $OUT on cpus $CPUS with $TRAIN_PY"
set +e
nice taskset -c "$CPUS" .venv/bin/python "$TRAIN_PY" "$DATA" "$OUT" \
  --epochs 20 --hidden 384 --lr 1e-3 --threads 2 --window "${WINDOW:-9-21}"
rc=$?
set -e
if [ "$rc" -eq 0 ]; then touch "$OUT.done"; log "done"; fi
exit "$rc"
