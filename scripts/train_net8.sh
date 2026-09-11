#!/usr/bin/env bash
# The #51 candidates. Once batches 1 and 2 of the self-play set are relabelled at depth 10 it
# builds the deduplicated training set (the new chunks first, then data/all_unique.npz, minus
# the held-out set of #44), trains two nets on it and gates them at fixed nodes and at 10+0.1
# against the released engine (BASE_REF with its own net):
#   net8a  the net6 architecture from main's trainer, so a win ships as a net-only release;
#   net8k  four king buckets and one output bucket (net7e's shape), trained and built from
#          the nnue-buckets branch with OUT_BUCKETS forced to 1.
# Every step is skipped when its output exists, so the job is rerun until it logs "done".
# Exit 0: done, or the batches are not complete yet. Exit 3: stopped for the window; rerun
# after 09:00. Meant for the sparring cores behind the lock:
#   scripts/queue.sh --window --est 420 --name cand51 -- scripts/train_net8.sh
# --ready only answers whether the batches are complete (exit 0) or not (exit 1).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
CPUS="${SPAR_CPUS:-2-3}"
BASE_REF="${BASE_REF:-v0.1.46}"
BUCKETS_REF="${BUCKETS_REF:-nnue-buckets}"
WINDOW="${WINDOW:-9-21}"
HELD=data/held_d10.npz
DATA=data/selfplay4_unique.npz
LOG=data/nets/net8.log
log() { echo "$(date +%FT%T) train_net8: $*" | tee -a "$LOG"; }

chunks_of() { ls "data/selfplay4_d10_$1_"*.npz 2>/dev/null | grep -c -E '_[0-9]+\.npz$' || true; }
batches_ready() {
  local i fens
  for i in 1 2; do
    fens=data/selfplay4_$i.fens
    [ -s "$fens" ] || return 1
    [ "$(chunks_of "$i")" -ge $(( ($(wc -l < "$fens") + 49999) / 50000 )) ] || return 1
  done
}
if [ "${1:-}" = --ready ]; then
  if batches_ready; then exit 0; else exit 1; fi
fi
batches_ready || { log "batches 1-2 are not fully relabelled yet; not starting"; exit 0; }

# True if a step of that many minutes, started now, ends inside today's window.
fits() { [ $(( $(date +%s) + $1 * 60 )) -le "$(date -d "${WINDOW#*-}:00" +%s)" ]; }

if [ ! -s "$DATA" ]; then
  chunks=$(ls data/selfplay4_d10_[12]_*.npz | grep -E '_[0-9]+\.npz$' | sort -V)
  log "building $DATA from $(echo "$chunks" | wc -l) chunks and data/all_unique.npz"
  # shellcheck disable=SC2086
  nice taskset -c "$CPUS" .venv/bin/python train/dedup.py "$DATA.tmp.npz" $chunks data/all_unique.npz --exclude "$HELD" | tee -a "$LOG"
  mv "$DATA.tmp.npz" "$DATA"
fi

train() {  # name trainer [trainer options]
  local name=$1 py=$2 rc; shift 2
  local out=data/nets/$name.bin
  if [ -f "$out.done" ]; then log "$name already trained"; return 0; fi
  log "training $name with $py"
  set +e
  nice taskset -c "$CPUS" .venv/bin/python "$py" "$DATA" "$out" --epochs 12 --hidden 384 --lr 1e-3 \
    --threads 2 --window "$WINDOW" --holdout "$HELD" "$@" >> "data/nets/$name.log" 2>&1
  rc=$?
  set -e
  if [ "$rc" -ne 0 ]; then log "$name training stopped with status $rc"; exit "$rc"; fi
  touch "$out.done"
  log "$name trained, best holdout loss $(sed -nE 's/.*holdout ([0-9.]+).*/\1/p' "data/nets/$name.log" | sort -n | head -n 1)"
}
train net8a train/train.py
BUCKETS_PY=data/nets/train_buckets.py
git show "$BUCKETS_REF:train/train.py" > "$BUCKETS_PY"
train net8k "$BUCKETS_PY" --king-buckets 4 --out-buckets 1

build() {  # name ref [net [sed expression for engine/src/nnue.rs]]
  local name=$1 ref=$2 net=${3:-} edit=${4:-} bin=matches/bin/$1 wt=matches/wt_$1 check
  if [ -x "$bin" ]; then log "$bin already built"; return 0; fi
  git worktree remove --force "$wt" 2>/dev/null || true
  git worktree prune
  git worktree add --detach -q "$wt" "$ref"
  if [ -n "$edit" ]; then
    sed -i -E "$edit" "$wt/engine/src/nnue.rs"
    git -C "$wt" diff --quiet -- engine/src/nnue.rs && { log "the edit of nnue.rs did not apply"; exit 1; }
  fi
  if [ -n "$net" ]; then cp "$net" "$wt/engine/nets/default.bin"; fi
  log "building $bin from $ref${net:+ with $net}"
  (cd "$wt/engine" && CARGO_TARGET_DIR="$ROOT/ci/target" nice taskset -c "$CPUS" cargo build --release -q)
  cp "$ROOT/ci/target/release/chessbot-engine" "$bin.tmp"
  git worktree remove --force "$wt"
  check=$("$bin.tmp" selfcheck | tail -n 1)
  log "$name $check"
  echo "$check" | grep -q " 0 mismatches" || { rm -f "$bin.tmp"; exit 1; }
  mv "$bin.tmp" "$bin"
}
build base46 "$BASE_REF"
build net8a "$BASE_REF" data/nets/net8a.bin
build net8k "$BUCKETS_REF" data/nets/net8k.bin 's/^pub const OUT_BUCKETS: usize = 4;/pub const OUT_BUCKETS: usize = 1;/'

elo() { grep -E '^Elo' "matches/$1.log" | tail -n 1; }
gate() {  # name new games nodes minutes [tc]
  local name=$1 new=$2 games=$3 nodes=$4 minutes=$5 tc=${6:-}
  if [ -f "matches/$name.done" ]; then log "$name already played: $(elo "$name")"; return 0; fi
  fits "$minutes" || { log "$name ($minutes min) would not finish inside $WINDOW; stopping"; exit 3; }
  log "playing $name"
  TC="$tc" scripts/spar.sh "$new" matches/bin/base46 "$games" "$nodes" "$name" 2 > /dev/null 2>&1
  touch "matches/$name.done"
  log "$name: $(elo "$name")"
}
for c in net8a net8k; do gate "${c}_vs_46_fixed" "matches/bin/$c" 600 40000 70; done
for c in net8a net8k; do
  if awk -v e="$(elo "${c}_vs_46_fixed" | sed -E 's/^Elo: (-?[0-9.]+).*/\1/')" 'BEGIN { exit !(e >= 0) }'; then
    gate "${c}_vs_46_10s" "matches/bin/$c" 200 0 75 10+0.1
  else
    log "$c lost the fixed-node gate; no timed gate"
  fi
done
log "done"
