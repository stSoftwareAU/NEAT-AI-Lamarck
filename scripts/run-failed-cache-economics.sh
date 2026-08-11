#!/usr/bin/env bash
# Paired production benchmark for the failed-candidate cache (issue #94 / #69).
#
# Arms (exclusive box time — never run two at once):
#   control     — cache off (today's behaviour)
#   treatment   — cache on, warm snapshot allowed
#   cold-start  — cache on, no snapshot; rebuild from a long journal
#
# Pairing: same --seed across control/treatment so RNG streams start identical
# (#71). Streams diverge once backfill draws extra candidates — that is
# expected. Repeats (REPEATS, SEEDS) are required: one pair on a rare-accept
# creature is anecdote.
#
# Gate metric: scoreImprovementPerWallHour from `neat_ai_lamarck report`
# (full-corpus anchored; unavailable under --skip-phase0 — do not pass it).
#
# Usage:
#   scripts/run-failed-cache-economics.sh [arm ...]
#   SEEDS="1 2 3" ARM_SECONDS=2700 scripts/run-failed-cache-economics.sh
#
# Default arms: control treatment cold-start (every seed in SEEDS).
set -euo pipefail

LAMARCK="${LAMARCK:-./target/release/neat_ai_lamarck}"
CREATURE="${CREATURE:-../GRQ-cluster/network.json}"
TRAIN_DATA="${TRAIN_DATA:-.lamarck-failed-cache/train-data}"
SCORER="${SCORER:-../NEAT-AI-scorer/target/release/rust_scorer}"
OUT_DIR="${OUT_DIR:-.lamarck-failed-cache}"

ARM_SECONDS="${ARM_SECONDS:-2700}"
SEEDS="${SEEDS:-1 2 3}"
# Fixed on/off state for #83-related knobs across both arms (do not vary).
# Backprop gating from #83 is in-tree; leave defaults unless overridden.
BACKPROP_RATE="${BACKPROP_RATE:-}"
BACKPROP_CAP="${BACKPROP_CAP:-}"

die() {
  echo "run-failed-cache-economics: $*" >&2
  exit 1
}

[[ -x "$LAMARCK" ]] || die "lamarck binary not executable: $LAMARCK (cargo build --release)"
[[ -x "$SCORER" ]] || die "scorer binary not executable: $SCORER"
[[ -f "$CREATURE" ]] || die "creature not found: $CREATURE"
[[ -d "$TRAIN_DATA" ]] || die "training-data directory not found: $TRAIN_DATA"

mkdir -p "$OUT_DIR"

load_average() {
  uptime | sed -e 's/.*load averages*: //'
}

common_args() {
  local seed="$1"
  local args=(
    --scorer "$SCORER"
    --timeout-seconds "$ARM_SECONDS"
    --candidates 100
    --seed "$seed"
    --focus-policy weighted
    --screen-sample-rate 0.05
    --screen-promote-threshold 1e-6
    --quick --quick-sample-records 25000
  )
  if [[ -n "$BACKPROP_RATE" ]]; then
    args+=(--backprop-learning-rate "$BACKPROP_RATE")
  fi
  if [[ -n "$BACKPROP_CAP" ]]; then
    args+=(--backprop-max-bias-adjustment-scale "$BACKPROP_CAP")
  fi
  printf '%s\n' "${args[@]}"
}

run_arm() {
  local name="$1"
  local seed="$2"
  shift 2
  local dir="$OUT_DIR/${name}-seed${seed}"
  rm -rf "$dir"
  mkdir -p "$dir"

  echo "=== arm $name seed=$seed — load before: $(load_average)"
  date -u +"start %Y-%m-%dT%H:%M:%SZ" | tee "$dir/timing.txt"
  echo "loadBefore: $(load_average)" >>"$dir/timing.txt"
  echo "seed: $seed" >>"$dir/timing.txt"

  mapfile -t base < <(common_args "$seed")
  "$LAMARCK" "$CREATURE" "$TRAIN_DATA" \
    --output-dir "$dir" \
    "${base[@]}" \
    "$@" 2>&1 | tee "$dir/run.log"

  date -u +"end %Y-%m-%dT%H:%M:%SZ" >>"$dir/timing.txt"
  echo "loadAfter: $(load_average)" >>"$dir/timing.txt"

  [[ -f "$dir/experiments.jsonl" ]] || die "arm $name seed=$seed produced no journal"
  "$LAMARCK" report "$dir/experiments.jsonl" >"$dir/report.json"

  # Pairing validity checks (issue #94 failure detection).
  local journal_seed
  journal_seed="$(python3 -c "
import json,sys
for line in open('$dir/experiments.jsonl'):
    o=json.loads(line)
    if o.get('record')=='runHeader':
        print(o['seed']); break
")"
  [[ "$journal_seed" == "$seed" ]] || die "seed mismatch: expected $seed got $journal_seed"

  echo "=== arm $name seed=$seed done — report: $dir/report.json"
}

run_control() {
  local seed="$1"
  run_arm control "$seed"
  # Explicitly cache-off (default, but pin it for the journal).
}

run_treatment() {
  local seed="$1"
  run_arm treatment "$seed" --failed-cache
}

run_cold_start() {
  local seed="$1"
  # Prefer a long prior journal from the control arm of the same seed so the
  # rebuild cost is real; fall back to empty (cold empty rebuild).
  local prior="$OUT_DIR/control-seed${seed}/experiments.jsonl"
  local dir="$OUT_DIR/cold-start-seed${seed}"
  rm -rf "$dir"
  mkdir -p "$dir"
  if [[ -f "$prior" ]]; then
    cp "$prior" "$dir/experiments.jsonl"
    rm -f "$dir/failed-candidates.cache.json"
  fi
  echo "=== arm cold-start seed=$seed — load before: $(load_average)"
  date -u +"start %Y-%m-%dT%H:%M:%SZ" | tee "$dir/timing.txt"
  mapfile -t base < <(common_args "$seed")
  "$LAMARCK" "$CREATURE" "$TRAIN_DATA" \
    --output-dir "$dir" \
    "${base[@]}" \
    --failed-cache 2>&1 | tee "$dir/run.log"
  date -u +"end %Y-%m-%dT%H:%M:%SZ" >>"$dir/timing.txt"
  "$LAMARCK" report "$dir/experiments.jsonl" >"$dir/report.json"
  echo "=== arm cold-start seed=$seed done"
}

arms=("$@")
if [[ ${#arms[@]} -eq 0 ]]; then
  arms=(control treatment cold-start)
fi

for seed in $SEEDS; do
  for arm in "${arms[@]}"; do
    case "$arm" in
      control) run_control "$seed" ;;
      treatment) run_treatment "$seed" ;;
      cold-start) run_cold_start "$seed" ;;
      *) die "unknown arm: $arm (control|treatment|cold-start)" ;;
    esac
  done
done

echo "All requested arms finished under $OUT_DIR"
echo "Summarise with: scripts/summarise-failed-cache-economics.sh $OUT_DIR"
