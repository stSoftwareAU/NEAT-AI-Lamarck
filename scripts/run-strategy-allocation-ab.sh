#!/usr/bin/env bash
# Paired production benchmark for adaptive strategy allocation (issue #218).
#
# Arms (exclusive box time — never run two at once):
#   control   — --strategy-allocation fixed (today's round-robin split)
#   adaptive  — --strategy-allocation adaptive (slots from measured return)
#
# Pairing: the same --seed across both arms, so the focus stream and the fixed
# opening quotas start identical and only the allocation moves. Streams diverge
# once the allocation reorders the batch — that is the treatment, not a fault.
# Repeats (SEEDS) are required: on a creature where accepts are rare, one pair
# is an anecdote.
#
# Gate metric: scoreImprovementPerWallHour from `neat_ai_lamarck report`
# (full-corpus anchored; unavailable under --skip-phase0 — do not pass it).
#
# Usage:
#   scripts/run-strategy-allocation-ab.sh [arm ...]
#   SEEDS="1 2 3" ARM_SECONDS=2700 scripts/run-strategy-allocation-ab.sh
set -euo pipefail

LAMARCK="${LAMARCK:-./target/release/neat_ai_lamarck}"
CREATURE="${CREATURE:-../GRQ-cluster/network.json}"
TRAIN_DATA="${TRAIN_DATA:-.lamarck-strategy-allocation/train-data}"
SCORER="${SCORER:-../NEAT-AI-scorer/target/release/rust_scorer}"
OUT_DIR="${OUT_DIR:-.lamarck-strategy-allocation}"

ARM_SECONDS="${ARM_SECONDS:-2700}"
SEEDS="${SEEDS:-1 2 3}"
EXPLORATION_FLOOR="${EXPLORATION_FLOOR:-0.2}"
EVIDENCE_DECAY="${EVIDENCE_DECAY:-0.9}"

die() {
  echo "run-strategy-allocation-ab: $*" >&2
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

# Populate COMMON_ARGS (bash 3 compatible — macOS /bin/bash has no mapfile).
COMMON_ARGS=()
set_common_args() {
  local seed="$1"
  COMMON_ARGS=(
    --scorer "$SCORER"
    --timeout-seconds "$ARM_SECONDS"
    --candidates 100
    --seed "$seed"
    --focus-policy weighted
    --screen-sample-rate 0.05
    --screen-promote-threshold 1e-6
    --quick --quick-sample-records 25000
  )
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

  set_common_args "$seed"
  "$LAMARCK" "$CREATURE" "$TRAIN_DATA" \
    --output-dir "$dir" \
    "${COMMON_ARGS[@]}" \
    "$@" 2>&1 | tee "$dir/run.log"

  date -u +"end %Y-%m-%dT%H:%M:%SZ" >>"$dir/timing.txt"
  echo "loadAfter: $(load_average)" >>"$dir/timing.txt"

  [[ -f "$dir/experiments.jsonl" ]] || die "arm $name seed=$seed produced no journal"
  "$LAMARCK" report "$dir/experiments.jsonl" >"$dir/report.json"
  echo "=== arm $name seed=$seed done — report: $dir/report.json"
}

arms=("$@")
if [[ ${#arms[@]} -eq 0 ]]; then
  arms=(control adaptive)
fi

for seed in $SEEDS; do
  for arm in "${arms[@]}"; do
    case "$arm" in
      control)
        run_arm control "$seed" --strategy-allocation fixed
        ;;
      adaptive)
        run_arm adaptive "$seed" \
          --strategy-allocation adaptive \
          --strategy-exploration-floor "$EXPLORATION_FLOOR" \
          --strategy-evidence-decay "$EVIDENCE_DECAY"
        ;;
      *) die "unknown arm: $arm (control|adaptive)" ;;
    esac
  done
done

echo "All requested arms finished under $OUT_DIR"
echo "Summarise with: scripts/summarise-strategy-allocation.sh $OUT_DIR"
