#!/usr/bin/env bash
# Paired production benchmark for transferable operator priors (issue #221).
#
# Arms (exclusive box time — never run two at once):
#   control  — --strategy-priors off (cold start; reads and writes nothing)
#   seeded   — --strategy-priors seed, run twice against one priors file:
#              the warm-up writes what it measured, and the second run is the
#              one the comparison is about, because it is the only run that
#              opens with a ledger it did not have to earn.
#
# Pairing: the same --seed across both arms, so the focus stream and the opening
# quotas start identical and only the seeded ledger moves. Repeats (SEEDS) are
# required: on a creature where accepts are rare, one pair is an anecdote.
#
# Gate metric: scoreImprovementPerWallHour from `neat_ai_lamarck report`
# (full-corpus anchored; unavailable under --skip-phase0 — do not pass it).
#
# Usage:
#   scripts/run-strategy-priors-ab.sh [arm ...]
#   SEEDS="1 2 3" ARM_SECONDS=2700 scripts/run-strategy-priors-ab.sh
set -euo pipefail

LAMARCK="${LAMARCK:-./target/release/neat_ai_lamarck}"
CREATURE="${CREATURE:-../GRQ-cluster/network.json}"
TRAIN_DATA="${TRAIN_DATA:-.lamarck-strategy-priors/train-data}"
SCORER="${SCORER:-../NEAT-AI-scorer/target/release/rust_scorer}"
OUT_DIR="${OUT_DIR:-.lamarck-strategy-priors}"

ARM_SECONDS="${ARM_SECONDS:-2700}"
SEEDS="${SEEDS:-1 2 3}"
HALF_LIFE_HOURS="${HALF_LIFE_HOURS:-24}"

die() {
  echo "run-strategy-priors-ab: $*" >&2
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
    --strategy-allocation adaptive
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
  arms=(control seeded)
fi

for seed in $SEEDS; do
  for arm in "${arms[@]}"; do
    case "$arm" in
      control)
        run_arm control "$seed" --strategy-priors off
        ;;
      seeded)
        # One priors file per seed: the warm-up fills it, the measured run reads
        # it. Removing it first keeps a re-run from inheriting an older pass.
        priors="$OUT_DIR/priors-seed${seed}.json"
        rm -f "$priors"
        run_arm seeded-warmup "$seed" \
          --strategy-priors seed \
          --strategy-priors-path "$priors" \
          --strategy-priors-half-life-hours "$HALF_LIFE_HOURS"
        [[ -f "$priors" ]] || die "warm-up seed=$seed wrote no priors file: $priors"
        run_arm seeded "$seed" \
          --strategy-priors seed \
          --strategy-priors-path "$priors" \
          --strategy-priors-half-life-hours "$HALF_LIFE_HOURS"
        ;;
      *) die "unknown arm: $arm (control|seeded)" ;;
    esac
  done
done

echo "All requested arms finished under $OUT_DIR"
echo "Summarise with: scripts/summarise-strategy-priors.sh $OUT_DIR"
