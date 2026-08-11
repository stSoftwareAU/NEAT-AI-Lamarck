#!/usr/bin/env bash
# Follow-up economics campaign for issue #75 — the four experiments the #8
# baseline (docs/baseline-economics.md) recommended and never ran.
#
# Each arm writes its own journal + `report` JSON under $OUT_DIR/<arm>/ so the
# arms stay independently analysable. Arms run one at a time: two Lamarck runs
# sharing the box would contend for the scorer and corrupt every per-minute
# figure the campaign exists to measure.
#
# Usage:
#   scripts/run-followup-economics.sh [arm ...]      # default: all arms
#   ARM_SECONDS=900 scripts/run-followup-economics.sh batch-size
#
# Arms: output-focus | backprop-step | batch-size | multi-seed
#       output-neuron | backprop-cap
#
# `output-neuron` and `backprop-cap` are #96 arms and are **not** in the default
# set: like `multi-seed` they need the production creature and exclusive use of
# the scorer, so name them explicitly and run them one at a time.
set -euo pipefail

LAMARCK="${LAMARCK:-./target/release/neat_ai_lamarck}"
CREATURE="${CREATURE:-../GRQ-cluster/network.json}"
TRAIN_DATA="${TRAIN_DATA:-.lamarck-followup/train-data}"
SCORER="${SCORER:-../NEAT-AI-scorer/target/release/rust_scorer}"
OUT_DIR="${OUT_DIR:-.lamarck-followup}"

# Short-arm budget (seconds). The batch-size A/B is fixed at 15 minutes by
# design (#75.3); the output-focus arm gets longer because #75.1 asks for >=15
# experiments and this box runs ~60s/experiment beside GRQ.
ARM_SECONDS="${ARM_SECONDS:-900}"
OUTPUT_FOCUS_SECONDS="${OUTPUT_FOCUS_SECONDS:-1200}"
# Production-budget budget for the multi-seed repeat.
SEED_SECONDS="${SEED_SECONDS:-2700}"
# Seeds for the multi-seed repeat (the #8 baseline used seed 1).
SEEDS="${SEEDS:-2 3 4 5}"
# Candidate counts for the batch-size A/B. #75 asked for 40/100/150, but
# candidate generation has a fixed per-phase ceiling (~29 on the GRQ creature),
# so every budget at or above it fills the same batch — see the ceiling test in
# lamarck/src/candidates.rs. 12 is added as a point that actually binds.
BATCH_SIZES="${BATCH_SIZES:-12 40 100 150}"
# Learning rates for the backprop step A/B (default arm first).
BACKPROP_RATES="${BACKPROP_RATES:-0.01 0.001}"
# Bias-step caps for the #96 backprop cap A/B (default cap first). The 10.0
# default is ~7 orders of magnitude coarser than the 1e-6 accept bar, so the
# ladder walks down towards it rather than around it.
BACKPROP_CAPS="${BACKPROP_CAPS:-10 0.01 0.000001}"
# Per-cap budget: three caps at 400s is the ~20 minutes #96 allows this arm.
CAP_SECONDS="${CAP_SECONDS:-400}"
# Neuron pinned by the #96 output slice. NEAT-AI names output neurons
# `output-<index>`; a UUID that does not exist aborts the run rather than
# quietly falling back to policy selection.
OUTPUT_NEURON="${OUTPUT_NEURON:-output-0}"
OUTPUT_NEURON_SECONDS="${OUTPUT_NEURON_SECONDS:-1200}"

die() {
  echo "run-followup-economics: $*" >&2
  exit 1
}

[[ -x "$LAMARCK" ]] || die "lamarck binary not executable: $LAMARCK (cargo build --release)"
[[ -x "$SCORER" ]] || die "scorer binary not executable: $SCORER"
[[ -f "$CREATURE" ]] || die "creature not found: $CREATURE"
[[ -d "$TRAIN_DATA" ]] || die "training-data directory not found: $TRAIN_DATA"

mkdir -p "$OUT_DIR"

# Load average at the head of each run: this box also runs GRQ, and every
# per-minute figure below has to be read against the load it was measured under.
load_average() {
  uptime | sed -e 's/.*load averages*: //'
}

# run_arm <name> <extra lamarck args...>
run_arm() {
  local name="$1"
  shift
  local dir="$OUT_DIR/$name"
  rm -rf "$dir"
  mkdir -p "$dir"

  echo "=== arm $name — load before: $(load_average)"
  date -u +"start %Y-%m-%dT%H:%M:%SZ" | tee "$dir/timing.txt"
  echo "loadBefore: $(load_average)" >>"$dir/timing.txt"

  # No `|| true`: a failed arm must stop the campaign rather than leave a
  # truncated journal to be read as a result.
  "$LAMARCK" "$CREATURE" "$TRAIN_DATA" \
    --scorer "$SCORER" \
    --output-dir "$dir" \
    --quick --quick-sample-records 25000 \
    --screen-sample-rate 0.05 \
    "$@" 2>&1 | tee "$dir/run.log"

  date -u +"end %Y-%m-%dT%H:%M:%SZ" >>"$dir/timing.txt"
  echo "loadAfter: $(load_average)" >>"$dir/timing.txt"

  [[ -f "$dir/experiments.jsonl" ]] || die "arm $name produced no journal"
  "$LAMARCK" report "$dir/experiments.jsonl" >"$dir/report.json"
  echo "=== arm $name done — report: $dir/report.json"
}

arm_output_focus() {
  # #75.1 — output-residual economics. `high-error` sticks on the largest
  # residual neuron, which is what the arm wants to measure.
  run_arm "output-focus" \
    --timeout-seconds "$OUTPUT_FOCUS_SECONDS" --candidates 100 --seed 11 \
    --focus-policy high-error
}

arm_backprop_step() {
  # #75.2 — backprop step A/B on a policy-selected hidden focus. Same seed in
  # both arms so the focus stream and candidate slots match; only the rate moves.
  for rate in $BACKPROP_RATES; do
    run_arm "backprop-step-$rate" \
      --timeout-seconds "$ARM_SECONDS" --candidates 100 --seed 21 \
      --focus-policy weighted \
      --backprop-learning-rate "$rate"
  done
}

arm_batch_size() {
  # #75.3 — fixed wall clock, candidates varied. Same seed across the three so
  # the difference is batch size, not the focus stream.
  for n in $BATCH_SIZES; do
    run_arm "batch-size-$n" \
      --timeout-seconds "$ARM_SECONDS" --candidates "$n" --seed 31 \
      --focus-policy weighted
  done
}

arm_multi_seed() {
  # #75.4 — repeat the production config on fresh seeds before deprioritising
  # any non-random strategy on the strength of the single #8 sample.
  for seed in $SEEDS; do
    run_arm "multi-seed-$seed" \
      --timeout-seconds "$SEED_SECONDS" --candidates 100 --seed "$seed" \
      --focus-policy weighted
  done
}

arm_output_neuron() {
  # #96.2 — the unrun half of #75.1. `--focus-policy high-error` ranks by
  # error-influence mass and never leaves the hidden layer on this creature, so
  # the output slice has to pin the neuron. This is the only arm that can
  # measure `mean_error_bias` / `stats_skew_bias`, which propose from the
  # output target and therefore need an output focus.
  run_arm "output-neuron" \
    --timeout-seconds "$OUTPUT_NEURON_SECONDS" --candidates 100 --seed 41 \
    --focus-neuron "$OUTPUT_NEURON"
}

arm_backprop_cap() {
  # #96.3 — the knob #75.2 identified. The learning-rate A/B returned identical
  # bias candidates because a saturating blame mass pins the step to
  # `maximum_bias_adjustment_scale` at every rate, so this arm varies the cap
  # instead. Same seed across caps so only the cap moves.
  for cap in $BACKPROP_CAPS; do
    run_arm "backprop-cap-$cap" \
      --timeout-seconds "$CAP_SECONDS" --candidates 100 --seed 51 \
      --focus-policy weighted \
      --backprop-max-bias-adjustment-scale "$cap"
  done
}

arms=("$@")
if [[ ${#arms[@]} -eq 0 ]]; then
  arms=(output-focus backprop-step batch-size multi-seed)
fi

for arm in "${arms[@]}"; do
  case "$arm" in
  output-focus) arm_output_focus ;;
  backprop-step) arm_backprop_step ;;
  batch-size) arm_batch_size ;;
  multi-seed) arm_multi_seed ;;
  output-neuron) arm_output_neuron ;;
  backprop-cap) arm_backprop_cap ;;
  *) die "unknown arm '$arm' (output-focus | backprop-step | batch-size | multi-seed | output-neuron | backprop-cap)" ;;
  esac
done

echo "campaign complete — reports under $OUT_DIR/*/report.json"
