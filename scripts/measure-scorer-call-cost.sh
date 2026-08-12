#!/usr/bin/env bash
# Measure the scorer's fixed per-call and marginal per-creature cost (#112).
#
# Usage: scripts/measure-scorer-call-cost.sh CREATURE TRAINING_DATA SCORER [OUT_DIR]
#   SIZES=0,1,29     candidate counts per call (creature count is one more)
#   RATES=0.05,1     scorer sample rates to sweep
#   REPEATS=1        sweeps per rate
#
# One `cargo run --example scorer_call_cost_bench` invocation per sample rate,
# with the 1-minute load average recorded either side of it: a fixed cost
# measured beside a live scorer run is meaningless, so the conditions are part
# of the result (docs/followup-economics.md load caveat).
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: measure-scorer-call-cost.sh CREATURE TRAINING_DATA SCORER [OUT_DIR]" >&2
  exit 1
fi

CREATURE="$1"
TRAINING_DATA="$2"
SCORER="$3"
OUT_DIR="${4:-docs/evidence/scorer-call-cost}"
SIZES="${SIZES:-0,1,29}"
RATES="${RATES:-0.05,1}"
REPEATS="${REPEATS:-1}"

[[ -f "$CREATURE" ]] || {
  echo "measure-scorer-call-cost: no such creature: $CREATURE" >&2
  exit 1
}
[[ -d "$TRAINING_DATA" ]] || {
  echo "measure-scorer-call-cost: no such training-data directory: $TRAINING_DATA" >&2
  exit 1
}
[[ -x "$SCORER" ]] || {
  echo "measure-scorer-call-cost: $SCORER is not executable" >&2
  exit 1
}

mkdir -p "$OUT_DIR"

# 1-minute load average, portable across macOS (BSD uptime) and Linux.
load_now() {
  uptime | sed -e 's/.*load average[s]*: *//' -e 's/,.*//' -e 's/ .*//'
}

echo "building the harness (release)..." >&2
cargo build --release --example scorer_call_cost_bench

IFS=',' read -r -a rate_list <<<"$RATES"
for rate in "${rate_list[@]}"; do
  label="rate-${rate//./_}"
  log="$OUT_DIR/$label.log"
  load_before="$(load_now)"
  echo "measuring sample-rate=$rate (loadBefore=$load_before) → $log" >&2
  {
    echo "# sample-rate: $rate"
    echo "# sizes: $SIZES  repeats: $REPEATS"
    echo "# creature: $CREATURE"
    echo "# trainingData: $TRAINING_DATA"
    echo "# scorer: $SCORER"
    echo "# loadBefore: $load_before"
    echo "# startedUtc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } >"$log"
  ./target/release/examples/scorer_call_cost_bench \
    "$CREATURE" "$TRAINING_DATA" "$SCORER" "$SIZES" "$rate" "$REPEATS" \
    >>"$log" 2>>"$log" </dev/null
  load_after="$(load_now)"
  {
    echo "# loadAfter: $load_after"
    echo "# finishedUtc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } >>"$log"
  echo "  loadAfter=$load_after" >&2
done

echo "measurement logs written under $OUT_DIR" >&2
