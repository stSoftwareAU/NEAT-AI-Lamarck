#!/usr/bin/env bash
# Turn one or more `experiments.jsonl` journals into the tables used by
# docs/screen-calibration.md (issue #110).
#
# Usage: scripts/summarise-screen-calibration.sh JOURNAL [JOURNAL...]
#   LAMARCK_BIN=path/to/neat_ai_lamarck   (default: target/release/neat_ai_lamarck)
#
# Every figure comes from `neat_ai_lamarck report`'s `screenCalibration`
# section, so the numbers in the document and the numbers the binary produces
# cannot drift apart.
set -euo pipefail

LAMARCK_BIN="${LAMARCK_BIN:-target/release/neat_ai_lamarck}"

command -v jq >/dev/null || {
  echo "summarise-screen-calibration: jq is required" >&2
  exit 1
}
[[ -x "$LAMARCK_BIN" ]] || {
  echo "summarise-screen-calibration: $LAMARCK_BIN is not executable — cargo build --release, or set LAMARCK_BIN" >&2
  exit 1
}
if [[ $# -eq 0 ]]; then
  echo "usage: summarise-screen-calibration.sh JOURNAL [JOURNAL...]" >&2
  exit 1
fi

for journal in "$@"; do
  [[ -f "$journal" ]] || {
    echo "summarise-screen-calibration: no such journal: $journal" >&2
    exit 1
  }
done

# Three-significant-figure scientific notation; jq has no printf("%e").
# shellcheck disable=SC2016  # $v is a jq parameter, not a shell variable.
SCI='def sci($v):
  if $v == null then "n/a"
  elif $v == 0 then "0"
  else (($v | fabs | log10 | floor) as $e
        | ((($v / pow(10; $e)) * 100 | round) / 100) as $m
        # Rounding 9.998 up to 10 carries into the exponent.
        | (if ($m | fabs) >= 10 then [$m / 10, $e + 1] else [$m, $e] end) as [$mm, $ee]
        | ($mm | tostring) + "e" + ($ee | tostring))
  end;
'

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT
POOLED="$WORK_DIR/pooled.jsonl"
: >"$POOLED"

# One row per journal, then a pooled row over every journal concatenated.
row() {
  local label="$1" report="$2"
  jq -r --arg label "$label" "$SCI"'
    .screenCalibration
    | def pct($v): if $v == null then "n/a" else (($v * 1000 | round) / 10 | tostring) + "%" end;
    "| " + $label
    + " | " + (.experiments | tostring)
    + " | " + ((.pairedCandidates + .screenOnlyCandidates) | tostring)
    + " | " + (.pairedCandidates | tostring)
    + " | " + (.distinctPairs | tostring)
    + " | " + (if .spearman == null then "n/a" else ((.spearman * 1000 | round) / 1000 | tostring) end)
    + " | " + (if .spearmanDistinct == null then "n/a" else ((.spearmanDistinct * 1000 | round) / 1000 | tostring) end)
    + " | " + pct(.promotionPrecision)
    + " | " + (.promotedClearingAcceptBar | tostring)
    + " | " + (.promotedMateriallyWorse | tostring)
    + " | " + sci(.screenNoise.stdDev)
    + " | " + sci(.baselineSampleGap.stdDev)
    + " |"
  ' "$report"
}

echo "| Journal | Exps | Screened | Paired | Distinct | Rank ρ | ρ distinct | Precision | Cleared bar | Materially worse | Screen-Δ noise sd | Baseline gap sd |"
echo "|---------|------|----------|--------|----------|--------|------------|-----------|-------------|------------------|-------------------|-----------------|"
for journal in "$@"; do
  label="$(basename "$(dirname "$journal")")"
  report="$WORK_DIR/$label.json"
  "$LAMARCK_BIN" report "$journal" >"$report"
  row "$label" "$report"
  cat "$journal" >>"$POOLED"
done

POOLED_REPORT="$WORK_DIR/pooled.json"
"$LAMARCK_BIN" report "$POOLED" >"$POOLED_REPORT"
row "**pooled**" "$POOLED_REPORT"

echo
echo "Screen Δ of every candidate that was ultimately accepted:"
echo
echo "| Experiment | Stem | Screen Δ | Full-corpus Δ |"
echo "|------------|------|----------|---------------|"
jq -r "$SCI"'
  .screenCalibration.acceptedCandidates[]
  | "| " + (.experimentNumber | tostring)
  + " | `" + .stem + "`"
  + " | " + (if .screenDelta == null then "not screened (combo)" else sci(.screenDelta) end)
  + " | " + sci(.fullDelta)
  + " |"
' "$POOLED_REPORT"

echo
echo "What a higher promote threshold would have kept (pooled promotions):"
echo
echo "| Threshold | Promotions kept | Share | Accepts kept |"
echo "|-----------|-----------------|-------|--------------|"
jq -r "$SCI"'
  .screenCalibration as $c
  | [1e-6, 2e-6, 3e-6, 4e-6, 5e-6][]
  | . as $t
  | ([$c.pairs[] | select(.screenDelta > $t)] | length) as $kept
  | ([$c.acceptedCandidates[] | select(.screenDelta != null and .screenDelta > $t)] | length) as $accepts
  | "| " + sci($t)
  + " | " + ($kept | tostring)
  + " | " + ((($kept / ($c.pairs | length)) * 1000 | round) / 10 | tostring) + "%"
  + " | " + ($accepts | tostring) + " / " + ([$c.acceptedCandidates[] | select(.screenDelta != null)] | length | tostring)
  + " |"
' "$POOLED_REPORT"
