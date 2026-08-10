#!/usr/bin/env bash
# Turn the #75 campaign's per-arm `report.json` files into the markdown tables
# used by docs/followup-economics.md.
#
# Usage: scripts/summarise-followup-economics.sh [out-dir]   # default .lamarck-followup
set -euo pipefail

OUT_DIR="${1:-.lamarck-followup}"

command -v jq >/dev/null || {
  echo "summarise-followup-economics: jq is required" >&2
  exit 1
}

shopt -s nullglob
reports=("$OUT_DIR"/*/report.json)
if [[ ${#reports[@]} -eq 0 ]]; then
  echo "summarise-followup-economics: no report.json under $OUT_DIR — run the campaign first" >&2
  exit 1
fi

echo "| Arm | Exps | Accepts | Full scores | Screen scores | Promote/scorer-min | Analysis share | Cumulative Δ | Wall (s) |"
echo "|-----|------|---------|-------------|---------------|--------------------|----------------|--------------|----------|"
for report in "${reports[@]}"; do
  arm="$(basename "$(dirname "$report")")"
  jq -r --arg arm "$arm" '
    [
      $arm,
      (.experiments | tostring),
      (.acceptances | tostring),
      (.candidatesScored | tostring),
      (.screenCandidatesScored | tostring),
      (.candidatesPerScorerMinute | . * 100 | round / 100 | tostring),
      ((.analysisTimeFraction // 0) * 1000 | round / 10 | tostring) + "%",
      (if .totalScoreImprovement == null then "n/a" else (.totalScoreImprovement | tostring) end),
      ((.wallDurationMs // 0) / 1000 | round | tostring)
    ] | "| " + join(" | ") + " |"
  ' "$report"
done

echo
echo "Per-arm strategy appearances and wins:"
echo
echo "| Arm | Strategy | Appearances | Wins | Combo wins | Acceptance rate |"
echo "|-----|----------|-------------|------|------------|-----------------|"
for report in "${reports[@]}"; do
  arm="$(basename "$(dirname "$report")")"
  jq -r --arg arm "$arm" '
    .strategies[] |
    "| " + $arm + " | `" + .strategy + "` | " + (.appearancesTotal | tostring) + " | " +
    (.wins | tostring) + " | " + (.comboWins | tostring) + " | " +
    ((.acceptanceRate * 10000 | round / 100 | tostring) + "%") + " |"
  ' "$report"
done
