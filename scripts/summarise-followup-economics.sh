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
echo "How close each strategy came (screen Δ is 5%-sample, full Δ is authoritative):"
echo
echo "| Arm | Strategy | Candidates | Best screen Δ | Promoted | Best full-corpus Δ |"
echo "|-----|----------|------------|---------------|----------|--------------------|"
for report in "${reports[@]}"; do
  dir="$(dirname "$report")"
  arm="$(basename "$dir")"
  jq -s -r --arg arm "$arm" '
    [ .[] | select(.experimentNumber) ] as $exps
    | [ $exps[]
        | (.screenScores.baseline) as $sb
        | (.scores.baseline) as $fb
        | . as $e
        | [ range(0; ($e.candidates | length)) as $i
            | ("candidate-" + ("00" + ($i | tostring) | .[-3:])) as $name
            | { strategy: $e.candidates[$i].strategy,
                screen: (($e.screenScores[$name] // null) as $s
                         | if $s == null or $sb == null then null else $s - $sb end),
                full: (($e.scores[$name] // null) as $s
                       | if $s == null or $fb == null then null else $s - $fb end) } ]
      ] | add // []
    | group_by(.strategy)
    | map({ strategy: .[0].strategy,
            candidates: length,
            bestScreen: ([.[].screen | select(. != null)] | if length > 0 then max else null end),
            promoted: ([.[].full | select(. != null)] | length),
            bestFull: ([.[].full | select(. != null)] | if length > 0 then max else null end) })
    | .[]
    | "| " + $arm + " | `" + .strategy + "` | " + (.candidates | tostring) + " | " +
      (if .bestScreen == null then "n/a" else (.bestScreen | tostring) end) + " | " +
      (.promoted | tostring) + " | " +
      (if .bestFull == null then "not promoted" else (.bestFull | tostring) end) + " |"
  ' "$dir/experiments.jsonl"
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
