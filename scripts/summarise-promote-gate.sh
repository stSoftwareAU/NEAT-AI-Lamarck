#!/usr/bin/env bash
# Turn one or more `experiments.jsonl` journals into the table used by
# docs/promote-gate.md (issue #111).
#
# Usage: scripts/summarise-promote-gate.sh JOURNAL [JOURNAL...]
#   LAMARCK_BIN=path/to/neat_ai_lamarck   (default: target/release/neat_ai_lamarck)
#
# Every figure comes from `neat_ai_lamarck report`'s `promoteGateReplay`
# section — the noise-aware gate replayed offline at its default σ̂ multiplier —
# so the document and the binary cannot drift apart.
set -euo pipefail

LAMARCK_BIN="${LAMARCK_BIN:-target/release/neat_ai_lamarck}"

command -v jq >/dev/null || {
  echo "summarise-promote-gate: jq is required" >&2
  exit 1
}
[[ -x "$LAMARCK_BIN" ]] || {
  echo "summarise-promote-gate: $LAMARCK_BIN is not executable — cargo build --release, or set LAMARCK_BIN" >&2
  exit 1
}
if [[ $# -eq 0 ]]; then
  echo "usage: summarise-promote-gate.sh JOURNAL [JOURNAL...]" >&2
  exit 1
fi

for journal in "$@"; do
  [[ -f "$journal" ]] || {
    echo "summarise-promote-gate: no such journal: $journal" >&2
    exit 1
  }
done

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT
POOLED="$WORK_DIR/pooled.jsonl"
: >"$POOLED"

row() {
  local label="$1" report="$2"
  jq -r --arg label "$label" '
    .promoteGateReplay
    | def pct($n; $d): if $d == 0 then "n/a" else (($n / $d * 1000 | round) / 10 | tostring) + "%" end;
    "| " + $label
    + " | " + (.gateAsRun // "none (pre-#111)")
    + " | " + (.screened | tostring)
    + " | " + (.promotedAsRun | tostring)
    + " | " + (.promotedUnderGate | tostring)
    + " | " + (.promotionsAvoided | tostring)
    + " | " + pct(.promotionsAvoided; .promotedAsRun)
    + " | " + (.acceptsKept | tostring) + " / " + ((.acceptsKept + .acceptsDropped) | tostring)
    + " |"
  ' "$report"
}

echo "| Journal | Gate as run | Screened | Promoted as run | Promoted under gate | Avoided | Avoided share | Accepts kept |"
echo "|---------|-------------|----------|-----------------|---------------------|---------|---------------|--------------|"
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
echo "Every accepted winner, replayed against the gate:"
echo
echo "| Experiment | Stem | Screen Δ | Gate demanded | σ̂ | Still promoted |"
echo "|------------|------|----------|---------------|-----|----------------|"
jq -r '
  def sci($v):
    if $v == null then "n/a"
    elif $v == 0 then "0"
    else (($v | fabs | log10 | floor) as $e
          | ((($v / pow(10; $e)) * 100 | round) / 100) as $m
          | (if ($m | fabs) >= 10 then [$m / 10, $e + 1] else [$m, $e] end) as [$mm, $ee]
          | ($mm | tostring) + "e" + ($ee | tostring))
    end;
  .promoteGateReplay.accepts[]
  | "| " + (.experimentNumber | tostring)
  + " | `" + .stem + "`"
  + " | " + sci(.screenDelta)
  + " | " + sci(.threshold)
  + " | " + sci(.sigma)
  + " | " + (if .wouldPromote then "yes" else "**no**" end)
  + " |"
' "$POOLED_REPORT"
