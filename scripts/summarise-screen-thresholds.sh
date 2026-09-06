#!/usr/bin/env bash
# Fold per-arm `report.json` files from run-screen-threshold-ab.sh into
# markdown tables for docs/screen-thresholds.md (issue #220).
#
# Prints the gate metric (score improvement per wall hour) and the full-corpus
# calls each arm bought, then the per-strategy calibration rows each arm
# measured — including the control promotions and the false negatives among
# them, which is the only false-negative evidence a screen can produce.
#
# Usage:
#   scripts/summarise-screen-thresholds.sh [.lamarck-screen-thresholds]
set -euo pipefail

OUT_DIR="${1:-.lamarck-screen-thresholds}"

[[ -d "$OUT_DIR" ]] || {
  echo "summarise-screen-thresholds: directory not found: $OUT_DIR" >&2
  exit 1
}

python3 - "$OUT_DIR" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])


def fmt(value, digits=6):
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


arms = []
for report_path in sorted(root.glob("*/report.json")):
    data = json.loads(report_path.read_text())
    arms.append((report_path.parent.name, data))

if not arms:
    print("No report.json files under", root, file=sys.stderr)
    sys.exit(1)

print("| Arm | Mode | Experiments | Accepts | Promoted | Controls | Δ score | Δ / wall-hour |")
print("|-----|------|-------------|---------|----------|----------|---------|---------------|")
for name, data in arms:
    replay = data.get("screenThresholdReplay") or {}
    print(
        f"| `{name}` | {fmt(replay.get('modeAsRun'))} | "
        f"{fmt(data.get('experiments'), 0)} | {fmt(data.get('acceptances'), 0)} | "
        f"{fmt(replay.get('promotedAsRun'), 0)} | "
        f"{fmt(replay.get('controlPromotions'), 0)} | "
        f"{fmt(data.get('totalScoreImprovement'))} | "
        f"{fmt(data.get('scoreImprovementPerWallHour'))} |"
    )

print()
print("| Arm | Strategy | Screened | Paired | Precision | Controls | False neg | Cost (s) | Threshold | Basis |")
print("|-----|----------|----------|--------|-----------|----------|-----------|----------|-----------|-------|")
for name, data in arms:
    calibration = data.get("screenCalibration") or {}
    for row in calibration.get("byStrategy", []):
        promote_ms = row.get("promoteMs")
        promote_s = None if promote_ms is None else promote_ms / 1000.0
        precision = row.get("promotionPrecision")
        print(
            f"| `{name}` | `{row.get('strategy')}` | {fmt(row.get('screened'), 0)} | "
            f"{fmt(row.get('paired'), 0)} | "
            f"{'unavailable' if precision is None else f'{precision * 100:.0f}%'} | "
            f"{fmt(row.get('controlPromotions'), 0)} | "
            f"{fmt(row.get('controlFalseNegatives'), 0)} | "
            f"{fmt(promote_s, 4)} | {fmt(row.get('recommendedThreshold'), 4)} | "
            f"{fmt(row.get('basis'))} |"
        )

zero_accept = [name for name, data in arms if (data.get("acceptances") or 0) == 0]
if zero_accept:
    print()
    print(
        "**Underpowered:** zero accepts on "
        + ", ".join(f"`{arm}`" for arm in zero_accept)
        + " — those arms cannot distinguish the treatment; re-run longer or with"
        " more repeats."
    )

no_controls = [
    name
    for name, data in arms
    if (data.get("screenThresholdReplay") or {}).get("modeAsRun") == "per-strategy"
    and not any(
        (row.get("controlPromotions") or 0) > 0
        for row in (data.get("screenCalibration") or {}).get("byStrategy", [])
    )
]
if no_controls:
    print()
    print(
        "**No false-negative evidence:** no control promotion was journalled on "
        + ", ".join(f"`{arm}`" for arm in no_controls)
        + " — a calibrated arm without controls cannot show what its thresholds"
        " threw away."
    )
PY
