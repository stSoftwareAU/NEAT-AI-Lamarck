#!/usr/bin/env bash
# Fold per-arm `report.json` files from run-strategy-priors-ab.sh into markdown
# tables for docs/strategy-priors.md (issue #221).
#
# Prints the gate metric (score improvement per wall hour) per arm with the
# confidence each seeded arm opened on, then the per-strategy split of inherited
# against measured evidence.
#
# Usage:
#   scripts/summarise-strategy-priors.sh [.lamarck-strategy-priors]
set -euo pipefail

OUT_DIR="${1:-.lamarck-strategy-priors}"

[[ -d "$OUT_DIR" ]] || {
  echo "summarise-strategy-priors: directory not found: $OUT_DIR" >&2
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

print("| Arm | Priors | Confidence | Seeded trials | Experiments | Accepts | Δ score | Δ / wall-hour |")
print("|-----|--------|------------|---------------|-------------|---------|---------|---------------|")
for name, data in arms:
    allocation = data.get("strategyAllocation") or {}
    priors = allocation.get("priors") or {}
    print(
        f"| `{name}` | {fmt(allocation.get('priorsMode'))} | "
        f"{fmt(priors.get('confidence'), 4)} | {fmt(priors.get('seededTrials'), 4)} | "
        f"{fmt(data.get('experiments'), 0)} | {fmt(data.get('acceptances'), 0)} | "
        f"{fmt(data.get('totalScoreImprovement'))} | "
        f"{fmt(data.get('scoreImprovementPerWallHour'))} |"
    )

print()
print("| Arm | Strategy | Prior trials | Prior share | Trials | Accepts | Δ score | Value |")
print("|-----|----------|--------------|-------------|--------|---------|---------|-------|")
for name, data in arms:
    allocation = data.get("strategyAllocation") or {}
    for row in allocation.get("strategies", []):
        print(
            f"| `{name}` | `{row.get('strategy')}` | {fmt(row.get('priorTrials'), 4)} | "
            f"{fmt(row.get('priorShare'), 4)} | {fmt(row.get('trials'), 0)} | "
            f"{fmt(row.get('accepts'), 0)} | {fmt(row.get('scoreGain'))} | "
            f"{fmt(row.get('estimatedValue'), 4)} |"
        )

rejected = [
    (name, (data.get("strategyAllocation") or {}).get("priors", {}).get("rejected"))
    for name, data in arms
]
rejected = [(name, why) for name, why in rejected if why]
if rejected:
    print()
    print(
        "**Seeded nothing:** "
        + ", ".join(f"`{name}` ({why})" for name, why in rejected)
        + " — those arms ran cold, so they measure the control, not the treatment."
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
PY
