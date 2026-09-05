#!/usr/bin/env bash
# Fold per-arm `report.json` files from run-strategy-allocation-ab.sh into
# markdown tables for docs/strategy-allocation.md (issue #218).
#
# Prints the gate metric (score improvement per wall hour) per arm, then the
# per-strategy allocation / return / cost rows each arm measured.
#
# Usage:
#   scripts/summarise-strategy-allocation.sh [.lamarck-strategy-allocation]
set -euo pipefail

OUT_DIR="${1:-.lamarck-strategy-allocation}"

[[ -d "$OUT_DIR" ]] || {
  echo "summarise-strategy-allocation: directory not found: $OUT_DIR" >&2
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

print("| Arm | Mode | Experiments | Accepts | Δ score | Δ / wall-hour |")
print("|-----|------|-------------|---------|---------|---------------|")
for name, data in arms:
    allocation = data.get("strategyAllocation") or {}
    print(
        f"| `{name}` | {fmt(allocation.get('mode'))} | "
        f"{fmt(data.get('experiments'), 0)} | {fmt(data.get('acceptances'), 0)} | "
        f"{fmt(data.get('totalScoreImprovement'))} | "
        f"{fmt(data.get('scoreImprovementPerWallHour'))} |"
    )

print()
print("| Arm | Strategy | Slots | Trials | Promotions | Accepts | Δ score | Cost (s) | Value |")
print("|-----|----------|-------|--------|------------|---------|---------|----------|-------|")
for name, data in arms:
    allocation = data.get("strategyAllocation") or {}
    for row in allocation.get("strategies", []):
        cost_ms = row.get("costMs")
        cost_s = None if cost_ms is None else cost_ms / 1000.0
        print(
            f"| `{name}` | `{row.get('strategy')}` | {fmt(row.get('allocatedSlots'), 0)} | "
            f"{fmt(row.get('trials'), 0)} | {fmt(row.get('promotions'), 0)} | "
            f"{fmt(row.get('accepts'), 0)} | {fmt(row.get('scoreGain'))} | "
            f"{fmt(cost_s, 4)} | {fmt(row.get('estimatedValue'), 4)} |"
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
