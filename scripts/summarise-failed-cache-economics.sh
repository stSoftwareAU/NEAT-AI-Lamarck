#!/usr/bin/env bash
# Fold per-arm `report.json` files from run-failed-cache-economics.sh into a
# markdown table for docs/failed-candidate-cache-economics.md (issue #94).
#
# Usage:
#   scripts/summarise-failed-cache-economics.sh [.lamarck-failed-cache]
set -euo pipefail

OUT_DIR="${1:-.lamarck-failed-cache}"

[[ -d "$OUT_DIR" ]] || {
  echo "summarise-failed-cache-economics: directory not found: $OUT_DIR" >&2
  exit 1
}

python3 - "$OUT_DIR" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for report_path in sorted(root.glob("*/report.json")):
    arm_dir = report_path.parent.name
    data = json.loads(report_path.read_text())
    cache = data.get("cache") or {}
    rows.append({
        "arm": arm_dir,
        "experiments": data.get("experiments"),
        "acceptances": data.get("acceptances"),
        "improvement": data.get("totalScoreImprovement"),
        "per_hour": data.get("scoreImprovementPerWallHour"),
        "hit_rate": cache.get("hitRate"),
        "saved_ms": cache.get("savedMs"),
        "spent_ms": cache.get("spentMs"),
        "net_ms": cache.get("netMs"),
        "stood_down": cache.get("stoodDownAtExperiment"),
        "peak_entries": cache.get("peakSize"),
    })

if not rows:
    print("No report.json files under", root, file=sys.stderr)
    sys.exit(1)

print("| Arm | Experiments | Accepts | Δ score | Δ / wall-hour | Hit rate | Saved ms | Spent ms | Net ms | Stood down | Peak entries |")
print("|-----|-------------|---------|---------|---------------|----------|----------|----------|--------|------------|--------------|")
for r in rows:
    def fmt(v, digits=6):
        if v is None:
            return "unavailable"
        if isinstance(v, float):
            return f"{v:.{digits}g}"
        return str(v)
    print(
        f"| `{r['arm']}` | {fmt(r['experiments'],0)} | {fmt(r['acceptances'],0)} | "
        f"{fmt(r['improvement'])} | {fmt(r['per_hour'])} | {fmt(r['hit_rate'],4)} | "
        f"{fmt(r['saved_ms'],4)} | {fmt(r['spent_ms'],4)} | {fmt(r['net_ms'],4)} | "
        f"{fmt(r['stood_down'],0)} | {fmt(r['peak_entries'],0)} |"
    )

zero_accept = [r["arm"] for r in rows if (r["acceptances"] or 0) == 0]
if zero_accept:
    print()
    print(
        "**Underpowered:** zero accepts on "
        + ", ".join(f"`{a}`" for a in zero_accept)
        + " — those arms cannot distinguish the treatment; re-run longer or with more repeats."
    )
PY
