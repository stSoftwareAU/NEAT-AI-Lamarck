#!/usr/bin/env bash
# WHAT: summarise-failed-cache-economics.sh reports per-run accepts (Issue #94).
#
# A zero-accept arm must stay visible — hiding it in a mean is how an
# underpowered pair gets treated as a decision. This runs the real summariser
# over a fixture tree.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUMMARISE="$SCRIPT_DIR/summarise-failed-cache-economics.sh"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

write_report() {
  local dir="$1"
  shift
  mkdir -p "$dir"
  python3 - "$dir/report.json" "$@" <<'PY'
import json, sys
path = sys.argv[1]
payload = json.loads(sys.argv[2])
path_parent = __import__("pathlib").Path(path)
path_parent.write_text(json.dumps(payload) + "\n")
PY
}

write_report "$TMP/control-seed1" '{"experiments":12,"acceptances":0,"totalScoreImprovement":0.0,"scoreImprovementPerWallHour":0.0,"cache":null}'
write_report "$TMP/treatment-seed1" '{"experiments":12,"acceptances":1,"totalScoreImprovement":1.2e-6,"scoreImprovementPerWallHour":4.8e-6,"cache":{"hitRate":0.2,"savedMs":100.0,"spentMs":10.0,"netMs":90.0,"stoodDownAtExperiment":null,"peakSize":40}}'

set +e
out="$("$SUMMARISE" "$TMP" 2>&1)"
status=$?
set -e

if [[ "$status" -ne 0 ]]; then
  echo "FAIL summariser exited $status" >&2
  echo "$out" >&2
  exit 1
fi

echo "$out" | grep -q 'control-seed1' || {
  echo "FAIL missing control row" >&2
  exit 1
}
echo "$out" | grep -q 'treatment-seed1' || {
  echo "FAIL missing treatment row" >&2
  exit 1
}
echo "$out" | grep -q 'Underpowered' || {
  echo "FAIL zero-accept arm was not flagged" >&2
  echo "$out" >&2
  exit 1
}

echo "OK   summarise-failed-cache-economics WHAT assertions passed"
exit 0
