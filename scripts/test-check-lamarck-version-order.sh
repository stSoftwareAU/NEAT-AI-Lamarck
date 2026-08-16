#!/usr/bin/env bash
# WHAT: check-lamarck-version-order.sh refuses downgrades (Issue #152).
#
# Covers: behind → fail, equal → pass, ahead → pass. Also rejects malformed
# tokens. Runs the real script; asserts exit codes only.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORDER="$SCRIPT_DIR/check-lamarck-version-order.sh"
FAILS=0

assert_exit() {
  local label="$1" expected="$2"
  shift 2
  set +e
  "$@" >/dev/null 2>&1
  local got=$?
  set -e
  if [[ "$got" -eq "$expected" ]]; then
    echo "OK   $label (exit $got)"
  else
    echo "FAIL $label (expected exit $expected, got $got)" >&2
    FAILS=$((FAILS + 1))
  fi
}

assert_exit "behind → fail" 1 "$ORDER" "0.1.19" "0.1.18"
assert_exit "equal → pass" 0 "$ORDER" "0.1.19" "0.1.19"
assert_exit "ahead → pass" 0 "$ORDER" "0.1.19" "0.1.20"
assert_exit "minor ahead → pass" 0 "$ORDER" "0.1.19" "0.2.0"
assert_exit "leading v tolerated" 0 "$ORDER" "v0.1.19" "v0.1.20"
assert_exit "malformed base → error" 2 "$ORDER" "0.1" "0.1.19"
assert_exit "missing args → error" 2 "$ORDER" "0.1.19"

if [[ "$FAILS" -ne 0 ]]; then
  echo "FAIL: $FAILS assertion(s) failed" >&2
  exit 1
fi

echo "OK   all check-lamarck-version-order WHAT assertions passed"
exit 0
