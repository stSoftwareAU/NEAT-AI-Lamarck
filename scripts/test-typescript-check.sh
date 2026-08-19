#!/usr/bin/env bash
# WHAT: typescript-check.sh is a real basic-validity gate (Issue #167).
#
# Runs the real script against throwaway roots and asserts exit codes only:
# broken TypeScript must fail loudly, valid TypeScript must pass, and a
# missing `deno` must fail rather than silently reporting success.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CHECK="$SCRIPT_DIR/typescript-check.sh"
FAILS=0

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

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

# --- fixtures -------------------------------------------------------------
mkdir -p "$WORK/valid" "$WORK/syntax" "$WORK/types" "$WORK/empty" "$WORK/skipped/target"

cat > "$WORK/valid/ok.ts" <<'TS'
export function add(a: number, b: number): number {
  return a + b;
}
TS

# Unbalanced brace — a plain syntax error.
cat > "$WORK/syntax/broken.ts" <<'TS'
export function broken(a: number): number {
  return a + 1;
TS

# Parses fine, but the types do not line up.
cat > "$WORK/types/mismatch.ts" <<'TS'
export function count(): number {
  const label: string = 42;
  return label.length;
}
TS

# Excluded roots must not be scanned, even when they hold broken sources.
cp "$WORK/syntax/broken.ts" "$WORK/skipped/target/broken.ts"

# --- assertions -----------------------------------------------------------
assert_exit "valid TypeScript → pass" 0 "$CHECK" --root "$WORK/valid"
assert_exit "syntax error → fail" 1 "$CHECK" --root "$WORK/syntax"
assert_exit "type error → fail" 1 "$CHECK" --root "$WORK/types"
assert_exit "no TypeScript files → pass" 0 "$CHECK" --root "$WORK/empty"
assert_exit "target/ excluded → pass" 0 "$CHECK" --root "$WORK/skipped"
assert_exit "repository sources → pass" 0 "$CHECK" --root "$REPO_ROOT"
assert_exit "missing root → usage error" 2 "$CHECK" --root "$WORK/nope"
assert_exit "unknown option → usage error" 2 "$CHECK" --nonsense
assert_exit "--root without value → usage error" 2 "$CHECK" --root
assert_exit "--help → pass" 0 "$CHECK" --help

# Fail loud when the toolchain is absent: no deno must never look like success.
# `bash` is invoked by absolute path because the stubbed PATH resolves nothing.
BASH_BIN="$(command -v bash)"
assert_exit "deno unavailable → fail" 1 \
  env -i PATH=/nonexistent "$BASH_BIN" "$CHECK" --root "$WORK/valid"

if [[ "$FAILS" -ne 0 ]]; then
  echo "FAIL: $FAILS assertion(s) failed" >&2
  exit 1
fi

echo "OK   all typescript-check WHAT assertions passed"
exit 0
