#!/usr/bin/env bash
# WHAT: check-auto-format-workflow.sh gates the milestone branch filter (Issue #168).
#
# Milestone sub-issue PRs target `milestone/<slug>`, so the auto-format gate
# must list a milestone glob in its `pull_request.branches` filter. Runs the
# real validator against generated fixtures; asserts exit codes only.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECK="$SCRIPT_DIR/check-auto-format-workflow.sh"
WORKFLOW="$SCRIPT_DIR/../.github/workflows/auto-format.yml"
FAILS=0

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

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

# Baseline: the workflow this repo actually ships must satisfy every rule.
assert_exit "shipped auto-format.yml passes" 0 "$CHECK" "$WORKFLOW"

# Strip every milestone reference to build the regression fixture.
NO_MILESTONE="$TMP_DIR/no-milestone.yml"
grep -v 'milestone/' "$WORKFLOW" >"$NO_MILESTONE"
assert_exit "no milestone branch filter → fail" 1 "$CHECK" "$NO_MILESTONE"

# A prose mention is not a branch filter — comments must not satisfy the rule.
COMMENT_ONLY="$TMP_DIR/comment-only.yml"
{
  echo "# milestone/** PRs are handled elsewhere"
  cat "$NO_MILESTONE"
} >"$COMMENT_ONLY"
assert_exit "milestone only in a comment → fail" 1 "$CHECK" "$COMMENT_ONLY"

# Single-level glob in block sequence form is accepted.
SINGLE_STAR="$TMP_DIR/single-star.yml"
sed -E 's|^([[:space:]]*)- Develop$|\1- Develop\n\1- "milestone/*"|' "$NO_MILESTONE" >"$SINGLE_STAR"
assert_exit "block sequence 'milestone/*' → pass" 0 "$CHECK" "$SINGLE_STAR"

# Inline flow sequence form is accepted too.
INLINE="$TMP_DIR/inline.yml"
sed -E -e 's|^([[:space:]]*)branches:$|\1branches: [Develop, "milestone/**"]|' \
  -e '/^[[:space:]]*- Develop$/d' "$NO_MILESTONE" >"$INLINE"
assert_exit "inline flow sequence 'milestone/**' → pass" 0 "$CHECK" "$INLINE"

# Existing rules still bite — the validator is not milestone-only.
NO_FMT="$TMP_DIR/no-fmt.yml"
grep -v 'cargo fmt --all' "$WORKFLOW" >"$NO_FMT"
assert_exit "missing 'cargo fmt --all' → fail" 1 "$CHECK" "$NO_FMT"

# Unreadable path is an error, not a pass.
assert_exit "missing workflow file → error" 2 "$CHECK" "$TMP_DIR/absent.yml"

if [[ "$FAILS" -ne 0 ]]; then
  echo "FAIL: $FAILS assertion(s) failed" >&2
  exit 1
fi

echo "OK   all check-auto-format-workflow WHAT assertions passed"
exit 0
