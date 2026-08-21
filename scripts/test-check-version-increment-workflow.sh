#!/usr/bin/env bash
# WHAT: check-version-increment-workflow.sh gates milestone PRs and a
# base-ref derived from the PR's own base branch (Issue #190).
#
# Remote `runlib`-style installs rebuild neat_ai_lamarck only when the crate
# version changes, so every PR that touches source must get a bump — including
# milestone sub-issue PRs, which merge into `milestone/<slug>` and never touch
# Develop. Diffing such a PR against a hardcoded `origin/Develop` also reads
# "already ahead of base" once the milestone branch carries one bump, so the
# validator must refuse a hardcoded base too.
#
# Runs the real validator against generated fixtures; asserts exit codes only.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECK="$SCRIPT_DIR/check-version-increment-workflow.sh"
WORKFLOW="$SCRIPT_DIR/../.github/workflows/version-increment.yml"
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
assert_exit "shipped version-increment.yml passes" 0 "$CHECK" "$WORKFLOW"

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

# A base ref hardcoded to Develop skips the bump on milestone PRs whose branch
# already carries one, so it must fail even though the milestone glob is there.
HARDCODED_BASE="$TMP_DIR/hardcoded-base.yml"
# shellcheck disable=SC2016  # the literal `${PR_BASE_REF}` is the match target
sed -e 's|origin/\${PR_BASE_REF}|origin/Develop|' \
  -e '/pull_request\.base\.ref/d' \
  -e '/github\.base_ref/d' "$WORKFLOW" >"$HARDCODED_BASE"
assert_exit "base ref hardcoded to origin/Develop → fail" 1 "$CHECK" "$HARDCODED_BASE"

# Existing rules still bite — the validator is not milestone-only.
NO_BUMP="$TMP_DIR/no-bump.yml"
grep -v 'bump-lamarck-version.sh' "$WORKFLOW" >"$NO_BUMP"
assert_exit "missing bump-lamarck-version.sh → fail" 1 "$CHECK" "$NO_BUMP"

# Unreadable path is an error, not a pass.
assert_exit "missing workflow file → error" 2 "$CHECK" "$TMP_DIR/absent.yml"

if [[ "$FAILS" -ne 0 ]]; then
  echo "FAIL: $FAILS assertion(s) failed" >&2
  exit 1
fi

echo "OK   all check-version-increment-workflow WHAT assertions passed"
exit 0
