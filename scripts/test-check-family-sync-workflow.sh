#!/usr/bin/env bash
# WHAT: check-family-sync-workflow.sh refuses a misdeclared family-sync job
# (Issue #234).
#
# `scripts/runlib.sh` is a byte-identical copy of NEAT-AI-core's canonical
# script, and the family-sync job is the only thing keeping it that way. Every
# rule below is a way that job can be broken while CI still looks green, so the
# validator must catch each one. These tests mutate the shipped workflow one
# rule at a time and assert the validator's exit code.
#
# Runs the real validator against generated fixtures; asserts exit codes only.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECK="$SCRIPT_DIR/check-family-sync-workflow.sh"
WORKFLOW="$SCRIPT_DIR/../.github/workflows/family-sync.yml"
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
assert_exit "shipped family-sync.yml passes" 0 "$CHECK" "$WORKFLOW"

# A missing file is an invocation error, not a rule failure.
assert_exit "missing workflow file → exit 2" 2 "$CHECK" "$TMP_DIR/absent.yml"

# Rule 1 — the trigger.
NO_PR="$TMP_DIR/no-pull-request.yml"
grep -v '^[[:space:]]*pull_request:' "$WORKFLOW" >"$NO_PR"
assert_exit "no pull_request trigger → fail" 1 "$CHECK" "$NO_PR"

PUSH_TRIGGER="$TMP_DIR/push-trigger.yml"
sed -E 's|^(on:)$|\1\n  push:\n    branches: [Develop]|' "$WORKFLOW" >"$PUSH_TRIGGER"
assert_exit "push trigger → fail" 1 "$CHECK" "$PUSH_TRIGGER"

# Rule 2 — a paths filter would skip the PRs that do not touch scripts/, which
# is most of them; drift arrives from core regardless of this PR's diff.
PATHS_FILTER="$TMP_DIR/paths-filter.yml"
sed -E 's|^([[:space:]]*)- "milestone/\*\*"$|\1- "milestone/**"\n\1paths:\n\1  - "scripts/**"|' "$WORKFLOW" >"$PATHS_FILTER"
assert_exit "paths filter → fail" 1 "$CHECK" "$PATHS_FILTER"

# Rule 3 — the milestone glob, and prose about it is not a branch filter.
NO_MILESTONE="$TMP_DIR/no-milestone.yml"
grep -v 'milestone/' "$WORKFLOW" >"$NO_MILESTONE"
assert_exit "no milestone branch filter → fail" 1 "$CHECK" "$NO_MILESTONE"

COMMENT_ONLY="$TMP_DIR/comment-only.yml"
{
  echo "# milestone/** PRs are handled elsewhere"
  cat "$NO_MILESTONE"
} >"$COMMENT_ONLY"
assert_exit "milestone only in a comment → fail" 1 "$CHECK" "$COMMENT_ONLY"

INLINE_MILESTONE="$TMP_DIR/inline-milestone.yml"
sed -E 's|^([[:space:]]*)branches:$|\1branches: [Develop, "milestone/**"]|' "$NO_MILESTONE" >"$INLINE_MILESTONE"
assert_exit "inline flow sequence milestone glob → pass" 0 "$CHECK" "$INLINE_MILESTONE"

# Rule 4 — least privilege.
WRITE_ALL="$TMP_DIR/write-all.yml"
sed -E 's|^([[:space:]]*)contents: write$|\1permissions: write-all|' "$WORKFLOW" >"$WRITE_ALL"
assert_exit "permissions: write-all → fail" 1 "$CHECK" "$WRITE_ALL"

NO_WRITE="$TMP_DIR/no-write.yml"
sed -E 's|^([[:space:]]*)contents: write$|\1contents: read|' "$WORKFLOW" >"$NO_WRITE"
assert_exit "no contents: write → fail" 1 "$CHECK" "$NO_WRITE"

# Rule 5 — the fork guard.
NO_FORK_GUARD="$TMP_DIR/no-fork-guard.yml"
grep -v 'head\.repo\.full_name' "$WORKFLOW" >"$NO_FORK_GUARD"
assert_exit "no fork guard → fail" 1 "$CHECK" "$NO_FORK_GUARD"

# Rule 6 — checkout credential persistence.
PERSISTED="$TMP_DIR/persisted-credentials.yml"
sed -E 's|^([[:space:]]*)persist-credentials: false$|\1persist-credentials: true|' "$WORKFLOW" >"$PERSISTED"
assert_exit "persist-credentials: true → fail" 1 "$CHECK" "$PERSISTED"

# Rule 7 — SHA pins. A floating tag is the classic supply-chain hole.
FLOATING_TAG="$TMP_DIR/floating-tag.yml"
sed -E 's|uses: actions/checkout@[0-9a-f]{40}|uses: actions/checkout@v5|' "$WORKFLOW" >"$FLOATING_TAG"
assert_exit "floating action tag → fail" 1 "$CHECK" "$FLOATING_TAG"

# Rule 8 — the canonical source must be named in full.
WRONG_SOURCE="$TMP_DIR/wrong-source.yml"
sed -E 's|stSoftwareAU/NEAT-AI-core|stSoftwareAU/NEAT-AI-somewhere-else|g' "$WORKFLOW" >"$WRONG_SOURCE"
assert_exit "canonical repo not named → fail" 1 "$CHECK" "$WRONG_SOURCE"

# Rule 9 — a swallowed fetch error is the silent-failure shape this gate exists
# to refuse: a stale copy would be reported as in sync.
SWALLOWED="$TMP_DIR/swallowed-fetch.yml"
grep -v '^[[:space:]]*exit 1$' "$WORKFLOW" >"$SWALLOWED"
assert_exit "no non-zero exit on fetch error → fail" 1 "$CHECK" "$SWALLOWED"

# Rule 10 — byte comparison, not a grep or a timestamp.
NO_CMP="$TMP_DIR/no-cmp.yml"
sed -E 's|if cmp -s|if grep -q sentinel|' "$WORKFLOW" >"$NO_CMP"
assert_exit "no cmp byte comparison → fail" 1 "$CHECK" "$NO_CMP"

# Rule 11 — the idempotency guard; without it every PR gets an empty commit.
NO_GUARD="$TMP_DIR/no-guard.yml"
grep -v "steps\.compare\.outputs\.changed" "$WORKFLOW" >"$NO_GUARD"
assert_exit "no change-detection guard → fail" 1 "$CHECK" "$NO_GUARD"

# Losing the guard on the *pushing* step alone must still fail: the
# token-minting step keeps its own guard, so a rule that only asks whether any
# step is guarded would wave this through and commit an empty change per PR.
PUSH_UNGUARDED="$TMP_DIR/push-unguarded.yml"
awk '
  /^      - name: Commit and push the refreshed runlib\.sh$/ { in_push = 1; print; next }
  in_push && /^[[:space:]]*if: steps\.compare\.outputs\.changed/ { in_push = 0; next }
  { print }
' "$WORKFLOW" >"$PUSH_UNGUARDED"
assert_exit "push step loses its guard → fail" 1 "$CHECK" "$PUSH_UNGUARDED"

# A workflow that never pushes cannot sync anything.
NO_PUSH="$TMP_DIR/no-push.yml"
grep -v 'push origin' "$WORKFLOW" >"$NO_PUSH"
assert_exit "nothing pushes to origin → fail" 1 "$CHECK" "$NO_PUSH"

# Rule 12 — rebase before push.
NO_REBASE="$TMP_DIR/no-rebase.yml"
grep -v 'rebase' "$WORKFLOW" >"$NO_REBASE"
assert_exit "no rebase before push → fail" 1 "$CHECK" "$NO_REBASE"

# Rule 13 — the push auth chain.
NO_FALLBACK="$TMP_DIR/no-fallback.yml"
sed -E 's|GH_PAT: .*|GH_PAT: ${{ secrets.GITHUB_TOKEN }}|' "$WORKFLOW" >"$NO_FALLBACK"
assert_exit "push auth without the App/ACTIONS_PUSH chain → fail" 1 "$CHECK" "$NO_FALLBACK"

# Rule 14 — strict bash.
NO_STRICT="$TMP_DIR/no-strict.yml"
grep -v 'set -euo pipefail' "$WORKFLOW" >"$NO_STRICT"
assert_exit "no set -euo pipefail → fail" 1 "$CHECK" "$NO_STRICT"

# The commit subject is the idempotency grep target.
NO_SUBJECT="$TMP_DIR/no-subject.yml"
sed -E 's|chore: sync scripts/runlib\.sh from NEAT-AI-core Develop|chore: update things|' "$WORKFLOW" >"$NO_SUBJECT"
assert_exit "missing sync commit subject → fail" 1 "$CHECK" "$NO_SUBJECT"

if [[ "$FAILS" -ne 0 ]]; then
  echo "test-check-family-sync-workflow: $FAILS failure(s)" >&2
  exit 1
fi
echo "test-check-family-sync-workflow: all assertions passed"
