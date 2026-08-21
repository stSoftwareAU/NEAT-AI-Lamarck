#!/usr/bin/env bash
# Validate the version-increment PR workflow (GRQ-taxation runlib contract).
#
# The workflow must:
#   1. Run on `pull_request` events only.
#   2. Declare minimal permissions (`contents: write`).
#   3. Invoke `scripts/bump-lamarck-version.sh`.
#   4. Gate commit/push behind a change-detection output (idempotent).
#   5. Refuse to push onto a fork's PR branch.
#   6. Use strict bash (`set -euo pipefail`).
#   7. Include a `milestone/<slug>` glob in the branch filter so milestone
#      sub-issue PRs are bumped too (Issue #190).
#   8. Diff against the PR's own base branch rather than a hardcoded
#      `origin/Develop` (Issue #190).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKFLOW="${1:-$REPO_ROOT/.github/workflows/version-increment.yml}"
EXIT_CODE=0

usage() {
  cat <<'EOF'
Usage: check-version-increment-workflow.sh [WORKFLOW_PATH]

Exits 0 when the workflow satisfies every rule listed in the script header.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -f "$WORKFLOW" ]]; then
  echo "FAIL: workflow file not found: $WORKFLOW" >&2
  exit 2
fi

ok() { echo "OK   $WORKFLOW: $*"; }
fail() {
  echo "FAIL $WORKFLOW: $*" >&2
  EXIT_CODE=1
}

if grep -qE '^[[:space:]]*pull_request:' "$WORKFLOW"; then
  ok "pull_request trigger present"
else
  fail "no pull_request trigger — version-increment must only run on PRs"
fi

if grep -qE '^[[:space:]]*permissions:[[:space:]]*write-all' "$WORKFLOW"; then
  fail "'permissions: write-all' grants more than this job needs (use contents: write)"
elif grep -qE '^[[:space:]]*contents:[[:space:]]*write' "$WORKFLOW"; then
  ok "minimal write permission (contents: write) present"
else
  fail "no 'contents: write' permission — the job cannot push bumps"
fi

if grep -qE 'bump-lamarck-version\.sh' "$WORKFLOW"; then
  ok "bump-lamarck-version.sh invocation present"
else
  fail "no bump-lamarck-version.sh invocation"
fi

# A milestone glob must appear as a real branch entry — either a block
# sequence item (`- "milestone/**"`) or inside an inline flow sequence
# (`branches: [Develop, "milestone/**"]`). A prose mention in a comment does
# not gate anything, so comments deliberately do not satisfy this rule.
milestone_branch_filter_present() {
  grep -qE '^[[:space:]]*-[[:space:]]*"?'"'"'?milestone/\*\*?"?'"'"'?[[:space:]]*$' "$WORKFLOW" && return 0
  grep -qE '^[[:space:]]*branches:[[:space:]]*\[[^]]*milestone/\*' "$WORKFLOW" && return 0
  return 1
}

if milestone_branch_filter_present; then
  ok "milestone branch filter present — milestone/<slug> PRs are bumped"
else
  fail "no 'milestone/*' branch filter — milestone sub-issue PRs merge with a stale crate version, so remote runlib installs keep the old binary (Issue #190)"
fi

# The bump must diff against the branch the PR actually targets. Hardcoding
# `origin/Develop` skips every milestone PR after the first bump on that
# milestone branch: the head already reads "ahead of Develop".
base_ref_is_dynamic() {
  grep -qE 'github\.(base_ref|event\.pull_request\.base\.ref)' "$WORKFLOW" || return 1
  ! grep -qE -- '--base-ref[[:space:]]+"?origin/[Dd]evelop"?' "$WORKFLOW"
}

if base_ref_is_dynamic; then
  ok "bump diffs against the PR's own base branch (github.base_ref)"
else
  fail "base ref is hardcoded to origin/Develop — milestone PRs would read 'already ahead of base' and skip the bump (Issue #190)"
fi

if grep -qE '^[[:space:]]*if:[[:space:]]*steps\.[A-Za-z0-9_-]+\.outputs\.' "$WORKFLOW"; then
  ok "commit/push is conditional on change-detection output (idempotent guard)"
else
  fail "no conditional 'if: steps.*.outputs.*' guard — commit/push must be conditional"
fi

if grep -qE 'github\.event\.pull_request\.head\.repo\.full_name[[:space:]]*==' "$WORKFLOW" \
  || grep -qE 'github\.event\.pull_request\.head\.repo\.fork' "$WORKFLOW"; then
  ok "fork PRs are excluded from the push step"
else
  fail "no head.repo check — pushes onto forks will fail silently"
fi

if grep -qE 'set -euo pipefail' "$WORKFLOW"; then
  ok "strict bash (set -euo pipefail) present"
else
  fail "no 'set -euo pipefail' in workflow run steps"
fi

if grep -qE 'chore: auto-increment versions for changed projects' "$WORKFLOW"; then
  ok "auto-increment commit subject present (idempotency grep target)"
else
  fail "missing auto-increment commit subject"
fi

exit "$EXIT_CODE"
