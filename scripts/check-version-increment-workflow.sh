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
