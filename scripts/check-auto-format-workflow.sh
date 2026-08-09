#!/usr/bin/env bash
# Validate the auto-format PR workflow (Issue #33).
#
# The auto-format workflow must:
#   1. Run on `pull_request` events only.
#   2. Declare minimal permissions (`contents: write`).
#   3. Invoke `cargo fmt --all`.
#   4. Invoke `cargo update -p neat-core` so Cargo.lock tracks NEAT-AI-core.
#   5. Gate commit/push behind a change-detection output (idempotent).
#   6. Refuse to push onto a fork's PR branch.
#   7. Use strict bash (`set -euo pipefail`).
#   8. Authenticate the push with ACTIONS_PUSH (GITHUB_TOKEN fallback).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKFLOW="${1:-$REPO_ROOT/.github/workflows/auto-format.yml}"
EXIT_CODE=0

usage() {
  cat <<'EOF'
Usage: check-auto-format-workflow.sh [WORKFLOW_PATH]

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
  fail "no pull_request trigger — auto-format must only run on PRs"
fi

if grep -qE '^[[:space:]]*permissions:[[:space:]]*write-all' "$WORKFLOW"; then
  fail "'permissions: write-all' grants more than this job needs (use contents: write)"
elif grep -qE '^[[:space:]]*contents:[[:space:]]*write' "$WORKFLOW"; then
  ok "minimal write permission (contents: write) present"
else
  fail "no 'contents: write' permission — the job cannot push fixes"
fi

if grep -qE 'cargo[[:space:]]+fmt[[:space:]]+--all' "$WORKFLOW"; then
  ok "cargo fmt --all invocation present"
else
  fail "no 'cargo fmt --all' invocation"
fi

if grep -qE 'cargo[[:space:]]+update[[:space:]]+-p[[:space:]]+neat-core' "$WORKFLOW"; then
  ok "cargo update -p neat-core lock sync present"
else
  fail "no 'cargo update -p neat-core' — PRs will leave Cargo.lock lagging NEAT-AI-core (Issue #33)"
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

if grep -qE 'set[[:space:]]+-euo[[:space:]]+pipefail' "$WORKFLOW"; then
  ok "strict bash (set -euo pipefail) present"
else
  fail "no 'set -euo pipefail' in run: blocks — failures may be swallowed"
fi

if grep -qE 'secrets\.ACTIONS_PUSH[[:space:]]*\|\|[[:space:]]*secrets\.GITHUB_TOKEN' "$WORKFLOW"; then
  ok "push authenticates with ACTIONS_PUSH (GITHUB_TOKEN fallback)"
else
  fail "no 'secrets.ACTIONS_PUSH || secrets.GITHUB_TOKEN' — bot pushes will gate PR checks behind Approve and run"
fi

exit "$EXIT_CODE"
