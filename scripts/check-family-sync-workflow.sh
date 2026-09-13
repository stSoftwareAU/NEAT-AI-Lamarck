#!/usr/bin/env bash
# Validate the family-sync PR workflow (Issue #234).
#
# `scripts/runlib.sh` is a byte-identical copy of NEAT-AI-core's canonical
# script. The family-sync job is what keeps it that way, so a misdeclared job
# means silent drift: the copy rots while CI stays green. This gate fails the
# build instead.
#
# The workflow must:
#    1. Run on `pull_request` events, and not on `push`.
#    2. Run on *every* PR — no `paths:` filter. Drift arrives when core
#       changes, not when this PR touches scripts/.
#    3. Include a `milestone/<slug>` glob in the branch filter, so milestone
#       sub-issue PRs are synced too.
#    4. Declare `contents: write` and not `permissions: write-all`.
#    5. Refuse to push onto a fork's PR branch.
#    6. Disable checkout credential persistence.
#    7. Pin every `uses:` to a 40-character commit SHA.
#    8. Fetch the canonical copy from NEAT-AI-core's `Develop`.
#    9. Fail non-zero on a fetch error — never leave a stale copy reported as
#       in sync.
#   10. Compare byte-for-byte (`cmp`) rather than by mtime or by grep.
#   11. Gate commit/push behind a change-detection output (idempotent).
#   12. Rebase before pushing, so a concurrent push is not clobbered.
#   13. Use the App token -> ACTIONS_PUSH -> GITHUB_TOKEN auth chain.
#   14. Use strict bash (`set -euo pipefail`).
#
# Exit codes: 0 the workflow satisfies every rule, 1 at least one rule is
# broken, 2 invalid invocation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKFLOW="${1:-$REPO_ROOT/.github/workflows/family-sync.yml}"
EXIT_CODE=0

usage() {
  cat <<'EOF'
Usage: check-family-sync-workflow.sh [WORKFLOW_PATH]

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

ok() { echo "OK   $(basename "$WORKFLOW"): $*"; }
fail() {
  echo "FAIL $(basename "$WORKFLOW"): $*" >&2
  EXIT_CODE=1
}

# The workflow with comment lines removed. Every rule below is about what the
# workflow *does*, and a comment mentioning `paths:` or `push:` does not gate
# anything — matching prose would make the gate trivially satisfiable.
BODY="$(mktemp)"
trap 'rm -f "$BODY"' EXIT
sed 's/[[:space:]]*#.*$//' "$WORKFLOW" >"$BODY"

has() { grep -qE "$1" "$BODY"; }

# 1. pull_request only.
if has '^[[:space:]]*pull_request:'; then
  ok "pull_request trigger present"
else
  fail "no pull_request trigger — family-sync must run on PRs"
fi

if has '^[[:space:]]*push:'; then
  fail "a push trigger would re-sync after merge instead of before review — keep this on pull_request"
else
  ok "no push trigger"
fi

# 2. No paths filter: drift comes from core, not from this PR's own diff.
if has '^[[:space:]]*paths(-ignore)?:'; then
  fail "a 'paths:' filter would skip every PR that does not touch those paths, but runlib.sh drifts when NEAT-AI-core changes — remove the filter"
else
  ok "no paths filter — every PR is checked for drift"
fi

# 3. Milestone branch filter, as a real branch entry rather than prose.
milestone_branch_filter_present() {
  grep -qE '^[[:space:]]*-[[:space:]]*"?'"'"'?milestone/\*\*?"?'"'"'?[[:space:]]*$' "$BODY" && return 0
  grep -qE '^[[:space:]]*branches:[[:space:]]*\[[^]]*milestone/\*' "$BODY" && return 0
  return 1
}

if milestone_branch_filter_present; then
  ok "milestone branch filter present — milestone/<slug> PRs are synced"
else
  fail "no 'milestone/*' branch filter — milestone sub-issue PRs would merge a stale runlib.sh into the milestone branch"
fi

# 4. Least privilege.
if has '^[[:space:]]*permissions:[[:space:]]*write-all'; then
  fail "'permissions: write-all' grants more than this job needs (use contents: write)"
elif has '^[[:space:]]*contents:[[:space:]]*write'; then
  ok "minimal write permission (contents: write) present"
else
  fail "no 'contents: write' permission — the job cannot push the refreshed copy"
fi

# 5. Fork guard.
if has 'github\.event\.pull_request\.head\.repo\.full_name[[:space:]]*==' \
  || has 'github\.event\.pull_request\.head\.repo\.fork'; then
  ok "fork PRs are excluded from the push step"
else
  fail "no head.repo check — pushes onto forks will fail"
fi

# 6. Checkout credentials are not persisted.
if has '^[[:space:]]*persist-credentials:[[:space:]]*false'; then
  ok "checkout credential persistence disabled"
else
  fail "no 'persist-credentials: false' on checkout — the job must push with an explicitly minted token, not a persisted one"
fi

# 7. Every action pinned to a commit SHA.
unpinned=0
while IFS= read -r reference; do
  [[ -n "$reference" ]] || continue
  case "$reference" in
    ./*) continue ;; # a local composite action has no SHA to pin
  esac
  if [[ ! "$reference" =~ @[0-9a-f]{40}$ ]]; then
    fail "action '$reference' is not pinned to a 40-character commit SHA"
    unpinned=1
  fi
done < <(sed -n 's/^[[:space:]]*uses:[[:space:]]*//p' "$BODY" | tr -d '"'"'"'' | sed 's/[[:space:]]*$//')
if [[ "$unpinned" -eq 0 ]]; then
  ok "every action is pinned to a commit SHA"
fi

# 8. The canonical source is named.
if has 'stSoftwareAU/NEAT-AI-core' && has '(^|[^A-Za-z])Develop' && has 'scripts/runlib\.sh'; then
  ok "fetches scripts/runlib.sh from stSoftwareAU/NEAT-AI-core Develop"
else
  fail "the canonical source is not fully named — the job must fetch scripts/runlib.sh from stSoftwareAU/NEAT-AI-core Develop"
fi

# 9. A fetch error must fail the job.
if has '::error::' && has '^[[:space:]]*exit 1[[:space:]]*$'; then
  ok "a fetch error exits non-zero"
else
  fail "no explicit non-zero exit on a fetch error — a swallowed fetch would report a stale copy as in sync"
fi

# 10. Byte comparison.
if has '(^|[^[:alnum:]_-])cmp([[:space:]]|$)'; then
  ok "the committed copy is compared byte-for-byte (cmp)"
else
  fail "no 'cmp' byte comparison — 'byte-identical' cannot be asserted by grep or by timestamp"
fi

# 11. The step that actually pushes must carry the change-detection guard.
# Checking only that *some* step has one is not enough: the token-minting step
# is guarded too, so a guard lost from the push step alone would slip through
# and commit an empty change on every PR.
pushing_step_is_guarded() {
  awk '
    # A step begins at a `- name:` entry; everything until the next one belongs
    # to it.
    /^[[:space:]]*-[[:space:]]*name:/ {
      if (in_step && has_push) { print (has_guard ? "guarded" : "unguarded") }
      in_step = 1; has_push = 0; has_guard = 0
      next
    }
    !in_step { next }
    /^[[:space:]]*if:[[:space:]]*.*steps\.[A-Za-z0-9_-]+\.outputs\./ { has_guard = 1 }
    /git[^|]*push[[:space:]]+origin|[[:space:]]push[[:space:]]+origin/ { has_push = 1 }
    END {
      if (in_step && has_push) { print (has_guard ? "guarded" : "unguarded") }
    }
  ' "$BODY"
}

verdicts="$(pushing_step_is_guarded)"
if [[ -z "$verdicts" ]]; then
  fail "no step pushes to origin — the refreshed copy would never reach the PR branch"
elif printf '%s\n' "$verdicts" | grep -q 'unguarded'; then
  fail "the step that pushes has no 'if: steps.*.outputs.*' change-detection guard — every PR would get an empty commit"
else
  ok "the pushing step is conditional on change-detection output (idempotent guard)"
fi

# 12. Rebase before push.
if has '(^|[^[:alnum:]_-])rebase([[:space:]]|$)'; then
  ok "the branch is rebased before the push"
else
  fail "no rebase before push — a commit pushed by another job while this ran would be clobbered or the push rejected"
fi

# 13. Push auth chain.
if has 'steps\.[A-Za-z0-9_-]+\.outputs\.token[[:space:]]*\|\|' \
  && has 'secrets\.ACTIONS_PUSH[[:space:]]*\|\|' \
  && has 'secrets\.GITHUB_TOKEN'; then
  ok "push auth falls back App token -> ACTIONS_PUSH -> GITHUB_TOKEN"
else
  fail "push auth chain is not App token -> ACTIONS_PUSH -> GITHUB_TOKEN"
fi

# 14. Strict bash.
if has 'set -euo pipefail'; then
  ok "strict bash (set -euo pipefail) present"
else
  fail "no 'set -euo pipefail' in workflow run steps"
fi

# The commit subject doubles as the idempotency grep target.
if has 'chore: sync scripts/runlib\.sh from NEAT-AI-core Develop'; then
  ok "sync commit subject present"
else
  fail "missing the 'chore: sync scripts/runlib.sh from NEAT-AI-core Develop' commit subject"
fi

exit "$EXIT_CODE"
