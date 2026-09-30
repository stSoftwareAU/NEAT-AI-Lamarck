#!/usr/bin/env bash
# Validate the shared bot-push composite action and its callers (Issue #252).
#
# `.github/actions/bot-push/action.yml` is the one home of the mint-token +
# commit + push logic the bot workflows share. Three copy-pasted copies had
# already drifted apart; this gate keeps the logic hardened and in one place.
#
# The action must:
#    1. Be a composite action.
#    2. Pin every `uses:` to a 40-character commit SHA.
#    3. Mint a repo-scoped App token with only `contents: write`.
#    4. Authenticate with App token -> ACTIONS_PUSH (`inputs.push-pat`) ->
#       GITHUB_TOKEN (`github.token`), in that order.
#    5. Run git and base64 from absolute paths, and disable hooks on every git
#       call — a bare `git` or a hook-enabled call could run repository code
#       with the token in the environment.
#    6. Never interpolate `${{ … }}` into the `run:` script; inputs travel
#       through `env:` so a branch name or commit message cannot inject shell.
#    7. Use strict bash (`set -euo pipefail`).
#    8. Tell a deleted head branch (`ls-remote` exit 2) apart from an
#       unreachable origin, which must fail rather than skip the push.
#    9. Rebase before pushing, aborting a conflicted rebase.
#   10. Push through a per-command `extraheader`, and never force-push.
#
# Every workflow must:
#   11. Carry no inline copy of the push logic (`create-github-app-token`,
#       `extraheader`, `push origin`) — it delegates to the action.
#   12. Pass the App secrets and ACTIONS_PUSH on every call to the action, so
#       no caller silently falls back to GITHUB_TOKEN (whose pushes start no
#       new workflow run).
#
# Exit codes: 0 every rule holds, 1 at least one rule is broken, 2 invalid
# invocation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ACTION="${1:-$REPO_ROOT/.github/actions/bot-push/action.yml}"
WORKFLOWS_DIR="${2:-$REPO_ROOT/.github/workflows}"
EXIT_CODE=0

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: check-bot-push-action.sh [ACTION_PATH [WORKFLOWS_DIR]]

Exits 0 when the action and every workflow satisfy the rules in the header.
EOF
  exit 0
fi

if [[ ! -f "$ACTION" ]]; then
  echo "FAIL: action file not found: $ACTION" >&2
  exit 2
fi
if [[ ! -d "$WORKFLOWS_DIR" ]]; then
  echo "FAIL: workflows directory not found: $WORKFLOWS_DIR" >&2
  exit 2
fi

ok() { echo "OK   bot-push: $*"; }
fail() {
  echo "FAIL bot-push: $*" >&2
  EXIT_CODE=1
}

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

# Comments removed: every rule is about what the file does, and prose that
# mentions a rule's pattern must not satisfy (or trip) it.
strip_comments() { sed 's/[[:space:]]*#.*$//' "$1"; }

BODY="$TMP_DIR/action.yml"
strip_comments "$ACTION" >"$BODY"
has() { grep -qE -- "$1" "$BODY"; }

# 1. Composite.
if has '^[[:space:]]*using:[[:space:]]*"?composite"?[[:space:]]*$'; then
  ok "composite action"
else
  fail "not a composite action — the push must run in the caller's job, on its working tree"
fi

# 2. SHA pins.
unpinned=0
while IFS= read -r reference; do
  [[ -n "$reference" ]] || continue
  if [[ ! "$reference" =~ @[0-9a-f]{40}$ ]]; then
    fail "action '$reference' is not pinned to a 40-character commit SHA"
    unpinned=1
  fi
done < <(sed -n 's/^[[:space:]]*-\{0,1\}[[:space:]]*uses:[[:space:]]*//p' "$BODY" | tr -d '"'"'"'' | sed 's/[[:space:]]*$//')
if [[ "$unpinned" -eq 0 ]]; then
  ok "every action is pinned to a commit SHA"
fi

# 3. Repo-scoped, contents-only App token.
if has 'uses:[[:space:]]*actions/create-github-app-token@' \
  && has '^[[:space:]]*permission-contents:[[:space:]]*write[[:space:]]*$' \
  && has '^[[:space:]]*repositories:[[:space:]]*\$\{\{[[:space:]]*github\.event\.repository\.name[[:space:]]*\}\}'; then
  ok "mints a token scoped to this repository with contents: write"
else
  fail "the App token is not minted repo-scoped with 'permission-contents: write'"
fi

# 4. Auth chain, in order.
if has 'steps\.[A-Za-z0-9_-]+\.outputs\.token[[:space:]]*\|\|[[:space:]]*inputs\.push-pat[[:space:]]*\|\|[[:space:]]*github\.token'; then
  ok "push auth falls back App token -> ACTIONS_PUSH -> GITHUB_TOKEN"
else
  fail "push auth chain is not 'steps.<mint>.outputs.token || inputs.push-pat || github.token'"
fi

# 5. Absolute tool paths, hooks disabled on every git call.
if has '^[[:space:]]*GIT=/usr/bin/git[[:space:]]*$' && has '^[[:space:]]*BASE64=/usr/bin/base64[[:space:]]*$'; then
  ok "git and base64 run from absolute paths"
else
  fail "git and base64 must run from absolute paths (GIT=/usr/bin/git, BASE64=/usr/bin/base64)"
fi
# `git` in command position — line start, after a separator, `$(`, `if` or
# `!` — so prose inside an echo string does not count.
# shellcheck disable=SC2016  # the literal text "$GIT", not an expansion
GIT_CALL='"$GIT"'
if grep -qE '(^[[:space:]]*|[;|&][[:space:]]*|\$\([[:space:]]*|(if|!)[[:space:]]+)git[[:space:]]' "$BODY"; then
  fail "a bare 'git' call resolves through PATH — call \"\$GIT\" instead"
elif grep -F "$GIT_CALL" "$BODY" | grep -vq 'core\.hooksPath=/dev/null'; then
  fail "a \"\$GIT\" call does not disable hooks (-c core.hooksPath=/dev/null)"
else
  ok "every git call is \"\$GIT\" with hooks disabled"
fi

# 6. No expression interpolated into a run: script.
interpolated="$(awk '
  /^[[:space:]]*run:[[:space:]]*[|>]/ {
    in_run = 1; match($0, /^[[:space:]]*/); run_indent = RLENGTH; next
  }
  in_run && /^[[:space:]]*$/ { next }
  in_run {
    match($0, /^[[:space:]]*/)
    if (RLENGTH <= run_indent) { in_run = 0 } else if (index($0, "${{")) { print; exit }
  }
' "$BODY")"
if [[ -n "$interpolated" ]]; then
  fail "a run: script interpolates an expression — pass it through env: instead: ${interpolated#"${interpolated%%[![:space:]]*}"}"
else
  ok "run: scripts take inputs through env:, never by interpolation"
fi

# 7. Strict bash.
if has 'set -euo pipefail'; then
  ok "strict bash (set -euo pipefail) present"
else
  fail "no 'set -euo pipefail' in the run: script"
fi

# 8. A deleted branch is not an unreachable origin.
if has 'ls-remote[[:space:]]+--exit-code' && has '-eq[[:space:]]+2' && has '::error::could not reach origin'; then
  ok "a deleted head branch is skipped, an unreachable origin fails"
else
  fail "ls-remote failures are not split into 'branch gone' (exit 2) and 'origin unreachable' (error) — a network blip would report success"
fi

# 9. Rebase before push, abort on conflict.
if has '(^|[^[:alnum:]_-])rebase([[:space:]]+--[a-z-]+)*[[:space:]]+"?refs/remotes/origin/' && has 'rebase[[:space:]]+--abort'; then
  ok "rebases onto the remote branch before pushing, aborting a conflict"
else
  fail "no rebase (with --abort on conflict) before push — a concurrent bot push would be clobbered or rejected"
fi

# 10. extraheader push, never forced.
if has 'http\.https://github\.com/\.extraheader=' && has 'push[[:space:]]+origin'; then
  ok "pushes with a per-command extraheader"
else
  fail "no 'push origin' through an http extraheader"
fi
if has 'push([[:space:]]+[^[:space:]]+)*[[:space:]]+(--force|-f|--force-with-lease)([[:space:]=]|$)|[[:space:]]"?\+HEAD:'; then
  fail "the push is forced — it would overwrite commits another job pushed"
else
  ok "the push is never forced"
fi

# 11 and 12. Callers delegate, and pass the secrets.
shopt -s nullglob
workflows=("$WORKFLOWS_DIR"/*.yml "$WORKFLOWS_DIR"/*.yaml)
shopt -u nullglob
for workflow in ${workflows[@]+"${workflows[@]}"}; do
  name="$(basename "$workflow")"
  wf_body="$TMP_DIR/$name"
  strip_comments "$workflow" >"$wf_body"
  if grep -qE 'create-github-app-token|extraheader|(^|[^[:alnum:]_-])push[[:space:]]+origin' "$wf_body"; then
    fail "$name carries its own push logic — commit and push through ./.github/actions/bot-push"
  fi
  # A step begins at `- name:` or `- uses:`; report each bot-push step that
  # is missing one of the secret inputs.
  while IFS= read -r missing; do
    [[ -n "$missing" ]] || continue
    fail "$name: $missing"
  done < <(awk '
    function report() {
      if (!bot) return
      if (!app_id) print "step \"" step "\" does not pass app-client-id: ${{ secrets.ACTIONS_PUSH_APP_CLIENT_ID }}"
      if (!app_key) print "step \"" step "\" does not pass app-private-key: ${{ secrets.ACTIONS_PUSH_APP_PRIVATE_KEY }}"
      if (!pat) print "step \"" step "\" does not pass push-pat: ${{ secrets.ACTIONS_PUSH }}"
    }
    /^[[:space:]]*-[[:space:]]+(name|uses):/ {
      report()
      bot = 0; app_id = 0; app_key = 0; pat = 0; step = $0
      sub(/^[[:space:]]*-[[:space:]]+(name|uses):[[:space:]]*/, "", step)
    }
    /uses:[[:space:]]*"?\.\/\.github\/actions\/bot-push"?[[:space:]]*$/ { bot = 1 }
    /^[[:space:]]*app-client-id:[[:space:]]*\$\{\{[[:space:]]*secrets\.ACTIONS_PUSH_APP_CLIENT_ID[[:space:]]*\}\}/ { app_id = 1 }
    /^[[:space:]]*app-private-key:[[:space:]]*\$\{\{[[:space:]]*secrets\.ACTIONS_PUSH_APP_PRIVATE_KEY[[:space:]]*\}\}/ { app_key = 1 }
    /^[[:space:]]*push-pat:[[:space:]]*\$\{\{[[:space:]]*secrets\.ACTIONS_PUSH[[:space:]]*\}\}/ { pat = 1 }
    END { report() }
  ' "$wf_body")
done
if [[ "$EXIT_CODE" -eq 0 ]]; then
  ok "every workflow delegates its push to the action and passes the push secrets"
fi

exit "$EXIT_CODE"
