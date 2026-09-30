#!/usr/bin/env bash
# WHAT: check-bot-push-action.sh keeps the shared bot-push logic hardened and
# in one place (Issue #252).
#
# Each case mutates the shipped action, or a copy of the shipped workflows, to
# break one rule and asserts the validator's exit code. Runs the real validator
# against generated fixtures; asserts exit codes only.
# The sed/awk programs match the action's literal `$GIT` / `${{ … }}` text.
# shellcheck disable=SC2016
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECK="$SCRIPT_DIR/check-bot-push-action.sh"
ACTION="$SCRIPT_DIR/../.github/actions/bot-push/action.yml"
WORKFLOWS="$SCRIPT_DIR/../.github/workflows"
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

# An action fixture made by one sed expression, checked against the shipped
# workflows.
action_case() {
  local label="$1" expected="$2" expression="$3"
  local fixture="$TMP_DIR/action-$FIXTURE_ID.yml"
  FIXTURE_ID=$((FIXTURE_ID + 1))
  sed -E "$expression" "$ACTION" >"$fixture"
  if cmp -s "$fixture" "$ACTION"; then
    echo "FAIL $label (the mutation did not change the action — fixture is stale)" >&2
    FAILS=$((FAILS + 1))
    return
  fi
  assert_exit "$label" "$expected" "$CHECK" "$fixture" "$WORKFLOWS"
}
FIXTURE_ID=0

# A copy of the shipped workflows with one file rewritten by an awk program.
workflows_case() {
  local label="$1" expected="$2" file="$3" program="$4"
  local dir="$TMP_DIR/workflows-$FIXTURE_ID"
  FIXTURE_ID=$((FIXTURE_ID + 1))
  mkdir -p "$dir"
  cp "$WORKFLOWS"/*.yml "$dir"/
  awk "$program" "$WORKFLOWS/$file" >"$dir/$file"
  if cmp -s "$dir/$file" "$WORKFLOWS/$file"; then
    echo "FAIL $label (the mutation did not change $file — fixture is stale)" >&2
    FAILS=$((FAILS + 1))
    return
  fi
  assert_exit "$label" "$expected" "$CHECK" "$ACTION" "$dir"
}

# Baseline and invocation errors.
assert_exit "shipped action and workflows pass" 0 "$CHECK" "$ACTION" "$WORKFLOWS"
assert_exit "missing action file → exit 2" 2 "$CHECK" "$TMP_DIR/absent.yml" "$WORKFLOWS"
assert_exit "missing workflows dir → exit 2" 2 "$CHECK" "$ACTION" "$TMP_DIR/absent"

# Rules 1–4: composite, SHA pins, scoped token, auth chain.
action_case "not composite → fail" 1 's|using: composite|using: node20|'
action_case "floating action tag → fail" 1 's|create-github-app-token@[0-9a-f]{40}|create-github-app-token@v3|'
action_case "token not contents-scoped → fail" 1 '/permission-contents: write/d'
action_case "token not repo-scoped → fail" 1 '/repositories:/d'
action_case "no ACTIONS_PUSH in the chain → fail" 1 's/\|\| inputs\.push-pat //'
action_case "GITHUB_TOKEN only → fail" 1 's|GH_PAT: .*|GH_PAT: ${{ github.token }}|'

# Rule 5: absolute paths, hooks disabled.
action_case "git resolved through PATH → fail" 1 's|^([[:space:]]*)GIT=/usr/bin/git$|\1GIT=git|'
action_case "a bare git call → fail" 1 's|"\$GIT" -c core.hooksPath=/dev/null commit|git commit|'
action_case "a git call with hooks enabled → fail" 1 's|"\$GIT" -c core.hooksPath=/dev/null commit|"$GIT" commit|'

# Rule 6: an interpolated input is a shell-injection sink.
action_case "input interpolated into run: → fail" 1 's|push origin "HEAD:\$BRANCH"|push origin "HEAD:${{ inputs.branch }}"|'

# Rule 7: strict bash.
action_case "no set -euo pipefail → fail" 1 '/set -euo pipefail/d'

# Rule 8: a network blip must not read as "branch deleted".
action_case "ls-remote failures collapsed → fail" 1 's|-eq 2|-ne 0|'

# Rule 9: rebase before push, and clean up a conflict.
action_case "no rebase before push → fail" 1 's|rebase --autostash "refs/remotes/origin/\$\{BRANCH\}"|log -1|'
action_case "conflicted rebase not aborted → fail" 1 's|rebase --abort|status|'

# Rule 10: a forced push overwrites a concurrent bot push.
action_case "force push → fail" 1 's|push origin "HEAD:|push --force origin "HEAD:|'
action_case "+refspec force push → fail" 1 's|push origin "HEAD:|push origin "+HEAD:|'

# Rule 11: an inline copy of the push logic in a workflow is the drift this
# gate exists to stop.
workflows_case "workflow grows an inline push → fail" 1 version-increment.yml '
  { print }
  /^[[:space:]]*set -euo pipefail$/ && !done { match($0, /^[[:space:]]*/); printf "%s/usr/bin/git push origin HEAD\n", substr($0, 1, RLENGTH); done = 1 }
'
workflows_case "workflow mints its own push token → fail" 1 auto-format.yml '
  /^      - name: Commit and push/ {
    print "      - name: Mint"
    print "        uses: actions/create-github-app-token@bcd2ba49218906704ab6c1aa796996da409d3eb1"
  }
  { print }
'
# Prose about the old pattern does not count as a copy of it.
workflows_case "push logic only in a comment → pass" 0 auto-format.yml '
  NR == 1 { print "# formerly: git push origin with an extraheader" }
  { print }
'

# Rule 12: a caller that drops a secret silently degrades to GITHUB_TOKEN.
workflows_case "caller drops push-pat → fail" 1 family-sync.yml '!/^[[:space:]]*push-pat:/'
workflows_case "caller drops the App client id → fail" 1 auto-format.yml '!/^[[:space:]]*app-client-id:/'
workflows_case "caller drops the App private key → fail" 1 version-increment.yml '!/^[[:space:]]*app-private-key:/'

if [[ "$FAILS" -ne 0 ]]; then
  echo "test-check-bot-push-action: $FAILS failure(s)" >&2
  exit 1
fi
echo "test-check-bot-push-action: all assertions passed"
