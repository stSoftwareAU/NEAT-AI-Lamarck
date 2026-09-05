#!/usr/bin/env bash
# WHAT: check-workflow-container-pins.sh is a real container pin gate (Issue #212).
#
# Runs the real gate against throwaway workflow fixtures and asserts exit codes
# and reported locations: a bare digest (no tag) and a bare tag (no digest) must
# both fail loudly, `name:tag@sha256:<digest>` must pass, and a deliberate
# exception must only pass when it carries an in-source suppression comment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECK="$SCRIPT_DIR/check-workflow-container-pins.sh"
FAILS=0

DIGEST="sha256:a9ea2d5621c29d815d90c2a3b2f9571da8972ef4ff855c9e4902681730240e35"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

assert_exit() {
  local label="$1" expected="$2"
  shift 2
  set +e
  "$@" > /dev/null 2>&1
  local got=$?
  set -e
  if [[ "$got" -eq "$expected" ]]; then
    echo "OK   $label (exit $got)"
  else
    echo "FAIL $label (expected exit $expected, got $got)" >&2
    FAILS=$((FAILS + 1))
  fi
}

assert_output_contains() {
  local label="$1" needle="$2"
  shift 2
  local output
  set +e
  output="$("$@" 2>&1)"
  set -e
  if [[ "$output" == *"$needle"* ]]; then
    echo "OK   $label"
  else
    echo "FAIL $label (output did not contain '$needle')" >&2
    echo "$output" >&2
    FAILS=$((FAILS + 1))
  fi
}

assert_output_matches() {
  local label="$1" pattern="$2"
  shift 2
  local output
  set +e
  output="$("$@" 2>&1)"
  set -e
  if grep -qE "$pattern" <<< "$output"; then
    echo "OK   $label"
  else
    echo "FAIL $label (output did not match /$pattern/)" >&2
    echo "$output" >&2
    FAILS=$((FAILS + 1))
  fi
}

# --- fixtures -------------------------------------------------------------
mkdir -p "$WORK"/{digest_only,tag_only,both,bad_digest,expression,suppressed,service,registry,quoted,empty,comment}

# The exact regression this gate exists for: immutable but un-bumpable.
cat > "$WORK/digest_only/scan.yml" <<YML
jobs:
  scan:
    container:
      image: semgrep/semgrep@${DIGEST}
YML

cat > "$WORK/tag_only/scan.yml" <<'YML'
jobs:
  scan:
    container:
      image: semgrep/semgrep:1.86.0
YML

cat > "$WORK/both/scan.yml" <<YML
jobs:
  scan:
    container:
      image: semgrep/semgrep:1.86.0@${DIGEST}
YML

cat > "$WORK/bad_digest/scan.yml" <<'YML'
jobs:
  scan:
    container:
      image: semgrep/semgrep:1.86.0@sha256:deadbeef
YML

cat > "$WORK/expression/scan.yml" <<'YML'
jobs:
  scan:
    container:
      image: semgrep/semgrep:${{ env.SEMGREP_VERSION }}
YML

cat > "$WORK/suppressed/scan.yml" <<YML
jobs:
  scan:
    container:
      # best-practice-ignore: BP-CONTAINER-PIN-local-build — built by an earlier job
      image: local/build@${DIGEST}
YML

# A \`services:\` image is pinned by the same rule as a job container.
cat > "$WORK/service/scan.yml" <<'YML'
jobs:
  scan:
    services:
      postgres:
        image: postgres:16
YML

# Earlier colons are registry ports, not tags.
cat > "$WORK/registry/scan.yml" <<YML
jobs:
  scan:
    container:
      image: registry.example.com:5000/team/scanner@${DIGEST}
YML

cat > "$WORK/quoted/scan.yml" <<YML
jobs:
  scan:
    container:
      image: "semgrep/semgrep:1.86.0@${DIGEST}"  # trailing note
YML

# A commented-out image is documentation, not a pin.
cat > "$WORK/comment/scan.yml" <<'YML'
jobs:
  scan:
    steps:
      # image: semgrep/semgrep
      - run: echo hello
YML

# --- assertions -----------------------------------------------------------
assert_exit "bare digest, no tag → fail" 1 "$CHECK" --dir "$WORK/digest_only"
assert_exit "bare tag, no digest → fail" 1 "$CHECK" --dir "$WORK/tag_only"
assert_exit "tag and digest → pass" 0 "$CHECK" --dir "$WORK/both"
assert_exit "malformed digest → fail" 1 "$CHECK" --dir "$WORK/bad_digest"
assert_exit "workflow expression tag → fail" 1 "$CHECK" --dir "$WORK/expression"
assert_exit "suppression comment → pass" 0 "$CHECK" --dir "$WORK/suppressed"
assert_exit "services image unpinned → fail" 1 "$CHECK" --dir "$WORK/service"
assert_exit "registry port is not a tag → fail" 1 "$CHECK" --dir "$WORK/registry"
assert_exit "quoted value with trailing comment → pass" 0 "$CHECK" --dir "$WORK/quoted"
assert_exit "commented-out image ignored → pass" 0 "$CHECK" --dir "$WORK/comment"
assert_exit "no workflows → pass" 0 "$CHECK" --dir "$WORK/empty"

# The gate must name the file, the line and the remedy so the fix is obvious.
assert_output_contains "failure names file:line" "scan.yml:4" \
  "$CHECK" --dir "$WORK/digest_only"
assert_output_contains "failure names the tagged remedy" "semgrep/semgrep:<tag>@${DIGEST}" \
  "$CHECK" --dir "$WORK/digest_only"
assert_output_contains "missing digest names the remedy" "sha256:<digest>" \
  "$CHECK" --dir "$WORK/tag_only"

# The repository's own workflows must satisfy the gate (Issue #212).
assert_exit "repository workflows → pass" 0 "$CHECK"
assert_output_matches "semgrep container carries tag and digest" \
  "^OK[[:space:]]+semgrep\.yml:[0-9]+: semgrep/semgrep:[0-9]+\.[0-9]+\.[0-9]+@sha256:[0-9a-f]{64}$" \
  "$CHECK" --verbose

# Usage errors and fail-loud behaviour.
assert_exit "missing directory → usage error" 2 "$CHECK" --dir "$WORK/nope"
assert_exit "unknown option → usage error" 2 "$CHECK" --nonsense
assert_exit "--dir without value → usage error" 2 "$CHECK" --dir
assert_exit "--help → pass" 0 "$CHECK" --help

if [[ "$FAILS" -ne 0 ]]; then
  echo "FAIL: $FAILS assertion(s) failed" >&2
  exit 1
fi

echo "check-workflow-container-pins.sh: all assertions passed"
