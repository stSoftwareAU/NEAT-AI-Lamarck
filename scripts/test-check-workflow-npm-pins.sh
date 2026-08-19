#!/usr/bin/env bash
# WHAT: check-workflow-npm-pins.sh is a real CI-install pin gate (Issue #169).
#
# Runs the real gate against throwaway workflow fixtures and asserts exit codes
# and reported locations: a floating `npm install -g <pkg>` in a `run:` block
# must fail loudly, an exact `<pkg>@x.y.z` pin must pass, and a deliberate
# float must only pass when it carries an in-source suppression comment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECK="$SCRIPT_DIR/check-workflow-npm-pins.sh"
FAILS=0

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

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
mkdir -p "$WORK"/{floating,pinned,range,tag,suppressed,expression,lockfile,paths,empty,scoped,multi}

cat > "$WORK/floating/lint.yml" <<'YML'
jobs:
  lint:
    steps:
      - name: Install markdownlint-cli2
        run: npm install -g markdownlint-cli2
YML

cat > "$WORK/pinned/lint.yml" <<'YML'
jobs:
  lint:
    steps:
      - name: Install markdownlint-cli2
        run: npm install -g markdownlint-cli2@0.23.2
YML

# A caret range still resolves to whatever the registry serves — not a pin.
cat > "$WORK/range/lint.yml" <<'YML'
jobs:
  lint:
    steps:
      - run: npm install -g markdownlint-cli2@^0.23.0
YML

cat > "$WORK/tag/lint.yml" <<'YML'
jobs:
  lint:
    steps:
      - run: npx markdownlint-cli2@latest
YML

cat > "$WORK/suppressed/lint.yml" <<'YML'
jobs:
  lint:
    steps:
      # best-practice-ignore: BP-CI-INSTALL-PIN-npm-corepack — tracks the runner image
      - run: npm install -g corepack
YML

# A version supplied by a workflow expression is resolved at run time, so it is
# not an exact pin either.
cat > "$WORK/expression/lint.yml" <<'YML'
jobs:
  lint:
    steps:
      - run: npm install -g markdownlint-cli2@${{ env.MDL_VERSION }}
YML

# `npm ci` / bare `npm install` install from the committed lockfile.
cat > "$WORK/lockfile/build.yml" <<'YML'
jobs:
  build:
    steps:
      - run: npm ci
      - run: npm install --no-audit
YML

# Flag values and paths are not package specs.
cat > "$WORK/paths/build.yml" <<'YML'
jobs:
  build:
    steps:
      - run: npm install -g --prefix /tmp/npm-global markdownlint-cli2@0.23.2
YML

cat > "$WORK/scoped/build.yml" <<'YML'
jobs:
  build:
    steps:
      - run: npm install -g @redocly/cli@2.6.1
YML

cat > "$WORK/multi/build.yml" <<'YML'
jobs:
  build:
    steps:
      - run: npm install -g markdownlint-cli2@0.23.2 && npx prettier
YML

# --- assertions -----------------------------------------------------------
assert_exit "floating npm install -g → fail" 1 "$CHECK" --dir "$WORK/floating"
assert_exit "exact pin → pass" 0 "$CHECK" --dir "$WORK/pinned"
assert_exit "caret range → fail" 1 "$CHECK" --dir "$WORK/range"
assert_exit "dist-tag via npx → fail" 1 "$CHECK" --dir "$WORK/tag"
assert_exit "suppression comment → pass" 0 "$CHECK" --dir "$WORK/suppressed"
assert_exit "workflow expression version → fail" 1 "$CHECK" --dir "$WORK/expression"
assert_exit "npm ci / lockfile install → pass" 0 "$CHECK" --dir "$WORK/lockfile"
assert_exit "flag values and paths ignored → pass" 0 "$CHECK" --dir "$WORK/paths"
assert_exit "scoped package pinned → pass" 0 "$CHECK" --dir "$WORK/scoped"
assert_exit "unpinned npx after && → fail" 1 "$CHECK" --dir "$WORK/multi"
assert_exit "no workflows → pass" 0 "$CHECK" --dir "$WORK/empty"

# The gate must name the file, the line and the package so the fix is obvious.
assert_output_contains "failure names file:line" "lint.yml:5" "$CHECK" --dir "$WORK/floating"
assert_output_contains "failure names the package" "markdownlint-cli2" \
  "$CHECK" --dir "$WORK/floating"

# The repository's own workflows must satisfy the gate (Issue #169).
assert_exit "repository workflows → pass" 0 "$CHECK"
assert_output_matches "markdown-lint install is pinned" \
  "^OK[[:space:]]+markdown-lint\.yml:[0-9]+: markdownlint-cli2@[0-9]+\.[0-9]+\.[0-9]+" \
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

echo "check-workflow-npm-pins.sh: all assertions passed"
