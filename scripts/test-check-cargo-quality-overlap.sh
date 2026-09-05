#!/usr/bin/env bash
# WHAT: check-cargo-quality-overlap.sh is a real trigger-overlap gate (Issue #213).
#
# Runs the real gate against throwaway workflow fixtures and asserts exit codes
# and reported reasons: a `cargo-quality.yml` whose branch filter also matches
# the branches `ci.yml` already gates must fail loudly, an exclusion that covers
# every gated branch must pass, and an exclusion so wide that no feature branch
# is left covered must fail too — the workflow exists for those branches.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECK="$SCRIPT_DIR/check-cargo-quality-overlap.sh"
FAILS=0

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

# --- fixtures -------------------------------------------------------------
# Every fixture pairs a ci.yml (the authoritative gate) with a cargo-quality.yml.
new_case() {
  mkdir -p "$WORK/$1"
  printf '%s' "$1"
}

standard_ci() {
  cat > "$WORK/$1/ci.yml" <<'YML'
name: CI
on:
  pull_request:
    types: [opened, synchronize, reopened]
    branches:
      - Develop
      - "milestone/**"
  workflow_dispatch:
jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - run: cargo fmt --all -- --check
YML
}

# The exact regression this gate exists for: `**` re-matches Develop.
new_case overlap > /dev/null
standard_ci overlap
cat > "$WORK/overlap/cargo-quality.yml" <<'YML'
name: Cargo Quality
on:
  pull_request:
    branches: ["**"]
  workflow_dispatch:
YML

new_case fixed > /dev/null
standard_ci fixed
cat > "$WORK/fixed/cargo-quality.yml" <<'YML'
name: Cargo Quality
on:
  pull_request:
    branches-ignore: [Develop, "milestone/**"]
  workflow_dispatch:
YML

# The same exclusion written as a block sequence must be read identically.
new_case fixed_block > /dev/null
standard_ci fixed_block
cat > "$WORK/fixed_block/cargo-quality.yml" <<'YML'
name: Cargo Quality
on:
  pull_request:
    branches-ignore:
      - Develop
      # milestone sub-issue PRs are gated by ci.yml
      - "milestone/**"
YML

# Excluding only half the gated set still duplicates the milestone path.
new_case partial > /dev/null
standard_ci partial
cat > "$WORK/partial/cargo-quality.yml" <<'YML'
name: Cargo Quality
on:
  pull_request:
    branches-ignore: [Develop]
YML

# `*` does not cross `/`, so it misses milestone/** but still re-matches Develop.
new_case single_star > /dev/null
standard_ci single_star
cat > "$WORK/single_star/cargo-quality.yml" <<'YML'
name: Cargo Quality
on:
  pull_request:
    branches: ["*"]
YML

# No branch filter at all matches every branch, gated ones included.
new_case no_filter > /dev/null
standard_ci no_filter
cat > "$WORK/no_filter/cargo-quality.yml" <<'YML'
name: Cargo Quality
on:
  pull_request:
  workflow_dispatch:
YML

# Removing the trigger "fixes" the overlap by deleting the coverage.
new_case no_pr > /dev/null
standard_ci no_pr
cat > "$WORK/no_pr/cargo-quality.yml" <<'YML'
name: Cargo Quality
on:
  workflow_dispatch:
YML

# Ignoring everything leaves no feature branch covered — same loss of coverage.
new_case ignores_all > /dev/null
standard_ci ignores_all
cat > "$WORK/ignores_all/cargo-quality.yml" <<'YML'
name: Cargo Quality
on:
  pull_request:
    branches-ignore: ["**"]
YML

# A ci.yml with no branch filter gates every branch — overlap is unavoidable
# and the gate must say so rather than pass silently.
new_case ci_no_filter > /dev/null
cat > "$WORK/ci_no_filter/ci.yml" <<'YML'
name: CI
on:
  pull_request:
    types: [opened, synchronize, reopened]
YML
cat > "$WORK/ci_no_filter/cargo-quality.yml" <<'YML'
name: Cargo Quality
on:
  pull_request:
    branches-ignore: [Develop, "milestone/**"]
YML

# A push-only branch filter in ci.yml must not be mistaken for its PR filter.
new_case push_filter > /dev/null
cat > "$WORK/push_filter/ci.yml" <<'YML'
name: CI
on:
  push:
    branches: [Develop]
  pull_request:
    branches:
      - Develop
YML
cat > "$WORK/push_filter/cargo-quality.yml" <<'YML'
name: Cargo Quality
on:
  push:
    branches: ["**"]
  pull_request:
    branches-ignore: [Develop]
YML

# --- assertions -----------------------------------------------------------
check_case() {
  local case_name="$1"
  shift
  "$CHECK" --ci "$WORK/$case_name/ci.yml" --quality "$WORK/$case_name/cargo-quality.yml" "$@"
}

assert_exit "branches: ['**'] re-matches gated branches → fail" 1 check_case overlap
assert_exit "branches-ignore covers gated branches → pass" 0 check_case fixed
assert_exit "block-sequence branches-ignore → pass" 0 check_case fixed_block
assert_exit "only half the gated set excluded → fail" 1 check_case partial
assert_exit "branches: ['*'] still re-matches Develop → fail" 1 check_case single_star
assert_exit "no branch filter → fail" 1 check_case no_filter
assert_exit "no pull_request trigger → fail" 1 check_case no_pr
assert_exit "excludes every branch → fail" 1 check_case ignores_all
assert_exit "ci.yml gates every branch → fail" 1 check_case ci_no_filter
assert_exit "push filters are not PR filters → pass" 0 check_case push_filter

# The gate must name the duplicated branch and the remedy so the fix is obvious.
assert_output_contains "overlap names the gated branch" "Develop" check_case overlap
assert_output_contains "overlap names branches-ignore as the remedy" "branches-ignore" \
  check_case overlap
assert_output_contains "partial exclusion names the milestone glob" "milestone/**" \
  check_case partial
assert_output_contains "lost coverage is reported as lost coverage" "no feature branch" \
  check_case ignores_all

# The repository's own workflows must satisfy the gate (Issue #213).
assert_exit "repository workflows → pass" 0 "$CHECK"
assert_output_contains "verbose run names the excluded branch" "Develop" "$CHECK" --verbose

# Usage errors and fail-loud behaviour.
assert_exit "missing ci workflow → usage error" 2 "$CHECK" --ci "$WORK/nope.yml"
assert_exit "missing quality workflow → usage error" 2 "$CHECK" --quality "$WORK/nope.yml"
assert_exit "unknown option → usage error" 2 "$CHECK" --nonsense
assert_exit "--ci without value → usage error" 2 "$CHECK" --ci
assert_exit "--quality without value → usage error" 2 "$CHECK" --quality
assert_exit "--help → pass" 0 "$CHECK" --help

if [[ "$FAILS" -ne 0 ]]; then
  echo "FAIL: $FAILS assertion(s) failed" >&2
  exit 1
fi

echo "check-cargo-quality-overlap.sh: all assertions passed"
