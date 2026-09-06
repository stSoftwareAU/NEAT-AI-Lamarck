#!/usr/bin/env bash
# WHAT: check-dependabot-config.sh is a real advisory-channel gate (Issue #215).
#
# Runs the real gate against throwaway Dependabot fixtures and asserts exit
# codes and reported reasons: a missing config, a `version: 1` config, a config
# that registers every ecosystem except cargo, and a cargo entry with no usable
# schedule must each fail loudly, while a well-formed cargo entry passes in
# every YAML spelling the file is legitimately written in.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECK="$SCRIPT_DIR/check-dependabot-config.sh"
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
fixture() {
  local name="$1"
  cat > "$WORK/$name.yml"
  printf '%s' "$WORK/$name.yml"
}

# The shape this gate exists to require.
fixture minimal > /dev/null <<'YML'
version: 2
updates:
  - package-ecosystem: "cargo"
    directory: "/"
    schedule:
      interval: "weekly"
YML

# Unquoted scalars are the same config to Dependabot, so they must read alike.
fixture unquoted > /dev/null <<'YML'
version: 2
updates:
  - package-ecosystem: cargo
    directory: /
    schedule:
      interval: weekly
YML

# Cargo alongside other ecosystems, and not first in the list.
fixture multi_ecosystem > /dev/null <<'YML'
version: 2
updates:
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "monthly"
  # cargo is what carries the RustSec advisory feed
  - package-ecosystem: "cargo"
    directory: "/"
    schedule:
      interval: "daily"
    open-pull-requests-limit: 5
YML

# `directories` (plural) is the documented multi-directory spelling.
fixture directories_plural > /dev/null <<'YML'
version: 2
updates:
  - package-ecosystem: "cargo"
    directories:
      - "/"
      - "/lamarck"
    schedule:
      interval: "weekly"
YML

# Every failing shape.
fixture wrong_version > /dev/null <<'YML'
version: 1
updates:
  - package-ecosystem: "cargo"
    directory: "/"
    schedule:
      interval: "weekly"
YML

fixture no_version > /dev/null <<'YML'
updates:
  - package-ecosystem: "cargo"
    directory: "/"
    schedule:
      interval: "weekly"
YML

fixture no_cargo > /dev/null <<'YML'
version: 2
updates:
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
YML

fixture no_updates > /dev/null <<'YML'
version: 2
YML

fixture no_schedule > /dev/null <<'YML'
version: 2
updates:
  - package-ecosystem: "cargo"
    directory: "/"
YML

fixture bad_interval > /dev/null <<'YML'
version: 2
updates:
  - package-ecosystem: "cargo"
    directory: "/"
    schedule:
      interval: "fortnightly"
YML

fixture no_directory > /dev/null <<'YML'
version: 2
updates:
  - package-ecosystem: "cargo"
    schedule:
      interval: "weekly"
YML

# A commented-out cargo entry is not a registered ecosystem.
fixture commented_out > /dev/null <<'YML'
version: 2
updates:
  # - package-ecosystem: "cargo"
  #   directory: "/"
  #   schedule:
  #     interval: "weekly"
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
YML

# --- assertions -----------------------------------------------------------
check_fixture() {
  local name="$1"
  shift
  "$CHECK" --config "$WORK/$name.yml" "$@"
}

assert_exit "minimal cargo entry → pass" 0 check_fixture minimal
assert_exit "unquoted scalars → pass" 0 check_fixture unquoted
assert_exit "cargo among other ecosystems → pass" 0 check_fixture multi_ecosystem
assert_exit "directories (plural) → pass" 0 check_fixture directories_plural
assert_exit "version: 1 → fail" 1 check_fixture wrong_version
assert_exit "no version key → fail" 1 check_fixture no_version
assert_exit "no cargo ecosystem → fail" 1 check_fixture no_cargo
assert_exit "no updates list → fail" 1 check_fixture no_updates
assert_exit "cargo entry with no schedule → fail" 1 check_fixture no_schedule
assert_exit "unsupported schedule interval → fail" 1 check_fixture bad_interval
assert_exit "cargo entry with no directory → fail" 1 check_fixture no_directory
assert_exit "commented-out cargo entry → fail" 1 check_fixture commented_out

# An absent config is the finding this gate exists for, not a usage error.
assert_exit "absent config → fail" 1 "$CHECK" --config "$WORK/absent.yml"
assert_output_contains "absent config names the expected path" "$WORK/absent.yml" \
  "$CHECK" --config "$WORK/absent.yml"
assert_output_contains "absent config explains the missing advisory feed" "advisory" \
  "$CHECK" --config "$WORK/absent.yml"

# Failures must name the remedy, not just the fault.
assert_output_contains "missing cargo entry names the ecosystem" "package-ecosystem" \
  check_fixture no_cargo
assert_output_contains "bad interval names the accepted values" "weekly" \
  check_fixture bad_interval
assert_output_contains "wrong version names version 2" "version: 2" \
  check_fixture wrong_version

# The repository's own configuration must satisfy the gate (Issue #215).
assert_exit "repository dependabot config → pass" 0 "$CHECK"
assert_output_contains "verbose run names the cargo ecosystem" "cargo" "$CHECK" --verbose

# Usage errors and fail-loud behaviour.
assert_exit "unknown option → usage error" 2 "$CHECK" --nonsense
assert_exit "--config without value → usage error" 2 "$CHECK" --config
assert_exit "--help → pass" 0 "$CHECK" --help

if [[ "$FAILS" -ne 0 ]]; then
  echo "FAIL: $FAILS assertion(s) failed" >&2
  exit 1
fi

echo "check-dependabot-config.sh: all assertions passed"
