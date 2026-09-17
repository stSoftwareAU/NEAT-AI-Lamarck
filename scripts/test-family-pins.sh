#!/usr/bin/env bash
# WHAT: the canonical scripts/family-pins.sh keeps the contract Lamarck relies
# on (Issue #235).
#
# `scripts/family-pins.sh` is a byte-identical copy of NEAT-AI-core's
# `scripts/family-pins.sh` (NEAT-AI-core#681) and is never edited here. The
# family-sync job refetches it on every PR and runs it over the real
# `lamarck/Cargo.toml` moments later, so these tests pin what that copy must
# keep doing:
#
#   * it recognises a family git-tag pin and leaves every other dependency —
#     including a commented-out or non-family declaration — byte-for-byte alone;
#   * a pin it cannot rewrite fails loud, naming the file and the line, rather
#     than reading as "already on the latest release";
#   * a usage error exits 2 and changes nothing.
#
# Every case is driven through `--manifest` over a generated fixture, so no
# manifest carrying a real family pin is scanned: nothing here resolves a
# remote, runs cargo, or touches the network. The move itself — resolve the
# newest release, rewrite the pin, let `Cargo.lock` follow — needs both, so the
# family-sync job that performs it is where it is covered, not faked here.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FAMILY_PINS="${SCRIPT_DIR}/family-pins.sh"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT

PASSED=0
FAILED=0

if [[ ! -x "${FAMILY_PINS}" ]]; then
  echo "FAIL: family-pins.sh not found or not executable: ${FAMILY_PINS}" >&2
  exit 2
fi

assert_eq() {
  local desc="$1" expected="$2" actual="$3"
  if [[ "${expected}" == "${actual}" ]]; then
    echo "  PASS: ${desc}"
    PASSED=$((PASSED + 1))
  else
    echo "  FAIL: ${desc}"
    echo "    expected: '${expected}'"
    echo "    actual:   '${actual}'"
    FAILED=$((FAILED + 1))
  fi
}

# Write $2 as a manifest named $1 and echo its path.
new_manifest() {
  local path="${WORK_DIR}/$1"
  printf '%s' "$2" >"${path}"
  printf '%s' "${path}"
}

# Run family-pins.sh over manifest $1, capturing output under the label $2.
# Echoes the exit status; never aborts the suite.
run_pins() {
  local manifest="$1" name="$2" rc=0
  (
    cd "${WORK_DIR}"
    "${FAMILY_PINS}" --manifest "${manifest}"
  ) >"${WORK_DIR}/${name}.out" 2>"${WORK_DIR}/${name}.err" || rc=$?
  printf '%s' "${rc}"
}

echo "=== --help prints the usage banner ==="
HELP_RC=0
"${FAMILY_PINS}" --help >"${WORK_DIR}/help.out" 2>&1 || HELP_RC=$?
assert_eq "--help exits 0" "0" "${HELP_RC}"
assert_eq "--help describes what the script does" "0" \
  "$(grep -q 'move this checkout' "${WORK_DIR}/help.out"; echo $?)"
assert_eq "--help documents the --manifest option" "0" \
  "$(grep -q -- '--manifest' "${WORK_DIR}/help.out"; echo $?)"

echo ""
echo "=== an unknown argument is a usage error ==="
UNKNOWN_RC=0
"${FAMILY_PINS}" --not-an-option >"${WORK_DIR}/unknown.out" 2>&1 || UNKNOWN_RC=$?
assert_eq "an unknown argument exits 2" "2" "${UNKNOWN_RC}"

echo ""
echo "=== a manifest that does not exist is a usage error ==="
MISSING_RC=0
"${FAMILY_PINS}" --manifest "${WORK_DIR}/no-such-manifest.toml" \
  >"${WORK_DIR}/missing.out" 2>&1 || MISSING_RC=$?
assert_eq "a missing manifest exits 2" "2" "${MISSING_RC}"
assert_eq "a missing manifest is named" "0" \
  "$(grep -q 'manifest not found' "${WORK_DIR}/missing.out"; echo $?)"

echo ""
echo "=== a manifest with no family pin is left byte-identical ==="
PLAIN="$(new_manifest plain.toml '[dependencies]
clap = { version = "4", features = ["derive"] }
serde = { version = "1" }
')"
BEFORE="$(cksum <"${PLAIN}")"
assert_eq "no family pin exits 0" "0" "$(run_pins "${PLAIN}" plain)"
assert_eq "no family pin leaves the manifest alone" "${BEFORE}" "$(cksum <"${PLAIN}")"

echo ""
echo "=== a commented-out family pin is not a pin ==="
COMMENTED="$(new_manifest commented.toml '[dependencies]
# neat-core = { git = "https://github.com/stSoftwareAU/NEAT-AI-core", tag = "v0.0.1" }
clap = { version = "4" }
')"
BEFORE="$(cksum <"${COMMENTED}")"
assert_eq "a commented-out pin exits 0" "0" "$(run_pins "${COMMENTED}" commented)"
assert_eq "a commented-out pin is never rewritten" "${BEFORE}" "$(cksum <"${COMMENTED}")"

echo ""
echo "=== a git dependency outside the family is left alone ==="
FOREIGN="$(new_manifest foreign.toml '[dependencies]
somedep = { git = "https://github.com/someone-else/somedep", tag = "v0.0.1" }
')"
BEFORE="$(cksum <"${FOREIGN}")"
assert_eq "a non-family git pin exits 0" "0" "$(run_pins "${FOREIGN}" foreign)"
assert_eq "a non-family git pin is left alone" "${BEFORE}" "$(cksum <"${FOREIGN}")"

echo ""
echo "=== a pin it cannot rewrite fails loud rather than reading as current ==="
SPLIT="$(new_manifest split.toml '[dependencies]
neat-core = { git = "https://github.com/stSoftwareAU/NEAT-AI-core",
  tag = "v0.0.1" }
')"
BEFORE="$(cksum <"${SPLIT}")"
SPLIT_RC="$(run_pins "${SPLIT}" split)"
assert_eq "a multi-line pin exits non-zero" "1" "${SPLIT_RC}"
assert_eq "a multi-line pin says it must be on one line" "0" \
  "$(grep -q 'one line' "${WORK_DIR}/split.err"; echo $?)"
assert_eq "a multi-line pin leaves the manifest alone" "${BEFORE}" "$(cksum <"${SPLIT}")"

echo ""
echo "=== summary: ${PASSED} passed, ${FAILED} failed ==="
[[ "${FAILED}" -eq 0 ]]
