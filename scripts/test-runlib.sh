#!/usr/bin/env bash
# Hermetic tests for the canonical scripts/runlib.sh (Issues #236, #234).
#
# `scripts/runlib.sh` is a byte-identical copy of NEAT-AI-core's canonical
# script and is never edited here, so these tests pin the *contract* this
# repository depends on rather than the script's internals:
#
#   * a second run over an up-to-date install runs NO cargo command at all —
#     not even `cargo metadata` — and prints `[neat_ai_lamarck] already
#     installed v<x>` on stderr with the binary path on stdout;
#   * a build installs `$CARGO_HOME/bin/neat_ai_lamarck`, stamps
#     `.neat_ai_lamarck.version` beside it, and removes the checkout `target/`;
#   * a missing stamp, a missing binary or a stale version all rebuild.
#
# The already-installed case runs against this repository's own manifests, so
# a manifest change that pushes the crate off the canonical fast path — an
# explicit `[[bin]]` table, say — fails here rather than silently costing a
# `cargo metadata` on every fleet invocation.
#
# No real cargo build ever runs: a shim stands in for cargo and rustc.
# Cross-platform: macOS bash 3.2, Ubuntu, AWS Linux.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNLIB="${SCRIPT_DIR}/runlib.sh"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT
REAL_PATH="${PATH}"

PASSED=0
FAILED=0

if [[ ! -x "${RUNLIB}" ]]; then
  echo "FAIL: runlib not found or not executable: ${RUNLIB}" >&2
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

assert_contains() {
  local desc="$1" needle="$2" file="$3"
  if grep -qF -- "${needle}" "${file}"; then
    echo "  PASS: ${desc}"
    PASSED=$((PASSED + 1))
  else
    echo "  FAIL: ${desc}"
    echo "    wanted line containing: '${needle}'"
    echo "    in:"
    sed 's/^/      /' "${file}"
    FAILED=$((FAILED + 1))
  fi
}

# The crate version straight from the manifest — read without cargo, so these
# tests stay hermetic and need no NEAT-AI-core sibling checkout.
crate_version() {
  awk '
    /^[[:space:]]*\[/ { in_pkg = ($0 ~ /^[[:space:]]*\[package\]/) ? 1 : 0; next }
    !in_pkg { next }
    /^[[:space:]]*version[[:space:]]*=/ {
      line = $0
      sub(/^[^=]*=[[:space:]]*/, "", line)
      gsub(/"/, "", line)
      gsub(/[[:space:]]/, "", line)
      print line
      exit
    }
  ' "${REPO_ROOT}/lamarck/Cargo.toml"
}

# A cargo that refuses every invocation: the already-installed path must run
# no cargo command whatsoever, `metadata` included.
install_refusing_cargo() {
  local bin_dir="$1"
  mkdir -p "${bin_dir}"
  cat >"${bin_dir}/cargo" <<'EOF'
#!/usr/bin/env bash
echo "UNEXPECTED cargo: $*" >&2
exit 99
EOF
  chmod +x "${bin_dir}/cargo"
}

# A cargo that answers `metadata` for the synthetic crate and fabricates a
# release binary on `build`, plus the `rustc` the toolchain check demands.
install_building_cargo() {
  local bin_dir="$1" repo="$2" crate="$3" version="$4"
  mkdir -p "${bin_dir}"
  cat >"${bin_dir}/cargo" <<EOF
#!/usr/bin/env bash
set -euo pipefail
CRATE="${crate}"
VERSION="${version}"
REPO="${repo}"
EOF
  cat >>"${bin_dir}/cargo" <<'EOF'
if [[ "${1:-}" == "metadata" ]]; then
  printf '{"packages":[{"name":"%s","version":"%s","manifest_path":"%s/lamarck/Cargo.toml","targets":[{"kind":["bin"],"name":"%s"},{"kind":["lib"],"name":"%s"}]}],"target_directory":"%s/target"}\n' \
    "$CRATE" "$VERSION" "$REPO" "$CRATE" "$CRATE" "$REPO"
  exit 0
fi
if [[ "${1:-}" == "build" ]]; then
  mkdir -p "$REPO/target/release"
  printf 'built binary\n' >"$REPO/target/release/$CRATE"
  chmod +x "$REPO/target/release/$CRATE"
  # Bulk to prove the freed-bytes report is measured, not guessed.
  mkdir -p "$REPO/target/debug"
  dd if=/dev/zero of="$REPO/target/debug/ballast" bs=1024 count=64 2>/dev/null
  exit 0
fi
echo "UNEXPECTED cargo: $*" >&2
exit 99
EOF
  chmod +x "${bin_dir}/cargo"
  cat >"${bin_dir}/rustc" <<'EOF'
#!/usr/bin/env bash
echo "rustc 1.98.0 (797e8a9bc 2026-08-05)"
EOF
  chmod +x "${bin_dir}/rustc"
}

# A synthetic single-member workspace mirroring this repository's shape.
make_fake_repo() {
  local repo="$1" crate="$2" version="$3"
  mkdir -p "${repo}/lamarck/src"
  cat >"${repo}/Cargo.toml" <<'EOF'
[workspace]
members = ["lamarck"]
resolver = "2"
EOF
  cat >"${repo}/lamarck/Cargo.toml" <<EOF
[package]
name = "${crate}"
version = "${version}"
edition = "2024"

[lib]
name = "${crate}"
path = "src/lib.rs"
EOF
  printf 'fn main() {}\n' >"${repo}/lamarck/src/main.rs"
  printf '\n' >"${repo}/lamarck/src/lib.rs"
}

# Run runlib from $1 with CARGO_HOME $2 and the shim dir $3 on PATH.
run_runlib() {
  local repo="$1" cargo_home="$2" shim="$3" errfile="$4"
  (
    cd "${repo}"
    HOME="${cargo_home%/.cargo}" \
      CARGO_HOME="${cargo_home}" \
      PATH="${shim}:${REAL_PATH}" \
      bash "${RUNLIB}" 2>"${errfile}"
  )
}

VERSION="$(crate_version)"
if [[ -z "${VERSION}" ]]; then
  echo "FAIL: could not read the neat_ai_lamarck version from lamarck/Cargo.toml" >&2
  exit 2
fi
echo "neat_ai_lamarck manifest version: ${VERSION}"
echo ""

echo "=== already installed: no cargo command at all, bin path on stdout ==="
CARGO_HOME_A="${WORK_DIR}/a/.cargo"
mkdir -p "${CARGO_HOME_A}/bin"
printf 'fake\n' >"${CARGO_HOME_A}/bin/neat_ai_lamarck"
chmod +x "${CARGO_HOME_A}/bin/neat_ai_lamarck"
printf '%s\n' "${VERSION}" >"${CARGO_HOME_A}/bin/.neat_ai_lamarck.version"
install_refusing_cargo "${WORK_DIR}/refuse"

OUT="$(run_runlib "${REPO_ROOT}" "${CARGO_HOME_A}" "${WORK_DIR}/refuse" "${WORK_DIR}/already.err")" && RC=0 || RC=$?
assert_eq "already-installed exits 0" "0" "${RC}"
assert_eq "already-installed stdout is the CLI path" \
  "${CARGO_HOME_A}/bin/neat_ai_lamarck" "${OUT}"
assert_contains "already-installed names the version on stderr" \
  "[neat_ai_lamarck] already installed v${VERSION}" "${WORK_DIR}/already.err"
assert_eq "already-installed ran no cargo command (not even metadata)" "0" \
  "$(grep -c 'UNEXPECTED cargo' "${WORK_DIR}/already.err" || true)"

echo ""
echo "=== missing stamp rebuilds ==="
rm -f "${CARGO_HOME_A}/bin/.neat_ai_lamarck.version"
run_runlib "${REPO_ROOT}" "${CARGO_HOME_A}" "${WORK_DIR}/refuse" "${WORK_DIR}/nostamp.err" >/dev/null && RC=0 || RC=$?
assert_eq "missing stamp does not report already-installed" "99" "${RC}"

echo ""
echo "=== missing binary beside a matching stamp rebuilds ==="
printf '%s\n' "${VERSION}" >"${CARGO_HOME_A}/bin/.neat_ai_lamarck.version"
rm -f "${CARGO_HOME_A}/bin/neat_ai_lamarck"
run_runlib "${REPO_ROOT}" "${CARGO_HOME_A}" "${WORK_DIR}/refuse" "${WORK_DIR}/nobin.err" >/dev/null && RC=0 || RC=$?
assert_eq "missing binary does not report already-installed" "99" "${RC}"

echo ""
echo "=== stale stamp rebuilds ==="
printf 'fake\n' >"${CARGO_HOME_A}/bin/neat_ai_lamarck"
chmod +x "${CARGO_HOME_A}/bin/neat_ai_lamarck"
printf '0.0.1-stale\n' >"${CARGO_HOME_A}/bin/.neat_ai_lamarck.version"
run_runlib "${REPO_ROOT}" "${CARGO_HOME_A}" "${WORK_DIR}/refuse" "${WORK_DIR}/stale.err" >/dev/null && RC=0 || RC=$?
assert_eq "stale stamp does not report already-installed" "99" "${RC}"

echo ""
echo "=== first install: binary, stamp, and target/ removed ==="
FAKE_REPO="${WORK_DIR}/fake-repo"
CARGO_HOME_B="${WORK_DIR}/b/.cargo"
mkdir -p "${CARGO_HOME_B}"
make_fake_repo "${FAKE_REPO}" "neat_ai_lamarck" "9.9.9"
install_building_cargo "${WORK_DIR}/build" "${FAKE_REPO}" "neat_ai_lamarck" "9.9.9"

OUT="$(run_runlib "${FAKE_REPO}" "${CARGO_HOME_B}" "${WORK_DIR}/build" "${WORK_DIR}/install.err")" && RC=0 || RC=$?
assert_eq "first install exits 0" "0" "${RC}"
assert_eq "first install prints the installed bin path" \
  "${CARGO_HOME_B}/bin/neat_ai_lamarck" "${OUT}"
assert_eq "the CLI is installed and executable" "yes" \
  "$([[ -x "${CARGO_HOME_B}/bin/neat_ai_lamarck" ]] && echo yes || echo no)"
assert_eq "the version stamp is written beside it" "9.9.9" \
  "$(cat "${CARGO_HOME_B}/bin/.neat_ai_lamarck.version" 2>/dev/null || true)"
assert_eq "target/ is removed after a successful install" "gone" \
  "$([[ -d "${FAKE_REPO}/target" ]] && echo present || echo gone)"
assert_contains "the removal names the path and the bytes freed" \
  "[neat_ai_lamarck] removed ${FAKE_REPO}/target" "${WORK_DIR}/install.err"

echo ""
echo "=== second run over that install: no cargo command at all ==="
install_refusing_cargo "${WORK_DIR}/refuse2"
OUT="$(run_runlib "${FAKE_REPO}" "${CARGO_HOME_B}" "${WORK_DIR}/refuse2" "${WORK_DIR}/second.err")" && RC=0 || RC=$?
assert_eq "second run exits 0" "0" "${RC}"
assert_eq "second run prints the installed bin path" \
  "${CARGO_HOME_B}/bin/neat_ai_lamarck" "${OUT}"
assert_contains "second run reports already installed" \
  "[neat_ai_lamarck] already installed v9.9.9" "${WORK_DIR}/second.err"
assert_eq "second run ran no cargo command" "0" \
  "$(grep -c 'UNEXPECTED cargo' "${WORK_DIR}/second.err" || true)"

echo ""
echo "=== summary: ${PASSED} passed, ${FAILED} failed ==="
[[ "${FAILED}" -eq 0 ]]
