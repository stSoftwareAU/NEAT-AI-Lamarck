#!/usr/bin/env bash
# WHAT: bump-lamarck-version.sh bumps the crate patch against whatever base
# branch the PR targets — Develop or milestone/<slug> (Issue #190).
#
# Runs the real script inside throwaway git repositories and asserts on the
# outcome (exit code plus the version actually written to lamarck/Cargo.toml
# and Cargo.lock), never on the script's source text.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FAILS=0

TMP_ROOT="$(mktemp -d)"
trap 'rm -rf "$TMP_ROOT"' EXIT

# Disable repository hooks: the fixture repos are throwaway and must not pick
# up a template or worker-installed pre-commit gate.
git_q() { git -C "$1" -c core.hooksPath=/dev/null "${@:2}"; }

# make_repo <dir> <base-branch> [version]
# Builds a minimal Lamarck-shaped repo: the two scripts under test, a crate
# manifest, a lockfile stanza and a source file, all committed on <base-branch>.
make_repo() {
  local dir="$1" base="$2" version="${3:-0.1.23}"
  mkdir -p "$dir/scripts" "$dir/lamarck/src"
  cp "$SCRIPT_DIR/bump-lamarck-version.sh" \
    "$SCRIPT_DIR/check-lamarck-version-order.sh" "$dir/scripts/"
  cat >"$dir/lamarck/Cargo.toml" <<EOF
[package]
name = "neat_ai_lamarck"
version = "$version"
edition = "2024"
EOF
  cat >"$dir/Cargo.lock" <<EOF
[[package]]
name = "neat_ai_lamarck"
version = "$version"
EOF
  echo 'fn main() {}' >"$dir/lamarck/src/main.rs"
  git_q "$dir" init --quiet --initial-branch "$base"
  git_q "$dir" config user.email "test@example.com"
  git_q "$dir" config user.name "Test"
  git_q "$dir" add -A
  git_q "$dir" commit --quiet -m "base"
}

# set_version <dir> <version> — rewrite the manifest version in place.
set_version() {
  local dir="$1" version="$2"
  sed -i.bak "s/^version = \".*\"/version = \"$version\"/" "$dir/lamarck/Cargo.toml"
  rm -f "$dir/lamarck/Cargo.toml.bak"
}

manifest_version() {
  sed -n 's/^version *= *"\([^"]*\)".*/\1/p' "$1/lamarck/Cargo.toml" | head -n 1
}

lock_version() {
  sed -n 's/^version *= *"\([^"]*\)".*/\1/p' "$1/Cargo.lock" | head -n 1
}

# Errexit is toggled by the caller (assert_bump), never here — flipping it
# back on inside the callee would abort the suite on an expected non-zero exit.
run_bump() {
  local dir="$1"
  shift
  (cd "$dir" && ./scripts/bump-lamarck-version.sh "$@") >/dev/null 2>&1
}

assert_eq() {
  local label="$1" expected="$2" got="$3"
  if [[ "$expected" == "$got" ]]; then
    echo "OK   $label ($got)"
  else
    echo "FAIL $label (expected '$expected', got '$got')" >&2
    FAILS=$((FAILS + 1))
  fi
}

assert_bump() {
  local label="$1" expected="$2" dir="$3"
  shift 3
  set +e
  run_bump "$dir" "$@"
  local got=$?
  set -e
  assert_eq "$label" "$expected" "$got"
}

# --- src changed vs a milestone base → bump (the Issue #190 case) ----------
MILESTONE="$TMP_ROOT/milestone"
make_repo "$MILESTONE" "milestone/demo" "0.1.23"
git_q "$MILESTONE" checkout --quiet -b issue-190
echo 'fn helper() {}' >>"$MILESTONE/lamarck/src/main.rs"
git_q "$MILESTONE" commit --quiet -am "src change"

assert_bump "milestone base, src changed → bump" 0 "$MILESTONE" --base-ref milestone/demo
assert_eq "milestone base bumps the manifest patch" "0.1.24" "$(manifest_version "$MILESTONE")"
assert_eq "milestone base syncs Cargo.lock" "0.1.24" "$(lock_version "$MILESTONE")"

# Re-running is a no-op: the branch now sits ahead of its base.
assert_bump "re-run on bumped branch → skip" 1 "$MILESTONE" --base-ref milestone/demo
assert_eq "re-run leaves the version alone" "0.1.24" "$(manifest_version "$MILESTONE")"

# --- src changed vs Develop → bump ---------------------------------------
DEVELOP="$TMP_ROOT/develop"
make_repo "$DEVELOP" "Develop" "1.4.9"
git_q "$DEVELOP" checkout --quiet -b issue-1
echo 'fn helper() {}' >>"$DEVELOP/lamarck/src/main.rs"
git_q "$DEVELOP" commit --quiet -am "src change"

assert_bump "--check reports a pending bump" 0 "$DEVELOP" --base-ref Develop --check
assert_eq "--check does not write the manifest" "1.4.9" "$(manifest_version "$DEVELOP")"

assert_bump "Develop base, src changed → bump" 0 "$DEVELOP" --base-ref Develop
assert_eq "patch rolls over past 9" "1.4.10" "$(manifest_version "$DEVELOP")"

# --- docs-only change → no bump ------------------------------------------
DOCS="$TMP_ROOT/docs"
make_repo "$DOCS" "Develop" "0.2.0"
git_q "$DOCS" checkout --quiet -b issue-2
echo "notes" >"$DOCS/NOTES.md"
git_q "$DOCS" add NOTES.md
git_q "$DOCS" commit --quiet -m "docs only"

assert_bump "no src change → skip" 1 "$DOCS" --base-ref Develop
assert_eq "docs-only leaves the version alone" "0.2.0" "$(manifest_version "$DOCS")"

# --- version behind base → loud failure, never a silent bump -------------
DOWNGRADE="$TMP_ROOT/downgrade"
make_repo "$DOWNGRADE" "Develop" "0.3.5"
git_q "$DOWNGRADE" checkout --quiet -b issue-3
set_version "$DOWNGRADE" "0.3.4"
echo 'fn helper() {}' >>"$DOWNGRADE/lamarck/src/main.rs"
git_q "$DOWNGRADE" commit --quiet -am "downgrade + src change"

assert_bump "version behind base → error" 2 "$DOWNGRADE" --base-ref Develop
assert_eq "downgrade is not papered over" "0.3.4" "$(manifest_version "$DOWNGRADE")"

# --- unknown base ref → error, not a silent skip -------------------------
assert_bump "missing base ref → error" 2 "$DOWNGRADE" --base-ref no/such/branch

if [[ "$FAILS" -ne 0 ]]; then
  echo "FAIL: $FAILS assertion(s) failed" >&2
  exit 1
fi

echo "OK   all bump-lamarck-version WHAT assertions passed"
exit 0
