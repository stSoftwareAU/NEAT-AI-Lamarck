#!/usr/bin/env bash
# Gate: working-tree neat_ai_lamarck version must not sit behind a base ref.
#
# A merge conflict that silently takes Develop's older `lamarck/Cargo.toml`
# version must fail CI rather than look like an intentional bump (Issue #152).
#
# Usage: check-lamarck-version-no-downgrade.sh [--base-ref REF]
#
# Exit codes:
#   0  current >= base (or base version unavailable — nothing to compare)
#   1  current < base (downgrade)
#   2  usage / parse error
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MANIFEST="$REPO_ROOT/lamarck/Cargo.toml"
BASE_REF="origin/Develop"

usage() {
  cat <<'EOF'
Usage: check-lamarck-version-no-downgrade.sh [--base-ref REF]

  --base-ref REF   Git ref whose lamarck/Cargo.toml version is the floor
                   (default: origin/Develop).

Exits 0 when the working-tree version is equal or ahead of the base, 1 on a
downgrade, 2 on a usage / parse error. If the base ref (or its manifest) is
missing, exits 0 with a notice so shallow / offline checkouts are not blocked.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-ref)
      BASE_REF="${2:?--base-ref requires a value}"
      shift 2
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "FAIL: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

cd "$REPO_ROOT"

if [[ ! -f "$MANIFEST" ]]; then
  echo "FAIL: missing $MANIFEST" >&2
  exit 2
fi

read_version() {
  local file="$1"
  sed -n 's/^version *= *"\([^"]*\)".*/\1/p' "$file" | head -n 1
}

CURRENT_VERSION="$(read_version "$MANIFEST")"
if [[ -z "$CURRENT_VERSION" ]]; then
  echo "FAIL: cannot read version from $MANIFEST" >&2
  exit 2
fi

if ! git rev-parse --verify --quiet "$BASE_REF" >/dev/null 2>&1; then
  echo "OK   base ref $BASE_REF not available — skip version floor check"
  exit 0
fi

BASE_VERSION="$(git show "${BASE_REF}:lamarck/Cargo.toml" 2>/dev/null | sed -n 's/^version *= *"\([^"]*\)".*/\1/p' | head -n 1 || true)"
if [[ -z "$BASE_VERSION" ]]; then
  echo "OK   no lamarck/Cargo.toml version on $BASE_REF — skip version floor check"
  exit 0
fi

set +e
"$SCRIPT_DIR/check-lamarck-version-order.sh" "$BASE_VERSION" "$CURRENT_VERSION"
status=$?
set -e
if [[ "$status" -eq 1 ]]; then
  echo "FAIL: neat_ai_lamarck version ${CURRENT_VERSION} is behind ${BASE_REF} ${BASE_VERSION}; package versions must never go backwards" >&2
  exit 1
elif [[ "$status" -ne 0 ]]; then
  exit "$status"
fi

exit 0
