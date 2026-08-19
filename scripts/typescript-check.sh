#!/usr/bin/env bash
# typescript-check.sh — basic TypeScript validity gate (Issue #167).
#
# Type-checks every tracked `.ts` source with `deno check` so a syntax or type
# error cannot land on main unnoticed. Basic validity only — this is not a
# style or lint gate.
#
# Usage:
#   scripts/typescript-check.sh                 # scan the repository root
#   scripts/typescript-check.sh --root <dir>    # scan a different directory
#
# Exit codes:
#   0 — every TypeScript source type-checks (or there are none to check)
#   1 — a source failed `deno check`, or deno is not installed
#   2 — invalid invocation
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: typescript-check.sh [--root <dir>]

Runs `deno check` over every .ts source under <dir> (default: repository
root), skipping build and vendor directories. Install Deno with:

  curl -fsSL https://deno.land/install.sh | sh

See the "Build and quality gate" section in README.md.
USAGE
}

ROOT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      ROOT="${2:-}"
      if [[ -z "$ROOT" ]]; then
        echo "typescript-check: --root requires a directory argument" >&2
        usage >&2
        exit 2
      fi
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "typescript-check: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

# Checked before anything else so a missing toolchain fails loudly rather than
# reporting a vacuous pass.
if ! command -v deno >/dev/null 2>&1; then
  echo "typescript-check: deno is not installed — install it with: curl -fsSL https://deno.land/install.sh | sh" >&2
  exit 1
fi

if [[ -z "$ROOT" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

if [[ ! -d "$ROOT" ]]; then
  echo "typescript-check: root directory not found: $ROOT" >&2
  exit 2
fi

sources=()
while IFS= read -r source; do
  sources+=("$source")
done < <(find "$ROOT" \
  \( -path '*/target' -o -path '*/node_modules' -o -path '*/.git' \) -prune -o \
  -name '*.ts' -type f -print | sort)

if [[ "${#sources[@]}" -eq 0 ]]; then
  echo "typescript-check: no TypeScript sources under $ROOT — nothing to check"
  exit 0
fi

echo "🔎 Type-checking ${#sources[@]} TypeScript source(s) under: $ROOT"
if ! deno check "${sources[@]}"; then
  echo "typescript-check: FAILED — fix the TypeScript errors above" >&2
  exit 1
fi

echo "typescript-check: all TypeScript sources are valid"
