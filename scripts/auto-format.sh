#!/usr/bin/env bash
# Auto-format / PR housekeeping helper (Issue #33).
#
# Responsibilities:
#   * Emit the deterministic commit message the workflow uses when pushing
#     rustfmt and/or Cargo.lock sync fixes back onto a PR branch.
#   * Detect whether the working tree has pending changes after `cargo fmt`
#     and `cargo update -p neat-core` ran, so the workflow can commit only
#     when needed (idempotent guard).
#
# Running cargo is the workflow's job — this script does not invoke any
# build tooling.
set -euo pipefail

COMMIT_MESSAGE="chore(fmt): apply rustfmt and sync neat-core lock

Automated by the auto-format PR job (rustfmt via cargo fmt; Cargo.lock
synced to the checked-out NEAT-AI-core path dependency via
\`cargo update -p neat-core\`) — see issue #33."

usage() {
  cat <<'EOF'
Usage: auto-format.sh <mode> [--repo DIR]

Modes (exactly one):
  --commit-message   Print the deterministic commit message.
  --check-changes    Exit 0 when tracked files have pending changes,
                     exit 1 when the tracked working tree is clean.

Options:
  --repo DIR         Repository directory for git operations (default: cwd).
  -h, --help         Show this message.
EOF
}

MODE=""
REPO_DIR="."

set_mode() {
  if [[ -n "$MODE" ]]; then
    echo "Usage error: only one mode may be supplied" >&2
    usage >&2
    exit 2
  fi
  MODE="$1"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --commit-message) set_mode commit-message; shift ;;
    --check-changes)  set_mode check-changes;  shift ;;
    --repo)           REPO_DIR="$2"; shift 2 ;;
    -h|--help)        usage; exit 0 ;;
    *)
      echo "Usage error: unknown option '$1'" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$MODE" ]]; then
  echo "Usage error: a mode is required" >&2
  usage >&2
  exit 2
fi

case "$MODE" in
  commit-message)
    printf '%s\n' "$COMMIT_MESSAGE"
    ;;

  check-changes)
    # Ignore untracked paths (e.g. the NEAT-AI-core checkout beside the
    # workspace) — cargo cannot modify them as tracked files.
    status_output="$(cd "$REPO_DIR" && git status --porcelain | grep -v '^??' || true)"
    if [[ -z "$status_output" ]]; then
      echo "clean: no formatting or lockfile changes detected"
      exit 1
    else
      echo "dirty: formatting or lockfile changes detected"
      exit 0
    fi
    ;;
esac
