#!/usr/bin/env bash
# Summarise Lamarck strategy economics from experiments.jsonl.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <experiments.jsonl>" >&2
  exit 2
fi

JOURNAL="$1"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${ROOT}/target/release/neat_ai_lamarck"
if [[ ! -x "${BIN}" ]]; then
  BIN="${ROOT}/target/debug/neat_ai_lamarck"
fi
if [[ ! -x "${BIN}" ]]; then
  echo "Building neat_ai_lamarck..."
  (cd "${ROOT}" && cargo build -q -p neat_ai_lamarck)
  BIN="${ROOT}/target/debug/neat_ai_lamarck"
fi

exec "${BIN}" report "${JOURNAL}"
