#!/usr/bin/env bash
set -euo pipefail

export RUSTFLAGS="-D warnings"

cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
