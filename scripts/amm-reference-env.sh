#!/usr/bin/env bash
# Build a scratch Python environment for the pinned AMM reference optimiser
# (issue #266, opt-in offline weight-optimiser benchmark; never part of CI).
#
# Downloads the pinned AMM source tarball plus numpy and scipy wheels into
# DIR, verifies every sha256, and unpacks them. Nothing is installed into the
# system or user site; nothing is fetched without a checksum.
#
# Usage:
#   scripts/amm-reference-env.sh DIR
#
# Prints exactly one line on stdout, for use with scripts/amm-reference.py:
#   PYTHONPATH=<DIR/site>:<DIR/src/amm-optimizer-<commit>/src>
# Re-running against the same DIR is safe (downloads are re-verified).
set -euo pipefail

# Pins resolved on 2026-11-14.
AMM_COMMIT="74d39b4f7988a1f1aa8b40a2a0c3ee547f251107" # version 0.7.0, Apache-2.0
AMM_URL="https://codeload.github.com/BjBodner/amm-optimizer/tar.gz/${AMM_COMMIT}"
AMM_SHA256="9d161338ff40be69c656e436073df6af38311acd1f0d0967dff9abc8f036fe0f"

# numpy 2.5.4 and scipy 1.18.1, cp313 manylinux wheels.
NUMPY_AARCH64_URL="https://files.pythonhosted.org/packages/db/b6/135bb0953b61dc21c6cafa14b424ae666944e4899cf140e00c2b322a1a45/numpy-2.5.4-cp313-cp313-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl"
NUMPY_AARCH64_SHA256="1c2e71b04c6cad90026e544501bbe0ab9290fa8a4d845e7e8c0d124fb429c988"
SCIPY_AARCH64_URL="https://files.pythonhosted.org/packages/e6/5f/4634e9d35c68496e4e34cb6946eafab044458e6cedab42b40b6588e475b6/scipy-1.18.1-cp313-cp313-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl"
SCIPY_AARCH64_SHA256="d416b16cccfd70fbf62400e84d0bb2f4e6af519a45557f1692c749b37f14b315"
NUMPY_X86_64_URL="https://files.pythonhosted.org/packages/da/24/3bd070f3269dc609d8f26b2643f62ef91bb415841c0b294805aaf7fe06da/numpy-2.5.4-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
NUMPY_X86_64_SHA256="6ffa07666f8da0eef81d149934a626d0d95fbd6838432a33e66245423a9062c0"
SCIPY_X86_64_URL="https://files.pythonhosted.org/packages/41/48/6450ed9243315322bbc19ac57b9b70d66a20bf1d38d124c96bc4bf6af9ea/scipy-1.18.1-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
SCIPY_X86_64_SHA256="fdaf5ea890a6183d0565f51a61799d67081bd5b1cf03c5f4b3fd3732108625c9"

die() {
  echo "amm-reference-env: $*" >&2
  exit 1
}

[[ $# -eq 1 && -n "$1" ]] || die "usage: $0 DIR"
DIR="$1"

os="$(uname -s)"
arch="$(uname -m)"
[[ "$os" == "Linux" ]] || die "unsupported platform: $os/$arch (Linux aarch64 or x86_64 only)"
case "$arch" in
  aarch64)
    NUMPY_URL="$NUMPY_AARCH64_URL"; NUMPY_SHA256="$NUMPY_AARCH64_SHA256"
    SCIPY_URL="$SCIPY_AARCH64_URL"; SCIPY_SHA256="$SCIPY_AARCH64_SHA256"
    ;;
  x86_64)
    NUMPY_URL="$NUMPY_X86_64_URL"; NUMPY_SHA256="$NUMPY_X86_64_SHA256"
    SCIPY_URL="$SCIPY_X86_64_URL"; SCIPY_SHA256="$SCIPY_X86_64_SHA256"
    ;;
  *) die "unsupported platform: $os/$arch (Linux aarch64 or x86_64 only)" ;;
esac

command -v python3 >/dev/null 2>&1 || die "python3 not found (3.13 required)"
pyver="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
[[ "$pyver" == "3.13" ]] || die "python3 is $pyver; 3.13 is required (the pinned wheels are cp313)"

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d' ' -f1
  else
    shasum -a 256 "$1" | cut -d' ' -f1
  fi
}

# fetch URL DEST SHA256 — download, then refuse on checksum mismatch.
fetch() {
  local url="$1" dest="$2" want="$3" got
  curl -fsSL -o "$dest" "$url" || die "download failed: $url"
  got="$(sha256_of "$dest")"
  [[ "$got" == "$want" ]] || die "sha256 mismatch for $(basename "$dest"): expected $want, got $got"
}

mkdir -p "$DIR/dl" "$DIR/site" "$DIR/src"

fetch "$AMM_URL" "$DIR/dl/amm-optimizer-${AMM_COMMIT}.tar.gz" "$AMM_SHA256"
fetch "$NUMPY_URL" "$DIR/dl/$(basename "$NUMPY_URL")" "$NUMPY_SHA256"
fetch "$SCIPY_URL" "$DIR/dl/$(basename "$SCIPY_URL")" "$SCIPY_SHA256"

tar -xzf "$DIR/dl/amm-optimizer-${AMM_COMMIT}.tar.gz" -C "$DIR/src"
python3 -I -m zipfile -e "$DIR/dl/$(basename "$NUMPY_URL")" "$DIR/site"
python3 -I -m zipfile -e "$DIR/dl/$(basename "$SCIPY_URL")" "$DIR/site"

amm_src="$DIR/src/amm-optimizer-${AMM_COMMIT}/src"
[[ -d "$amm_src" ]] || die "AMM package directory missing after unpack: $amm_src"

echo "PYTHONPATH=$DIR/site:$amm_src"
