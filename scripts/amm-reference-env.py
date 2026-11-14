#!/usr/bin/env python3
"""Build a scratch Python environment for the pinned AMM reference optimiser
(issue #266, opt-in offline weight-optimiser benchmark; never part of CI).

Downloads the pinned AMM source tarball plus numpy and scipy wheels into
DIR, verifies every sha256, and unpacks them. Nothing is installed into the
system or user site; nothing is fetched without a checksum.

Usage:
  scripts/amm-reference-env.py DIR

Prints exactly one line on stdout, for use with scripts/amm-reference.py:
  PYTHONPATH=<DIR/site>:<DIR/src/amm-optimizer-<commit>/src>
Re-running against the same DIR is safe (downloads are re-verified).
"""

import hashlib
import http.client
import os
import platform
import sys
import tarfile
import urllib.request
import zipfile

# Pins resolved on 2026-11-14.
AMM_COMMIT = "74d39b4f7988a1f1aa8b40a2a0c3ee547f251107"  # version 0.7.0, Apache-2.0
AMM_URL_TEMPLATE = "https://codeload.github.com/BjBodner/amm-optimizer/tar.gz/{commit}"
AMM_URL = AMM_URL_TEMPLATE.format(commit=AMM_COMMIT)
AMM_SHA256 = "9d161338ff40be69c656e436073df6af38311acd1f0d0967dff9abc8f036fe0f"

# numpy 2.5.4 and scipy 1.18.1, cp313 manylinux wheels.
NUMPY_AARCH64_URL = "https://files.pythonhosted.org/packages/db/b6/135bb0953b61dc21c6cafa14b424ae666944e4899cf140e00c2b322a1a45/numpy-2.5.4-cp313-cp313-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl"
NUMPY_AARCH64_SHA256 = "1c2e71b04c6cad90026e544501bbe0ab9290fa8a4d845e7e8c0d124fb429c988"
SCIPY_AARCH64_URL = "https://files.pythonhosted.org/packages/e6/5f/4634e9d35c68496e4e34cb6946eafab044458e6cedab42b40b6588e475b6/scipy-1.18.1-cp313-cp313-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl"
SCIPY_AARCH64_SHA256 = "d416b16cccfd70fbf62400e84d0bb2f4e6af519a45557f1692c749b37f14b315"
NUMPY_X86_64_URL = "https://files.pythonhosted.org/packages/da/24/3bd070f3269dc609d8f26b2643f62ef91bb415841c0b294805aaf7fe06da/numpy-2.5.4-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
NUMPY_X86_64_SHA256 = "6ffa07666f8da0eef81d149934a626d0d95fbd6838432a33e66245423a9062c0"
SCIPY_X86_64_URL = "https://files.pythonhosted.org/packages/41/48/6450ed9243315322bbc19ac57b9b70d66a20bf1d38d124c96bc4bf6af9ea/scipy-1.18.1-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
SCIPY_X86_64_SHA256 = "fdaf5ea890a6183d0565f51a61799d67081bd5b1cf03c5f4b3fd3732108625c9"

# arch -> ((numpy url, sha256), (scipy url, sha256))
WHEELS = {
    "aarch64": ((NUMPY_AARCH64_URL, NUMPY_AARCH64_SHA256), (SCIPY_AARCH64_URL, SCIPY_AARCH64_SHA256)),
    "x86_64": ((NUMPY_X86_64_URL, NUMPY_X86_64_SHA256), (SCIPY_X86_64_URL, SCIPY_X86_64_SHA256)),
}


def die(message):
    print(f"amm-reference-env: {message}", file=sys.stderr)
    sys.exit(1)


def fetch(url, dest, want):
    """Download url to dest, then refuse on checksum mismatch."""
    if not url.startswith("https://"):
        die(f"refusing non-https download: {url}")
    digest = hashlib.sha256()
    with open(dest, "wb") as out:
        try:
            # URLs are the pinned https constants above (checked to be https), never user input.
            # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected
            with urllib.request.urlopen(url) as response:
                for chunk in iter(lambda: response.read(1 << 16), b""):
                    digest.update(chunk)
                    out.write(chunk)
        except (OSError, ValueError, http.client.HTTPException):
            die(f"download failed: {url}")
    got = digest.hexdigest()
    if got != want:
        die(f"sha256 mismatch for {os.path.basename(dest)}: expected {want}, got {got}")


def main(argv):
    if len(argv) != 2 or not argv[1]:
        die(f"usage: {argv[0]} DIR")
    target = argv[1]

    system, machine = platform.system(), platform.machine()
    if system != "Linux" or machine not in WHEELS:
        die(f"unsupported platform: {system}/{machine} (Linux aarch64 or x86_64 only)")
    if sys.version_info[:2] != (3, 13):
        die(f"python3 is {sys.version_info.major}.{sys.version_info.minor}; "
            "3.13 is required (the pinned wheels are cp313)")

    dl, site, src = (os.path.join(target, name) for name in ("dl", "site", "src"))
    for path in (dl, site, src):
        os.makedirs(path, exist_ok=True)

    amm_tar = os.path.join(dl, f"amm-optimizer-{AMM_COMMIT}.tar.gz")
    fetch(AMM_URL, amm_tar, AMM_SHA256)
    wheels = []
    for url, want in WHEELS[machine]:
        dest = os.path.join(dl, os.path.basename(url))
        fetch(url, dest, want)
        wheels.append(dest)

    with tarfile.open(amm_tar, "r:gz") as tar:
        # Member-by-member with the "data" filter, which rejects absolute paths, ".."
        # traversal, links escaping the destination, and special files.
        for member in tar:
            tar.extract(member, src, filter="data")
    for wheel in wheels:
        with zipfile.ZipFile(wheel) as zf:
            zf.extractall(site)

    amm_src = os.path.join(src, f"amm-optimizer-{AMM_COMMIT}", "src")
    if not os.path.isdir(amm_src):
        die(f"AMM package directory missing after unpack: {amm_src}")
    print(f"PYTHONPATH={site}:{amm_src}")


if __name__ == "__main__":
    main(sys.argv)
