#!/usr/bin/env python3
"""Compare released c46 engines on the public synthetic manifest's first seed.

Run once with each pinned source tree on PYTHONPATH, sharing the pinned wheels.
This is an offline compatibility measurement, not a representative trial or CI.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import tempfile

import amm_optimizer
from amm_optimizer import presets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--compare")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        "amm_reference", Path(__file__).with_name("amm-reference.py")
    )
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    config_hash = presets.get(driver.PRESET).config_hash()
    if amm_optimizer.__version__ not in ("0.3.0", driver.AMM_VERSION) or config_hash != driver.PRESET_HASH:
        raise ValueError("method check requires the pinned 0.3.0 or 0.7.0 c46 engine")
    original = amm_optimizer.minimize

    def compatible(*positional, **keywords):
        # v0.3.0 takes a 32-bit seed and config object; v0.7.0 also accepts these.
        keywords["seed"] = int(keywords["seed"].generate_state(1)[0])
        keywords["config"] = presets.get(keywords["config"])
        return original(*positional, **keywords)

    amm_optimizer.minimize = compatible
    manifest = json.loads(Path(args.manifest).read_text())
    if not manifest["validationSeeds"]:
        raise ValueError("method check requires a validation seed")
    manifest["validationSeeds"] = manifest["validationSeeds"][:1]
    with tempfile.TemporaryDirectory(prefix="amm-method-") as scratch:
        path = Path(scratch) / "manifest.json"
        path.write_text(json.dumps(manifest))
        rows = [driver.run_one((args.bench, str(path), cell)) for cell in driver.plan(manifest)]
    if not rows:
        raise ValueError("method check planned no runs")
    result = {"version": amm_optimizer.__version__, "configHash": config_hash,
              "manifest": manifest, "rows": rows}
    Path(args.out).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    failed = sum(row["error"] is not None for row in rows)
    if failed:
        raise RuntimeError(f"{failed}/{len(rows)} method-check runs failed; see {args.out}")
    if args.compare:
        previous = json.loads(Path(args.compare).read_text())
        if previous["manifest"] != manifest or previous["rows"] != rows:
            raise RuntimeError("released engines produced different RunRows or used different manifests")
    print(f"AMM {amm_optimizer.__version__}: {len(rows)} completed rows"
          + (", identical to comparison" if args.compare else ""))


if __name__ == "__main__":
    main()
