#!/usr/bin/env python3
"""Run the pinned AMM reference optimiser over a weight-router bench plan.

Issue #266. An opt-in offline research tool, never part of CI.

Build the environment with scripts/amm-reference-env.py, then run (without
-I, which would ignore PYTHONPATH):

    env PYTHONPATH=... python3 scripts/amm-reference.py \
        --bench target/release/examples/weight_router_bench \
        --manifest OUT/manifest.json --out-dir OUT/amm --jobs 2

Protocol. For each validation run the bench binary is started as

    BIN --manifest M --serve-reference --family F --dimension D --seed S
        --budget-multiple K

It prints one JSON header line (dimension, budget, seed, start,
boxHalfWidth). Each stdin line holding a JSON array of D floats is answered
with {"loss": v} or {"error": "msg"}. A stdin line "done" (or EOF) makes it
print {"row": {...}} (a RunRow) and exit, non-zero when the row has an error.

Box. AMM needs bounds; we use start +/- boxHalfWidth from the bench, so the
reference searches the same neighbourhood the in-tree optimisers start from.
AMM runs with its default configuration (preset c46, frozen for the paper).
"""
import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time

import numpy as np

AMM_COMMIT = "74d39b4f7988a1f1aa8b40a2a0c3ee547f251107"
AMM_VERSION = "0.7.0"
PRESET = "c46"
PRESET_HASH = "2153948ebed51150"


def plan(manifest):
    """Validation runs in manifest order: (family, dimension, seed, multiple)."""
    families = ["corpus"] if manifest.get("corpus") else manifest["families"]
    return [
        (f, d, s, k)
        for f in families
        for d in manifest["dimensions"]
        for s in manifest["validationSeeds"]
        for k in manifest["budgetMultiples"]
    ]


def failure(run, why):
    """A complete RunRow for a run that never produced one (camelCase, as serde reads it)."""
    family, dimension, seed, multiple = run
    return {"family": family, "dimension": dimension,
            "budgetMultiple": multiple, "budget": multiple * dimension,
            "seed": seed, "phase": "validation", "arm": "amm-reference",
            "routedTo": None, "startScreeningLoss": None,
            "finalScreeningLoss": None, "normalisedScreeningLoss": None,
            "evaluationsUsed": 0, "probeEvaluations": 0,
            "preparationEvaluations": 0, "fullLossStart": None,
            "fullLossFinal": None, "heldOutLossStart": None,
            "heldOutLossFinal": None, "normalisedHeldOutLoss": None,
            "confirmationEvaluations": 0,
            # Mirrors EXTRAPOLATION_ABOVE in lamarck/src/weight_bench/suite.rs.
            "extrapolation": dimension > 40,
            "error": f"amm run {family}/d{dimension}/seed{seed}/x{multiple}: {why}"}


def stderr_cause(text):
    """The bench prints `weight_router_bench: <error>` on failure; prefer that line."""
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    named = [x for x in lines if x.startswith("weight_router_bench:")]
    return named[-1] if named else (lines[-1] if lines else "no stderr")


def run_one(args):
    """Worker: drive one served run through amm_optimizer.minimize."""
    bench, manifest_path, run = args
    import amm_optimizer

    family, dimension, seed, multiple = run
    cmd = [bench, "--manifest", manifest_path, "--serve-reference",
           "--family", family, "--dimension", str(dimension),
           "--seed", str(seed), "--budget-multiple", str(multiple)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, bufsize=1)
    try:
        header_line = proc.stdout.readline()
        if not header_line:
            _, err = proc.communicate()
            return failure(run, f"no header from server: {stderr_cause(err)}")
        header = json.loads(header_line)
        start = np.array(header["start"], dtype=float)
        h = header["boxHalfWidth"]

        def f(x):
            proc.stdin.write(json.dumps([float(v) for v in x]) + "\n")
            proc.stdin.flush()
            reply = json.loads(proc.stdout.readline() or '{"error":"server closed"}')
            if "error" in reply:
                raise RuntimeError(reply["error"])
            return float(reply["loss"])

        raised = None
        try:
            amm_optimizer.minimize(f, x0=start, budget=header["budget"],
                                   bounds=(start - h, start + h),
                                   seed=np.random.SeedSequence(header["seed"]))
        except Exception as exc:  # recorded on the row, never swallowed
            raised = f"{type(exc).__name__}: {exc}"

        try:
            proc.stdin.write("done\n")
            proc.stdin.flush()
        except BrokenPipeError:
            pass
        row = None
        for line in proc.stdout:
            if line.startswith('{"row"'):
                row = json.loads(line)["row"]
        err = proc.stderr.read()
        code = proc.wait()
        if row is None:
            return failure(run, f"server exit {code} without a row: {stderr_cause(err)}")
        if raised:  # the AMM cause leads; keep the server's error after it
            row["error"] = f"amm: {raised}" + (
                f"; server: {row['error']}" if row.get("error") else "")
        return row
    except Exception as exc:
        proc.kill()
        return failure(run, f"{type(exc).__name__}: {exc}")
    finally:
        proc.wait()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bench", required=True, help="built weight_router_bench binary")
    ap.add_argument("--manifest", required=True, help="bench output manifest.json")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()

    import amm_optimizer
    import scipy
    from amm_optimizer import presets
    got = presets.get(PRESET).config_hash()
    if amm_optimizer.__version__ != AMM_VERSION or got != PRESET_HASH:
        print(f"amm-reference: need amm_optimizer {AMM_VERSION} with {PRESET} hash "
              f"{PRESET_HASH}; found {amm_optimizer.__version__} / {got}", file=sys.stderr)
        return 2

    with open(a.manifest) as fh:
        runs = plan(json.load(fh))
    t0 = time.time()
    work = [(a.bench, a.manifest, r) for r in runs]
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.jobs) as pool:
        rows = list(pool.map(run_one, work))
    failed = sum(1 for r in rows if r.get("error"))

    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "amm-runs.jsonl"), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    meta = {"ammVersion": amm_optimizer.__version__, "ammCommit": AMM_COMMIT,
            "preset": PRESET, "configHash": got, "numpy": np.__version__,
            "scipy": scipy.__version__, "python": sys.version.split()[0],
            "bench": a.bench, "manifest": a.manifest, "planned": len(runs),
            "completed": len(rows) - failed, "failed": failed,
            "wallSeconds": round(time.time() - t0, 3)}
    with open(os.path.join(a.out_dir, "amm-meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
        fh.write("\n")
    print(f"amm-reference: planned {len(runs)}, completed {len(rows) - failed}, "
          f"failed {failed}, {meta['wallSeconds']}s")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
