# Budget-aware routing of local weight optimisers (issue #266)

**Status: opt-in offline experiment.** Nothing in the optimisation loop calls
this code: it runs only when the `weight_router_bench` example (or a test)
drives it, so ordinary runs are unchanged. No default was changed; these Rust arms are offline research code, not
production integrations. The synthetic suite has been run and is reported below; the
**representative run on the private production creature and corpus has not been
made**. AMM (arXiv 2611.06879) is **not ported and not reimplemented**. It was
run as an offline reference arm through its authors' implementation and lost in
all 27 cells (see [AMM reference results](#amm-reference-results)). The full method has now been read and pinned to
[arXiv:2611.06879v1](https://arxiv.org/pdf/2611.06879v1). The refinement control
is a simplified surrogate, **not the production candidate pipeline**; the
reported margins do not establish superiority over current Lamarck refinement.

**The router and its landscape probes have been removed.** They never beat
CMA-ES alone (see [Findings](#findings)), so they did not earn their
maintenance cost, and the issue ports only winning probes or routers. The
current code runs two in-process arms, refinement and CMA-ES, on paired
validation seeds with no calibration phase. The committed `manifest.json`,
`summary.json` and `reference-summary.json` in
[`docs/evidence/weight-router/`](evidence/weight-router/summary.json) are
output of the current code (crate 0.1.49, `formatVersion` 3) and are reproduced
by the commands in [How to run](#how-to-run). The router, probe and calibration
sections below describe an **earlier router-era run** (crate 0.1.46). Its
numbers are kept as a recorded negative finding; they cannot be reproduced from
the current code, because that code was removed.

The code is [`lamarck/src/weight_bench/`](../lamarck/src/weight_bench/mod.rs),
the driver is
[`lamarck/examples/weight_router_bench.rs`](../lamarck/examples/weight_router_bench.rs),
and the committed evidence is in
[`docs/evidence/weight-router/`](evidence/weight-router/summary.json).

## Question and hypothesis

The benchmark's simplified refinement control spends its evaluation budget on
random axis moves. Production Lamarck also uses statistics, backpropagation,
focus selection and candidate filtering; this control does not reproduce them. Is a covariance-adapting optimiser (CMA-ES) a better use of the same
budget, and can a cheap rule that probes the landscape and routes by budget and
smoothness do at least as well as either fixed choice?

Hypothesis: with enough budget on a smooth landscape CMA-ES beats the current
refinement, and a router that spends a few evaluations probing, then picks the
better arm, beats both. The router is only worth keeping if it beats CMA-ES
alone.

## Arms

| Arm | What it is |
|-----|------------|
| `lamarck-refine` | A simplified random-axis surrogate on a fixed weight set, not a call to `generate_candidate_batch`. Each round picks an axis and a delta drawn like the Random strategy (uniform in `±RANDOM_DELTA_HALF_WIDTH`, `0.05`, from `candidates.rs`, zero excluded), tries the mirrored pair `x ± delta` as `mirror.rs` does, and accepts the better one only if it strictly improves. After an accept it probes the #219 follow-up `STEP_FRACTIONS` (`[0.5, -0.5, 1.0]`, from `followup.rs`) along the winning move and takes the best strict improvement. |
| `cma-es` | Full-covariance (mu/mu_w, lambda)-CMA-ES, written in-repo with no third-party dependency. Strategy parameters are Hansen's tutorial defaults (arXiv:1604.00772); the eigendecomposition is a cyclic Jacobi method. Initial per-coordinate standard deviation `CMA_SIGMA0 = 0.05`. This equals the surrogate's uniform half-width numerically, but CMA-ES moves all coordinates and the search radii are not matched. |
| `router` (removed; router-era run only) | Spent `2 * probe_axes + 1` evaluations probing the landscape, then ran `lamarck-refine` or `cma-es` on the remaining budget. The probe was charged to the **same** budget. The chosen arm re-evaluated the start, one further counted evaluation. |
| AMM (reference only) | Adaptive Multi Mode, run offline through its authors' reference implementation, not reimplemented here and not ported. It needs a known budget and a finite box, spends a counted part of the budget on online probes and routes the rest among several engines. See [AMM reference](#amm-reference). |

In the router-era run the router picked CMA-ES iff the total budget is at least `minBudgetMultiple`
times the dimension **and** the probe's plateau share plus kink share is at most
`maxNonSmoothShare`; otherwise it picked the refinement.

## Predeclared protocol

The manifest is fixed in code (`predeclared_manifest` in
`lamarck/src/weight_bench/experiment.rs`) before the run, and every run echoes
the manifest it used to `manifest.json` (the committed copy is
[`manifest.json`](evidence/weight-router/manifest.json)).

* **Families.** Hidden squash `tanh` (smooth), `relu` (kinks at zero) and
  `hard-tanh` (kinks where it clips). The free weights are the most
  loss-sensitive synapses, which steers away from flat ones. Format 3 measures
  the flat-axis count directly (Finding 4): no chosen free weight was flat in
  any problem, so the predeclared suite does not exercise plateaus. Its families
  vary the activation, not plateau structure. The format 3 flat-axis test is
  one-sided (see Finding 4). The two-sided landscape diagnosis and the
  `saturated-hard-tanh` family that does have flat axes were added after the
  predeclared run, outside this protocol (Finding 5).
* **Problem construction** (`suite.rs`). A random teacher network with 16
  inputs, 16 hidden neurons of one family, fan-in 8 and one identity output
  generates the records (target noise `0.01`). That is 16 x 8 + 16 = 144
  synapses. The start creature is the teacher with every weight nudged by up to
  `0.3` (`START_PERTURBATION`), so it is an already-fit creature that is
  slightly off. There are 256 screening records (the first 256 of 1024 full
  records) and 512 held-out records from a disjoint stream. The free weights are
  the `D` synapses whose loss is most sensitive to a one-sided `0.05` step; that
  preparation costs 145 evaluations per problem.
* **Dimensions.** `D` = 20, 40 and 100. `D` above 40 (`EXTRAPOLATION_ABOVE`) is
  **extrapolation** beyond the calibrated range: those cells are marked and
  need validation on the representative run before being believed.
* **Budgets.** 10D, 30D and 100D evaluations (`ALLOWED_BUDGET_MULTIPLES`).
* **Seeds.** Validation seeds 1001 to 1012 (the manifest is rejected if any
  cell or seed identifier is duplicated). Each validation seed runs every arm on
  the same problem, so differences are paired. In the router-era run, calibration
  seeds 1 to 8 chose the router rule and the manifest refused any overlap with
  validation. Neither remaining arm has a tuned parameter, so the current code
  has no calibration seeds. A manifest that still carries `calibrationSeeds`,
  or any other unknown field, is refused.
* **Landscape probe** (removed; router-era `probes.rs`). Up to `ROUTER_PROBE_AXES = 8` evenly spaced
  axes, each evaluated at `x0 ± 0.05 e_i` (`2 x 8 + 1 = 17` evaluations). An
  axis is a **plateau** when both moves change the loss by at most
  `PLATEAU_REL_TOL = 1e-6` relative to the start loss. Otherwise it is a
  **kink** when the left and right slopes differ by more than
  `KINK_ASYMMETRY = 0.5` of the sum of their magnitudes. The format 3 flat-axis
  count reuses `PLATEAU_REL_TOL` against the base loss (the loss at the start
  weights); see Finding 4.
* **Calibration** (removed; router-era run only). Grid search on calibration seeds only: `minBudgetMultiple` in
  `{10, 30, 100, never}` by `maxNonSmoothShare` in `{0.25, 0.5, 0.75, 1.01}`,
  minimising the mean normalised loss of the arm each row would be routed to.
  Calibration scores each routed row with that arm's full-budget calibration
  result, so it does not charge the router's 17 probe evaluations; validation
  runs the router itself and does charge them.
  Calibrated result: `minBudgetMultiple` 10, `maxNonSmoothShare` 0.25,
  `probeAxes` 8.
* **Metric.** Normalised loss, final over start, on the screening records;
  confirmed on the held-out records and, from format 3, on the full-window
  records (`full`: the full-record loss at the best point over the full-record
  loss at the start). Negative paired differences favour the first arm.
* **Success and failure criteria** (`verdict` in `experiment.rs`). Per cell, the
  paired differences over the 12 validation seeds are bootstrapped (2000
  resamples, seed 266, 95% percentile interval of the mean). The screening,
  held-out and full-window draws share one bootstrap RNG, the full-window draw
  coming after the other two in every comparison, so format 3 moved the interval
  of every cell after the first (the paired differences did not change). The
  first arm is `-better` iff the interval's upper bound on screening loss is
  below zero **and** the held-out mean difference is at most zero **and** the
  full-window mean difference is at most zero (the full-window term is format 3;
  format 2 had only the first two conditions). The full-window veto was added
  after the format 2 results had been seen; it can only turn a `-better` verdict
  into `inconclusive`, never create one, and it changed exactly one verdict
  (hard-tanh, D=40, 100D). It is `-worse` iff the interval's
  lower bound is above zero. Otherwise the cell is `inconclusive`. A run that
  fails is recorded with its error and counted in coverage. In validation it is
  dropped from that pair even if it retained finite metrics (counted under
  `failures`). In the router-era calibration it counted as 1.0, no improvement.
  The router-era, format 2 and format 3 runs had no failures.
* **Compute limits.** Each run's evaluation budget is enforced by
  `BudgetedObjective`, which refuses (fails loud with `BudgetExhausted`) any
  evaluation past the limit.
* **Cost accounting.** Total evaluations include preparation (145 per problem,
  counted once per problem built) and 5 confirmation evaluations per successful run
  (`CONFIRMATION_EVALUATIONS`; failed confirmation attempts count too): the start on the screening records, the start
  and the best point on the full records, and the start and the best point on the
  held-out records. The router-era total also counted the calibration probe
  passes and the router's 17 probe evaluations per run, which came out of the
  router's budget. Each AMM serve process rebuilds its problem, so the AMM
  reference total counts preparation once per AMM run.

## AMM reference

The full method is pinned to [arXiv:2611.06879v1](https://arxiv.org/pdf/2611.06879v1)
(submitted 4 November 2026): Section 3, Algorithms 1–2, Tables 2 and 6,
Appendices A–B and Section 6 were read. It specifies preset `c46`, documents
probe defaults and validity ranges, and requires the harness to stop evaluations
at the budget. Its probes depend on the box width; the removed router-era probes
used different plateau/kink tests and were not AMM. The tested dimension range ends at
40, so D=100 remains extrapolation. These are constraints on reproduction, not
evidence of transfer to neural weight search.

The paper identifies release `v0.3.0`, resolved with `gh api` to commit
`c4f0ab160aae912311328145aa5a426b26a06b26`. Its source tarball SHA-256 is
`c3435857f04aa9a8c9c18519924469ad00bb67ce836da47aedff2ec990493187`.
The existing driver keeps `BjBodner/amm-optimizer` (Apache-2.0) at commit
`74d39b4f7988a1f1aa8b40a2a0c3ee547f251107`, release 0.7.0. Both releases give
`c46` config hash `2153948ebed51150`. The pinned release's
[`FREEZE_2026-10-22.md`](https://github.com/BjBodner/amm-optimizer/blob/74d39b4f7988a1f1aa8b40a2a0c3ee547f251107/docs/FREEZE_2026-10-22.md)
and `DEVIATIONS.md` describe preservation of the frozen configuration; a
matching config hash alone is not a behavioural proof.

The offline compatibility check is `scripts/check-amm-method.py`: it uses the
existing driver and real Rust objective server, one validation seed (1001),
three families, three dimensions and three budgets. It explicitly supplies `c46`
and converts `SeedSequence` to its first 32-bit word so both public APIs receive
the same seed and configuration. This checks completed RunRows, not every
evaluation trace or every AMM route. The source/dependency pins and observed
result are recorded in
[`method-check.json`](evidence/weight-router/method-check.json). It was
regenerated at commit `f0a7b4c` (crate 0.1.49) after the router was removed;
later commits leave the benchmark code (`lamarck/src/weight_bench/`, the
`weight_router_bench` example and `Cargo.lock`) unchanged. The v0.7.0 package
completed 27 rows, identical to the v0.3.0 tree's, with no failures. The check
takes the committed manifest and keeps only its first validation seed, so the
`manifest` it records has `validationSeeds` [1001] and no `calibrationSeeds`. A
second run, at `f0a7b4c`, reproduced the earlier run at `757ac15` (crate 0.1.48)
exactly: the canonical-rows hash (`33203af0…`) and every row file (`runs.jsonl`,
`amm-runs.jsonl`, and the cell values of both summaries). The bench binary hash
(`8ccd83c8…`) is that build's (crate 0.1.49), not the 0.1.48 build's.

To repeat the compatibility check, extract the two pinned source tarballs
(after verifying the SHA-256 values above), share the wheels produced by
`amm-reference-env.py`, and run with one source tree at a time:

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=SITE:PAPER_SRC \
  python3 scripts/check-amm-method.py --bench BIN \
  --manifest docs/evidence/weight-router/manifest.json --out /tmp/amm-paper.json
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=SITE:PINNED_SRC \
  python3 scripts/check-amm-method.py --bench BIN \
  --manifest docs/evidence/weight-router/manifest.json --out /tmp/amm-current.json \
  --compare /tmp/amm-paper.json
```

No dependency is added to the repository. AMM runs in a scratch Python 3.13
environment built by
[`scripts/amm-reference-env.py`](../scripts/amm-reference-env.py) from
sha256-pinned numpy 2.5.4 and scipy 1.18.1 wheels and the pinned AMM tarball. It
is opt-in, offline and never part of CI.

Protocol:

* The Rust bench serves the exact same budgeted objective
  (`--serve-reference`, one JSON line per evaluation). The budget is enforced by
  `BudgetedObjective`, so AMM's probes are counted inside the same budget.
* Only validation seeds 1001 to 1012 are served; any other seed is refused.
* AMM gets `x0` = the start weights, budget = the same multiple x D, box = start
  ± `REFERENCE_BOX_HALF_WIDTH` (`0.5`, predeclared before the run, not tuned) and
  seed = the arm seed via `numpy.random.SeedSequence`.
* `--compare-reference` uses the same paired bootstrap (2000 resamples, seed
  266) and the same predeclared verdict rule as the main experiment, and pairs
  AMM against refine and AMM against CMA-ES. The router-era run also paired AMM
  against the removed router.
* The serve mode honours `--creature` and `--training-data` too (the manifest
  the flags select is used), so the same reference can be run on a corpus
  manifest. That has not been done.

## How to run

```bash
# Predeclared synthetic experiment (release build)
cargo run --release --example weight_router_bench -- OUT_DIR

# One tiny cell, to check the plumbing
cargo run --release --example weight_router_bench -- OUT_DIR --smoke

# A different, validated manifest
cargo run --release --example weight_router_bench -- OUT_DIR --manifest FILE

# Representative comparison on a real creature and corpus
cargo run --release --example weight_router_bench -- OUT_DIR \
  --creature FILE --training-data DIR \
  [--screening-records 256] [--held-out-records 512] [--max-records 4096]
```

In corpus mode, within the loaded window of T records, the screening records
are the first N, the held-out records are the last M, and the full set is the
window minus its held-out records. The window therefore needs at least N + M
records. Nothing is calibrated, so no records are set aside for it. Seeds then
vary only the arms' random streams. The router-era code instead split the
window into calibration and validation halves and needed 2 x (N + M) records.

AMM reference (offline, opt-in; needs network once to fetch the pinned
artefacts). Run the base bench first, then:

```bash
scripts/amm-reference-env.py /tmp/amm-env   # prints PYTHONPATH=...
env PYTHONPATH=... python3 scripts/amm-reference.py \
  --bench target/release/examples/weight_router_bench \
  --manifest /tmp/wrb-full/manifest.json --out-dir /tmp/amm-out --jobs 18
target/release/examples/weight_router_bench /tmp/wrb-full \
  --compare-reference /tmp/amm-out/amm-runs.jsonl
```

The run writes `manifest.json`, `runs.jsonl` and `summary.json` to `OUT_DIR`
(default `weight-router-bench`):

| File | Contents |
|------|----------|
| `manifest.json` | The manifest that was run. |
| [`runs.jsonl`](evidence/weight-router/runs.jsonl) | One line per run (648 for the synthetic suite), failures included: arm, start and final screening, full and held-out losses, normalised screening and held-out losses, evaluations used, preparation and confirmation evaluations, flat-axis count (`plateauFreeAxes`), error. |
| `summary.json` | Format version (`formatVersion`, 3), crate version (`crateVersion`, 0.1.49), `gitCommit`, `gitDirty`, `neatCoreSource` (the neat-core source pinned in `Cargo.lock`), `wallClockSeconds`, total evaluations, coverage (planned, completed, failed runs), `plateauFreeAxes` per validation seed, and per-cell comparisons: paired differences, mean, median, 95% interval, wins, losses, ties, failures, for screening, held-out and full-window loss, plus the verdict. A corpus run also carries `corpusFingerprint`: FNV-1a 64-bit hashes of the creature file (`creatureFnv1a64`) and of the records actually loaded (`recordsFnv1a64`), and the record count. The synthetic suite omits it, since the crate version and manifest seeds fix its problems. |

Output format version 2 dropped the router-era fields: `calibratedRule` from
`summary.json`, and `routedTo` and `probeEvaluations` from run rows. Format 3
adds the full-window column, `plateauFreeAxes` on rows and cells, and the
provenance fields listed above. The committed
[`runs.jsonl`](evidence/weight-router/runs.jsonl) (about 364 KB) holds the
synthetic run's per-run rows, and the committed
[`summary.json`](evidence/weight-router/summary.json) carries the per-cell
paired distributions. The AMM campaign was regenerated at commit `f0a7b4c`
(crate 0.1.49); later commits leave the benchmark code
(`lamarck/src/weight_bench/`, the `weight_router_bench` example and
`Cargo.lock`) unchanged. The committed `amm-runs.jsonl` (324 rows) no longer
carries `routedTo` or `probeEvaluations`; its rows carry `plateauFreeAxes`. The
AMM reference adds three committed files:

| File | Contents |
|------|----------|
| [`amm-runs.jsonl`](evidence/weight-router/amm-runs.jsonl) | One `RunRow` per AMM validation run (324). |
| [`amm-meta.json`](evidence/weight-router/amm-meta.json) | AMM version, commit, preset and config hash, numpy, scipy and Python versions, coverage and wall time. |
| [`reference-summary.json`](evidence/weight-router/reference-summary.json) | Format version (`formatVersion` 3), crate version (`crateVersion` 0.1.49) and provenance, coverage, total evaluations and per-cell paired comparisons of AMM against refine and CMA-ES, with verdicts. Regenerated from the committed `amm-runs.jsonl` with `--compare-reference`. |

## Results (synthetic suite only)

**Current committed results** (`summary.json`, crate 0.1.49, `formatVersion`
3, git commit `f0a7b4c`, clean tree): `cargo run --release --example
weight_router_bench -- OUT_DIR`, 648 of 648 runs completed, 0 failed, 1,631,615
evaluations, 224.355 s wall on a shared host (indicative only). CMA-ES beat the
random-axis surrogate in **18 of 27** cells, was worse in **0** and inconclusive
in **9**. The per-cell table below gives the current CMA-ES-minus-refine values
and verdicts.

**Changes from format 2** (crate 0.1.47, `formatVersion` 2, the previously
committed evidence). The screening and held-out paired differences and means are
unchanged in all 27 cells. The bootstrap intervals moved, because the full-window
draws now come from the same RNG stream after the other two. Two verdicts
changed:

* tanh, D=100, 30D: `cma-es-worse` is now `inconclusive`. The screening mean
  (+0.0033) is the same; the interval moved from `[0.0001, 0.0066]` to
  `[-0.0001, 0.0065]`.
* hard-tanh, D=40, 100D: `cma-es-better` is now `inconclusive`, because the
  full-window mean is +0.0004 and the full-window term vetoes the verdict. The
  screening mean (-0.0024) is the same.

Format 2 counted 19 better, 1 worse and 7 inconclusive. relu, D=40, 100D stays
`inconclusive`, with interval `[-0.0041, 0.00004]` (format 2: `[-0.0043, 0.00004]`).
That cell sits on the interval boundary and its verdict is not robust.

| Family | D | Budget | CMA-ES − refine (screening mean, 95% CI) | held-out mean | full-window mean | verdict |
|---|---|---|---|---|---|---|
| tanh | 20 | 10D | -0.1485 [-0.1819, -0.1162] | -0.1572 | -0.1540 | cma-es-better |
| tanh | 20 | 30D | -0.0087 [-0.0153, -0.0009] | -0.0162 | -0.0139 | cma-es-better |
| tanh | 20 | 100D | -0.0011 [-0.0019, -0.0005] | -0.0012 | -0.0007 | cma-es-better |
| tanh | 40 | 10D | -0.0899 [-0.1212, -0.0606] | -0.1078 | -0.0985 | cma-es-better |
| tanh | 40 | 30D | -0.0051 [-0.0135, 0.0019] | -0.0063 | -0.0031 | inconclusive |
| tanh | 40 | 100D | -0.0038 [-0.0051, -0.0026] | -0.0024 | -0.0032 | cma-es-better |
| tanh | 100 (extrapolation) | 10D | -0.0459 [-0.0676, -0.0286] | -0.0515 | -0.0515 | cma-es-better |
| tanh | 100 (extrapolation) | 30D | 0.0033 [-0.0001, 0.0065] | 0.0066 | 0.0059 | inconclusive |
| tanh | 100 (extrapolation) | 100D | -0.0028 [-0.0040, -0.0014] | -0.0072 | -0.0060 | cma-es-better |
| relu | 20 | 10D | -0.1481 [-0.1888, -0.1120] | -0.1382 | -0.1459 | cma-es-better |
| relu | 20 | 30D | -0.0205 [-0.0333, -0.0077] | -0.0120 | -0.0179 | cma-es-better |
| relu | 20 | 100D | -0.0005 [-0.0007, -0.0002] | 0.0002 | 0.0006 | inconclusive |
| relu | 40 | 10D | -0.1395 [-0.1623, -0.1170] | -0.1360 | -0.1392 | cma-es-better |
| relu | 40 | 30D | -0.0158 [-0.0227, -0.0088] | -0.0136 | -0.0137 | cma-es-better |
| relu | 40 | 100D | -0.0018 [-0.0041, 0.0000] | -0.0023 | -0.0020 | inconclusive |
| relu | 100 (extrapolation) | 10D | -0.0964 [-0.1078, -0.0844] | -0.0970 | -0.0963 | cma-es-better |
| relu | 100 (extrapolation) | 30D | -0.0003 [-0.0068, 0.0065] | -0.0093 | -0.0040 | inconclusive |
| relu | 100 (extrapolation) | 100D | -0.0011 [-0.0036, 0.0013] | -0.0039 | -0.0035 | inconclusive |
| hard-tanh | 20 | 10D | -0.1653 [-0.2031, -0.1263] | -0.1877 | -0.1790 | cma-es-better |
| hard-tanh | 20 | 30D | -0.0177 [-0.0258, -0.0108] | -0.0179 | -0.0190 | cma-es-better |
| hard-tanh | 20 | 100D | -0.0009 [-0.0013, -0.0005] | -0.0000 | -0.0005 | cma-es-better |
| hard-tanh | 40 | 10D | -0.0901 [-0.1222, -0.0562] | -0.0959 | -0.0887 | cma-es-better |
| hard-tanh | 40 | 30D | 0.0022 [-0.0039, 0.0080] | -0.0006 | 0.0040 | inconclusive |
| hard-tanh | 40 | 100D | -0.0024 [-0.0032, -0.0016] | -0.0004 | 0.0004 | inconclusive |
| hard-tanh | 100 (extrapolation) | 10D | -0.0624 [-0.0839, -0.0441] | -0.0743 | -0.0701 | cma-es-better |
| hard-tanh | 100 (extrapolation) | 30D | 0.0017 [-0.0041, 0.0077] | 0.0034 | 0.0003 | inconclusive |
| hard-tanh | 100 (extrapolation) | 100D | -0.0033 [-0.0054, -0.0013] | -0.0051 | -0.0060 | cma-es-better |

**Earlier router-era run** (crate 0.1.46; not reproducible from the current
code). The full suite and the AMM reference were run twice; re-running reproduced every
cell exactly (determinism check), and the AMM losses were identical.

Run with `cargo run --release --example weight_router_bench -- /tmp/wrb-full`:
crate `neat_ai_lamarck` 0.1.46, neat-core tag v0.23.4, 13 min 34 s wall on a
shared, parallel 7-core container, so the time is indicative only. Coverage was
1404 of 1404 runs completed, 0 failed, and 3,528,609 evaluations in total.

All 324 validation router runs routed to CMA-ES. Verdicts over the 27 cells (12
paired validation seeds each):

| Comparison | First arm better | First arm worse | Inconclusive |
|------------|-----------------:|----------------:|-------------:|
| CMA-ES vs random-axis surrogate (router-era run; format 2: 19 / 1 / 7, format 3: 18 / 0 / 9) | 20 | 1 | 6 |
| Router vs CMA-ES | 0 | 13 | 14 |
| Router vs random-axis surrogate | 19 | 0 | 8 |

Per cell, differences in normalised screening loss (final over start); negative
means the first arm is better. The mean and interval are CMA-ES minus refinement
on screening loss, the held-out column is the same difference on held-out loss,
and the verdict is the format 2 verdict (crate 0.1.47), as it stood before the
format 3 regeneration; the format 3 values are in the table above. The last two
columns are the router-era run's router verdicts.

| Family | D | Budget | CMA-ES − refine (screening mean, 95% CI) | held-out mean | verdict | router − CMA-ES verdict | router − refine verdict |
|---|---|---|---|---|---|---|---|
| tanh | 20 | 10D | -0.1485 [-0.1819, -0.1162] | -0.1572 | cma-es-better | router-worse | router-better |
| tanh | 20 | 30D | -0.0087 [-0.0154, -0.0009] | -0.0162 | cma-es-better | inconclusive | router-better |
| tanh | 20 | 100D | -0.0011 [-0.0018, -0.0006] | -0.0012 | cma-es-better | router-worse | router-better |
| tanh | 40 | 10D | -0.0899 [-0.118, -0.0621] | -0.1078 | cma-es-better | inconclusive | router-better |
| tanh | 40 | 30D | -0.0051 [-0.0138, 0.0019] | -0.0063 | inconclusive | router-worse | inconclusive |
| tanh | 40 | 100D | -0.0038 [-0.005, -0.0026] | -0.0024 | cma-es-better | router-worse | router-better |
| tanh | 100 (extrapolation) | 10D | -0.0459 [-0.0665, -0.0288] | -0.0515 | cma-es-better | inconclusive | router-better |
| tanh | 100 (extrapolation) | 30D | 0.0033 [0.0001, 0.0066] | 0.0066 | cma-es-worse | inconclusive | inconclusive |
| tanh | 100 (extrapolation) | 100D | -0.0028 [-0.0041, -0.0014] | -0.0072 | cma-es-better | router-worse | router-better |
| relu | 20 | 10D | -0.1481 [-0.1819, -0.1125] | -0.1382 | cma-es-better | inconclusive | router-better |
| relu | 20 | 30D | -0.0205 [-0.0338, -0.0083] | -0.012 | cma-es-better | router-worse | router-better |
| relu | 20 | 100D | -0.0005 [-0.0007, -0.0002] | 0.0002 | inconclusive | router-worse | inconclusive |
| relu | 40 | 10D | -0.1395 [-0.162, -0.1169] | -0.136 | cma-es-better | router-worse | router-better |
| relu | 40 | 30D | -0.0158 [-0.0227, -0.0089] | -0.0136 | cma-es-better | inconclusive | router-better |
| relu | 40 | 100D | -0.0018 [-0.0043, 0] | -0.0023 | inconclusive | inconclusive | inconclusive |
| relu | 100 (extrapolation) | 10D | -0.0964 [-0.1077, -0.0843] | -0.097 | cma-es-better | inconclusive | router-better |
| relu | 100 (extrapolation) | 30D | -0.0003 [-0.0069, 0.0061] | -0.0093 | inconclusive | inconclusive | inconclusive |
| relu | 100 (extrapolation) | 100D | -0.0011 [-0.0036, 0.0012] | -0.0039 | inconclusive | inconclusive | inconclusive |
| hard-tanh | 20 | 10D | -0.1653 [-0.2043, -0.1269] | -0.1877 | cma-es-better | router-worse | router-better |
| hard-tanh | 20 | 30D | -0.0177 [-0.0258, -0.0107] | -0.0179 | cma-es-better | router-worse | router-better |
| hard-tanh | 20 | 100D | -0.0009 [-0.0013, -0.0005] | -0 | cma-es-better | router-worse | router-better |
| hard-tanh | 40 | 10D | -0.0901 [-0.1214, -0.055] | -0.0959 | cma-es-better | inconclusive | router-better |
| hard-tanh | 40 | 30D | 0.0022 [-0.0037, 0.0076] | -0.0006 | inconclusive | router-worse | inconclusive |
| hard-tanh | 40 | 100D | -0.0024 [-0.0032, -0.0016] | -0.0004 | cma-es-better | router-worse | router-better |
| hard-tanh | 100 (extrapolation) | 10D | -0.0624 [-0.0837, -0.043] | -0.0743 | cma-es-better | inconclusive | router-better |
| hard-tanh | 100 (extrapolation) | 30D | 0.0017 [-0.004, 0.0079] | 0.0034 | inconclusive | inconclusive | inconclusive |
| hard-tanh | 100 (extrapolation) | 100D | -0.0033 [-0.0055, -0.0013] | -0.0051 | cma-es-better | inconclusive | router-better |

## Findings

1. **CMA-ES beat the random-axis surrogate in 18 of 27 synthetic cells** with
   the format 3 code (19 of 27 in format 2; 20 of 27 in the router-era run; relu
   D=40 100D sits on the interval boundary and is inconclusive), by the
   most at 10D (mean normalised-loss reduction of about 0.05 to 0.17, mostly
   0.09 to 0.17), with the held-out confirmation agreeing in sign in most cells.
   This is a hypothesis for the representative run. It is **not** evidence of a
   NEAT-AI benefit: a fixture built from a teacher network does not demonstrate
   one.
2. **Negative finding: the router never beat CMA-ES alone.** It was worse in 13
   cells and inconclusive in 14. With the calibrated rule it routed every
   validation run to CMA-ES, so it reduces to CMA-ES plus a 17-evaluation probe
   charge. These runs did not show an advantage for that charge. On this suite the landscape probes did
   not discriminate between families. Its wins over the refinement (19 cells)
   are CMA-ES's wins. Per the issue, the router and probes are therefore **not
   integrated** into the production loop, and their offline code has been
   **removed** because it did not earn its maintenance cost. The router-era
   numbers above are kept as the record of this negative finding.
3. **The CMA-ES margin shrinks with budget and is not uniform.** At 30D the
   margins are smaller and five cells are inconclusive (tanh D=40 and D=100,
   relu D=100, hard-tanh D=40 and D=100). The tanh, D=100, 30D cell was
   `cma-es-worse` in format 2 (an extrapolation cell; mean difference +0.0033,
   interval `[0.0001, 0.0066]`, held-out +0.0066). In format 3 it is
   `inconclusive` (interval `[-0.0001, 0.0065]`, full-window mean +0.0059). At
   100D the mean differences are about 0.0005 to 0.004 in size. The data do not
   say why.
4. **Plateaus: none found in the synthetic suite.** For each problem, format 3
   counts how many of its chosen free weights are flat: a one-sided 0.05 step
   changes the loss by at most `PLATEAU_REL_TOL` (1e-6) times the base loss, the
   loss at the start weights. The count reuses the sensitivity scan that chose
   the weights, so it costs no extra evaluations. Every row and every cell has 0
   flat axes (`plateauFreeAxes` is 0 in all 648 runs and in all 324 cell-seed
   entries). This is a measured negative: the suite does not exercise plateaus,
   so it says nothing about how the method behaves on flat axes. The test is
   one-sided (a +0.05 step only), so an axis flat on one side of a kink (relu at
   zero, hard-tanh at its clip) can count as flat, and a flat axis with a kink
   just past +0.05 can count as not flat. Finding 5 measures kinks and plateaus
   with a two-sided test; the representative run still has to measure them on
   real inputs rather than assume them.

5. **Plateaus and kinks, measured with a two-sided diagnosis (added after the
   predeclared run).** `weight_bench::landscape` takes `f(x±h)` and
   `f(x±h/2)` (`h` = 0.05) on every chosen axis from the start weights over the
   screening records, 4 evaluations per axis plus 1 base, outside every arm's
   budget. Building each problem also runs the 145-evaluation selection scan
   (`scan_sensitivities`), which the summary counts once per problem as
   `preparationEvaluations`. Neither count is charged to any arm's budget,
   because the diagnosis is not an arm. An axis is `flat` when both one-sided
   changes are within `PLATEAU_REL_TOL` (1e-6) of the base loss and `edge`
   when exactly one is.
   Otherwise, with `D2(s) = f(x+s) + f(x-s) - 2f(x)`, it is `indeterminate`
   when `|D2(h)|` is at most `CURVATURE_REL_TOL` (1e-9) of the base loss,
   `kinked` when `D2(h)/D2(h/2)` is more than `KINK_RATIO_REL_TOL` (0.25)
   away from 4 (the smooth value; a slope jump pulls it towards 2) or
   `D2(h/2)` has the other sign or is zero, and `smooth` otherwise. A new
   family, `saturated-hard-tanh`, gives 14 of the 16 hidden units a teacher and
   start bias of ±12. That is more than any `|sum(w x)|` can reach, so
   `HARD_TANH` clamps those units and their 112 input weights are exactly flat.
   Only 2 × 8 input and 16 output synapses (32 axes) are live, so D=40 and D=100
   have to choose flat axes. The other families, the predeclared manifest
   and `docs/evidence/weight-router/` are unchanged. Totals over the 12
   validation seeds (1001..1012) in
   `docs/evidence/weight-router-landscape/landscape-summary.json` (crate
   0.1.49, commit `d505d1a`, clean tree, 1.2 s, 51 744 evaluations: 30 864
   diagnosis plus 20 880 selection-scan preparation):

   | Family | D | flat | edge | kinked | smooth | indeterminate | one-sided `plateauFreeAxes` |
   | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
   | tanh | 20 | 0 | 0 | 0 | 240 | 0 | 0 |
   | tanh | 40 | 0 | 0 | 0 | 480 | 0 | 0 |
   | tanh | 100 | 0 | 0 | 1 | 1199 | 0 | 0 |
   | relu | 20 | 0 | 0 | 19 | 221 | 0 | 0 |
   | relu | 40 | 0 | 0 | 67 | 413 | 0 | 0 |
   | relu | 100 | 0 | 0 | 349 | 851 | 0 | 0 |
   | hard-tanh | 20 | 0 | 0 | 39 | 201 | 0 | 0 |
   | hard-tanh | 40 | 0 | 0 | 108 | 372 | 0 | 0 |
   | hard-tanh | 100 | 0 | 0 | 428 | 772 | 0 | 0 |
   | saturated-hard-tanh | 20 | 0 | 0 | 12 | 228 | 0 | 0 |
   | saturated-hard-tanh | 40 | 96 | 1 | 78 | 305 | 0 | 96 |
   | saturated-hard-tanh | 100 | 816 | 1 | 78 | 305 | 0 | 816 |

   The three predeclared families have **no flat or edge axis**, which confirms
   Finding 4 with a two-sided test. Kinks, however, are common: 8 to 29% of the
   relu axes and 16 to 36% of the hard-tanh axes, rising with D. The
   extra axes chosen at larger D are less sensitive ones; the data do not
   establish why those kink more often. The smooth tanh family has **one** kinked axis in 2160
   (D=100, seed 1006). That is a false positive at rounding level:
   `D2(h)` is only about 1.5e-7 of the loss and the ratio is about 13. It is
   reported as measured; the declared tolerances were not changed after the
   run. On `saturated-hard-tanh`, every flat axis is the one-sided scan's
   plateau axis (96 = 8 × 12 at D=40, 816 = 68 × 12 at D=100). The one `edge`
   axis in each of those cells is a live-unit weight that is flat on one side.

   Refine against CMA-ES on `saturated-hard-tanh`, with the same protocol
   (12 seeds, budgets 10D/30D/100D, bootstrap 2000, seed 266) in
   `docs/evidence/weight-router-landscape/saturated/summary.json` (216/216 runs,
   0 failed, 543 852 evaluations, 108.6 s, commit `8badaf5`, clean tree). The
   values are mean paired differences CMA-ES − refine of normalised loss
   (negative favours CMA-ES), as screening `[95% interval]` / full window /
   held out:

   | D | 10D | 30D | 100D |
   | ---: | --- | --- | --- |
   | 20 | inconclusive: −0.01648 `[−0.04012, +0.00890]` / −0.01865 / −0.02217 | cma-es-worse: +0.00925 `[+0.00385, +0.01610]` / +0.01000 / +0.00964 | cma-es-worse: +0.00036 `[+0.00012, +0.00067]` / +0.00027 / +0.00021 |
   | 40 | inconclusive: −0.01433 `[−0.05913, +0.01988]` / −0.01134 / −0.01241 | cma-es-worse: +0.01442 `[+0.00662, +0.02450]` / +0.01439 / +0.01440 | cma-es-worse: +0.00272 `[+0.00095, +0.00517]` / +0.00290 / +0.00301 |
   | 100 | cma-es-better: −0.01628 `[−0.03180, −0.00312]` / −0.01679 / −0.01810 | cma-es-worse: +0.00772 `[+0.00219, +0.01680]` / +0.00737 / +0.00727 | cma-es-worse: +0.00339 `[+0.00107, +0.00686]` / +0.00354 / +0.00395 |

   The full-window and held-out means agree in sign with the screening mean
   in all 9 cells. This **reverses Finding 1 on this family**: CMA-ES is worse
   in 6 of 9 cells (every 30D and 100D cell), better only at D=100, 10D, and
   inconclusive at 10D for D=20 and D=40. On plain hard-tanh it was better in
   6 of 9 cells and worse in none. The reversal is **not explained by flat
   axes alone**: D=20 has no flat axis yet shows the same 30D/100D losses.
   That points to the family's very different live landscape (2 live hidden
   units) as the cause, not just the plateaus. D=100 is extrapolation. These
   are synthetic fixtures, so they are a hypothesis for the representative
   run, not evidence about NEAT-AI creatures.

   Reproduce (from `lamarck/`, release build). The arm run reads its manifest
   from outside its `OUT_DIR`, because it writes `OUT_DIR/manifest.json`
   itself. The committed files were written to a scratch directory, so the
   provenance probe saw a clean tree, and then copied in:

   ```bash
   cargo run --release --example weight_router_bench -- ../docs/evidence/weight-router-landscape \
       --manifest ../docs/evidence/weight-router-landscape/manifest.json --landscape
   cargo run --release --example weight_router_bench -- ../docs/evidence/weight-router-landscape/saturated \
       --manifest ../docs/evidence/weight-router-landscape/saturated-manifest.json
   cargo test --test weight_bench_landscape
   ```

## AMM reference results

Coverage was 324 of 324 AMM validation runs completed, 0 failed, and 855,000
evaluations (budgeted, preparation per AMM run and 5 confirmation evaluations
per run). The campaign was regenerated at commit `f0a7b4c` (crate 0.1.49); later
commits leave the benchmark code (`lamarck/src/weight_bench/`, the
`weight_router_bench` example and `Cargo.lock`) unchanged. It ran with
`--jobs 8`: 133.038 s wall on a shared host, so the time is indicative only. Its
normalised screening losses are identical to the previously committed router-era
rows.

**Verdict: `amm-reference-worse` in all 27 cells against both refine and
CMA-ES** in the committed `reference-summary.json` (54 of 54 comparisons; 0
better, 0 inconclusive), regenerated by the current code from the committed AMM
rows. The router-era run also found AMM worse than the router in all 27 cells
(81 of 81 comparisons then). The `router` column below is from that run.

Median normalised screening loss (final over start, lower is better) on the
validation seeds, and AMM's held-out loss. The refine, CMA-ES and AMM columns
were recomputed at format 3 from the committed `runs.jsonl` and `amm-runs.jsonl`
and are unchanged; the router column is from the router-era run:

| Family | D | Budget | refine | cma-es | router | AMM | AMM held-out |
|---|---|---|---|---|---|---|---|
| hard-tanh | 20 | 10D | 0.4242 | 0.2324 | 0.2364 | 0.7336 | 0.7391 |
| hard-tanh | 20 | 30D | 0.1840 | 0.1692 | 0.1692 | 0.3459 | 0.3809 |
| hard-tanh | 20 | 100D | 0.1501 | 0.1488 | 0.1488 | 0.2872 | 0.2927 |
| hard-tanh | 40 | 10D | 0.2491 | 0.1498 | 0.1524 | 0.8473 | 0.8467 |
| hard-tanh | 40 | 30D | 0.1040 | 0.0972 | 0.0972 | 0.2673 | 0.3030 |
| hard-tanh | 40 | 100D | 0.0722 | 0.0705 | 0.0705 | 0.2716 | 0.3255 |
| hard-tanh | 100 (extrapolation) | 10D | 0.1754 | 0.1039 | 0.1039 | 0.9316 | 0.9366 |
| hard-tanh | 100 (extrapolation) | 30D | 0.0428 | 0.0547 | 0.0547 | 0.1927 | 0.2663 |
| hard-tanh | 100 (extrapolation) | 100D | 0.0176 | 0.0158 | 0.0158 | 0.2077 | 0.2651 |
| relu | 20 | 10D | 0.4814 | 0.2898 | 0.2898 | 0.8605 | 0.8711 |
| relu | 20 | 30D | 0.2497 | 0.2450 | 0.2453 | 0.3841 | 0.4380 |
| relu | 20 | 100D | 0.2215 | 0.2212 | 0.2213 | 0.4202 | 0.4691 |
| relu | 40 | 10D | 0.3354 | 0.1927 | 0.1957 | 0.8789 | 0.8913 |
| relu | 40 | 30D | 0.1338 | 0.1257 | 0.1257 | 0.3244 | 0.3886 |
| relu | 40 | 100D | 0.0945 | 0.0944 | 0.0944 | 0.3330 | 0.4232 |
| relu | 100 (extrapolation) | 10D | 0.2098 | 0.1095 | 0.1095 | 0.9511 | 0.9488 |
| relu | 100 (extrapolation) | 30D | 0.0489 | 0.0568 | 0.0568 | 0.2071 | 0.3255 |
| relu | 100 (extrapolation) | 100D | 0.0225 | 0.0211 | 0.0211 | 0.2328 | 0.3435 |
| tanh | 20 | 10D | 0.3496 | 0.2083 | 0.2151 | 0.7039 | 0.7690 |
| tanh | 20 | 30D | 0.1413 | 0.1503 | 0.1503 | 0.2861 | 0.3101 |
| tanh | 20 | 100D | 0.1208 | 0.1203 | 0.1203 | 0.3055 | 0.3286 |
| tanh | 40 | 10D | 0.2188 | 0.1286 | 0.1297 | 0.7429 | 0.7164 |
| tanh | 40 | 30D | 0.0850 | 0.0714 | 0.0714 | 0.2322 | 0.2777 |
| tanh | 40 | 100D | 0.0550 | 0.0515 | 0.0515 | 0.2272 | 0.2708 |
| tanh | 100 (extrapolation) | 10D | 0.1117 | 0.0756 | 0.0756 | 0.8845 | 0.9272 |
| tanh | 100 (extrapolation) | 30D | 0.0366 | 0.0368 | 0.0370 | 0.1622 | 0.1963 |
| tanh | 100 (extrapolation) | 100D | 0.0157 | 0.0139 | 0.0139 | 0.1728 | 0.2134 |

Interpretation (a negative finding, stated carefully). On this suite, under the
predeclared box, AMM as published is a poor fit for local weight refinement from
a good incumbent. At 10D its median final/start loss is 0.70 to 0.95; the 30D and 100D
results are lower but trail the other arms. This is a result for the declared
box and surrogate baseline. A likely reason,
offered as a **hypothesis, not a finding**: AMM is a box-global method whose
probes and engines scale to the box, here ±0.5 around the start, whereas the
local arms take steps of 0.05. A box-size sensitivity sweep was **not** run,
because choosing a box after seeing these results would be post-hoc.

Consequence: nothing from AMM is ported, its probes do not justify a port and,
per the issue, the in-repo router stays unported too. No default changed. The
synthetic suite does not demonstrate any NEAT-AI benefit either way; the
representative run on the private corpus remains unperformed.

## Controls and remaining experiment

Inspection found that `refine.rs` reuses two production constants, but manually
recreates mirrors and follow-up steps. It does not call `generate_candidate_batch`,
`mirror_candidate` or `FollowUpPlan`, nor use the production focus/statistics/
learning pipeline. Its uniform half-width and CMA-ES's coordinate standard
deviation also define different search radii. The archived comparison therefore
has a **surrogate baseline**, not baseline equivalence. Correcting that baseline
changes the protocol and needs fresh validation data; the archived results must
not select parameters for the corrected experiment.

The local input search on 14 November 2026 found GRQ test models, but no
production model/corpus pair at the campaign's documented paths or in the local
GRQ checkouts. Test fixtures are not representative evidence. Representative
coverage is **0 runs**; no private model/data versions have been selected and no
representative manifest is frozen. Private inputs, their provenance manifests
and results must remain outside this public checkout.

A corpus run records `corpusFingerprint` (creature-file and loaded-record
hashes) in its results, so the model and records it used can be checked
against a frozen provenance manifest. Before a representative run, freeze the
model revision/hash, corpus snapshot and ordered-file hashes, screening/full/held-out split, and baseline pipeline
configuration in its private manifest. Keep D=20/40/100, 10D/30D/100D and the
three activation families in the public synthetic controls. Keep twelve paired
validation trials per cell. The current arms have no tuned parameter, so there
is no calibration phase; any future tuned parameter (a router rule, a box or
radius) needs its own calibration seeds and records, disjoint from validation.
A revised protocol uses fresh validation problems/seeds and untouched
validation records. Repeated
seeds on one fixed corpus measure optimiser randomness, not twelve independent
models or corpora. Report that sampling limitation with the uncertainty.

The remaining comparison must charge preparation, any calibration, search
and confirmation separately, enforce the per-arm evaluation ceilings, and set
an overall wall-clock ceiling before launch. Use the declared paired bootstrap
and held-out criterion as descriptive evidence, then confirm candidates through
the full authoritative scorer. Corpus mode's `full` is only the loaded window
minus its held-out tail, **not the entire corpus**. This benchmark exports
metrics; it does not invoke the authoritative acceptance pipeline or install a
candidate. Ordinary production acceptance in `run.rs` is unchanged.

## Verification and process failures

```bash
cargo test --test weight_bench --test weight_bench_reference --test weight_bench_landscape
cargo test --lib failed_confirmations_keep_the_calls_already_spent
# Integration checks, using the pinned scratch Python environment and a built bench:
env PYTHONPATH=... python3 scripts/test-amm-reference.py --bench BIN
```

`Manifest::validate` refuses duplicate cells/seeds; `WeightObjective::new`
requires positive input/output counts and distinct synapse indices. The tests
also check fixed starting weights, immutable parents, disjoint corpus halves
and budget refusal. Successful confirmation uses five calls; failures retain
the attempts already spent. Failed rows are excluded from paired distributions
even when their metrics are populated, while their costs stay in the totals.

The Python driver records startup failures and non-zero exits after a completed
row as failures. Server stderr goes to a temporary file so verbose diagnostics
cannot fill a pipe and block objective replies. A missing final row is labelled
`costs incomplete`; its zero/default cost fields are not a complete measurement.
Use a bounded foreground command for offline campaigns, for example
`timeout 900 env PYTHONPATH=... python3 scripts/amm-reference.py ...`.
These process checks use the real AMM package and Rust server; they are separate
from routine feature CI and are not representative optimisation measurements.

## Outstanding

* The comparison against the actual production refinement pipeline and a frozen
  representative model/corpus campaign, including full authoritative confirmation.
* Plateau axes among the free weights on a representative problem. The
  predeclared synthetic suite has none (Findings 4 and 5); only the added
  `saturated-hard-tanh` family has them by construction (Finding 5).
* Box-size sensitivity and search-radius controls, predeclared on fresh data.
  No production optimiser or default change is supported by the current evidence.
