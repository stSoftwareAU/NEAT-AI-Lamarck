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
output of the current code (crate 0.1.47, `formatVersion` 2) and are reproduced
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

* **Families.** Hidden squash `tanh` (smooth), `relu` (kinks) and `hard-tanh`
  (kinks and plateaus).
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
  `KINK_ASYMMETRY = 0.5` of the sum of their magnitudes.
* **Calibration** (removed; router-era run only). Grid search on calibration seeds only: `minBudgetMultiple` in
  `{10, 30, 100, never}` by `maxNonSmoothShare` in `{0.25, 0.5, 0.75, 1.01}`,
  minimising the mean normalised loss of the arm each row would be routed to.
  Calibration scores each routed row with that arm's full-budget calibration
  result, so it does not charge the router's 17 probe evaluations; validation
  runs the router itself and does charge them.
  Calibrated result: `minBudgetMultiple` 10, `maxNonSmoothShare` 0.25,
  `probeAxes` 8.
* **Metric.** Normalised loss, final over start, on the screening records;
  confirmed on the held-out records. Negative paired differences favour the first
  arm.
* **Success and failure criteria** (`verdict` in `experiment.rs`). Per cell, the
  paired differences over the 12 validation seeds are bootstrapped (2000
  resamples, seed 266, 95% percentile interval of the mean). The first arm is
  `-better` iff the interval's upper bound on screening loss is below zero **and**
  the held-out mean difference is at most zero. It is `-worse` iff the interval's
  lower bound is above zero. Otherwise the cell is `inconclusive`. A run that
  fails is recorded with its error and counted in coverage. In validation it is
  dropped from that pair even if it retained finite metrics (counted under
  `failures`). In the router-era calibration it counted as 1.0, no improvement.
  The router-era and current runs had no failures.
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
at the budget. Its probes depend on the box width; the local router here uses
different plateau/kink tests and is not AMM. The tested dimension range ends at
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
three families, three dimensions and three budgets. It explicitly supplies
`c46` and converts `SeedSequence` to its first 32-bit word so both public APIs
receive the same seed and configuration. This checks completed RunRows, not
every evaluation trace or every AMM route. The source/dependency pins and
observed result are recorded in
[`method-check.json`](evidence/weight-router/method-check.json). That file is **router-era output** (crate 0.1.46, code checkpoint
`8f77004`), like `amm-runs.jsonl`: the manifest it embeds still carries
`calibrationSeeds`, which the current `Manifest` refuses, and its binary hash
is of the removed code. It was not re-run after the router was removed, because
no scratch AMM environment was available. Re-running the check with the current
manifest records a different `manifest` field, and its `canonicalRowsSha256`
has not been confirmed to reproduce.

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
| `runs.jsonl` | One line per run, failures included: arm, start and final screening, full and held-out losses, evaluations used, preparation and confirmation evaluations, error. |
| `summary.json` | Format version (`formatVersion`, 2), crate version (`crateVersion`, 0.1.47), total evaluations, coverage (planned, completed, failed runs) and per-cell comparisons: paired differences, mean, median, 95% interval, wins, losses, ties, failures, for screening and held-out loss, plus the verdict. |

Output format version 2 drops the router-era fields: `calibratedRule` from
`summary.json`, and `routedTo` and `probeEvaluations` from run rows. The
committed `amm-runs.jsonl` was written by the router-era driver and still
carries `routedTo` and `probeEvaluations`; run rows tolerate and ignore them.
`runs.jsonl` (about 360 KB) is not committed. The committed
[`summary.json`](evidence/weight-router/summary.json) carries the per-cell paired
distributions. The AMM reference adds three committed files:

| File | Contents |
|------|----------|
| [`amm-runs.jsonl`](evidence/weight-router/amm-runs.jsonl) | One `RunRow` per AMM validation run (324). |
| [`amm-meta.json`](evidence/weight-router/amm-meta.json) | AMM version, commit, preset and config hash, numpy, scipy and Python versions, coverage and wall time. |
| [`reference-summary.json`](evidence/weight-router/reference-summary.json) | Crate version (`formatVersion` 2, `crateVersion` 0.1.47), coverage, total evaluations and per-cell paired comparisons of AMM against refine and CMA-ES, with verdicts. Regenerated from the committed `amm-runs.jsonl` with `--compare-reference`. |

## Results (synthetic suite only)

**Current committed results** (`summary.json`, crate 0.1.47, `formatVersion`
2): `cargo run --release --example weight_router_bench -- OUT_DIR`, 648 of 648
runs completed, 0 failed, 1,631,615 evaluations, 3 min 32 s single-threaded on a
shared container (indicative only). CMA-ES beat the random-axis surrogate in
**19 of 27** cells, was worse in 1 and inconclusive in 7. The CMA-ES-minus-refine
paired differences are identical to the router-era run's in all 27 cells, on
screening and held-out loss, as are the refine and CMA-ES medians below. The
bootstrap RNG no longer draws resamples for the two router pairs first, so the
intervals moved slightly. One verdict changed: relu, D=40, 100D is now
`inconclusive`, with interval `[-0.0043, 0.00004]`. That cell sits on the
interval boundary and its verdict is not robust. The per-cell table below shows
the current CMA-ES-minus-refine values and verdicts.

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
| CMA-ES vs random-axis surrogate (router-era run; 19 / 1 / 7 now) | 20 | 1 | 6 |
| Router vs CMA-ES | 0 | 13 | 14 |
| Router vs random-axis surrogate | 19 | 0 | 8 |

Per cell, differences in normalised screening loss (final over start); negative
means the first arm is better. The mean and interval are CMA-ES minus refinement
on screening loss, the held-out column is the same difference on held-out loss,
and the verdict is the current code's; all three come from the committed
`summary.json`. The last two columns are the router-era run's router verdicts.

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

1. **CMA-ES beat the random-axis surrogate in 19 of 27 synthetic cells** with
   the current code (20 of 27 in the router-era run; relu D=40 100D sits on the
   interval boundary), by the
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
   margins are smaller and four cells are inconclusive (tanh D=40, relu D=100,
   hard-tanh D=40 and D=100). The one `cma-es-worse` cell is tanh, D=100, 30D
   (an extrapolation cell; mean difference +0.0033, interval `[0.0001, 0.0066]`,
   held-out +0.0066). At 100D the mean differences are about 0.001 to 0.004 in
   size. The data do not say why.

## AMM reference results

Coverage was 324 of 324 AMM validation runs completed, 0 failed, and 855,000
evaluations (budgeted, preparation per AMM run and 5 confirmation evaluations
per run). AMM took 100.2 s wall with
18 workers on a shared, parallel container, so the time is indicative only.

**Verdict: `amm-reference-worse` in all 27 cells against both refine and
CMA-ES** in the committed `reference-summary.json` (54 of 54 comparisons; 0
better, 0 inconclusive), regenerated by the current code from the committed AMM
rows. The router-era run also found AMM worse than the router in all 27 cells
(81 of 81 comparisons then). The `router` column below is from that run.

Median normalised screening loss (final over start, lower is better) on the
validation seeds, and AMM's held-out loss:

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

Before a representative run, freeze the model revision/hash, corpus snapshot
and ordered-file hashes, screening/full/held-out split, and baseline pipeline
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
cargo test --test weight_bench --test weight_bench_reference
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
* Box-size sensitivity and search-radius controls, predeclared on fresh data.
  No production optimiser or default change is supported by the current evidence.
