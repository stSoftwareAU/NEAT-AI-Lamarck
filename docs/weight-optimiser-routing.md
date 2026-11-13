# Budget-aware routing of local weight optimisers (issue #266)

**Status: opt-in offline experiment.** Nothing in the optimisation loop calls
this code: it runs only when the `weight_router_bench` example (or a test)
drives it, so ordinary runs are unchanged. No default was changed and nothing
was ported. The synthetic suite has been run and is reported below; the
**representative run on the private production creature and corpus has not been
made**. AMM (arXiv 2611.06879) is **not implemented**, because the paper could
not be retrieved from the build sandbox (see [Outstanding](#outstanding)).

The code is [`lamarck/src/weight_bench/`](../lamarck/src/weight_bench/mod.rs),
the driver is
[`lamarck/examples/weight_router_bench.rs`](../lamarck/examples/weight_router_bench.rs),
and the committed evidence is in
[`docs/evidence/weight-router/`](evidence/weight-router/summary.json).

## Question and hypothesis

Lamarck's local weight refinement spends a small evaluation budget on random
axis moves. Is a covariance-adapting optimiser (CMA-ES) a better use of the same
budget, and can a cheap rule that probes the landscape and routes by budget and
smoothness do at least as well as either fixed choice?

Hypothesis: with enough budget on a smooth landscape CMA-ES beats the current
refinement, and a router that spends a few evaluations probing, then picks the
better arm, beats both. The router is only worth keeping if it beats CMA-ES
alone.

## Arms

| Arm | What it is |
|-----|------------|
| `lamarck-refine` | The current refinement replayed on a fixed weight set. Each round picks an axis and a delta drawn like the Random strategy (uniform in `±RANDOM_DELTA_HALF_WIDTH`, `0.05`, from `candidates.rs`, zero excluded), tries the mirrored pair `x ± delta` as `mirror.rs` does, and accepts the better one only if it strictly improves. After an accept it probes the #219 follow-up `STEP_FRACTIONS` (`[0.5, -0.5, 1.0]`, from `followup.rs`) along the winning move and takes the best strict improvement. |
| `cma-es` | Full-covariance (mu/mu_w, lambda)-CMA-ES, written in-repo with no third-party dependency. Strategy parameters are Hansen's tutorial defaults (arXiv:1604.00772); the eigendecomposition is a cyclic Jacobi method. Initial step size `CMA_SIGMA0 = 0.05`, deliberately equal to the refinement's step scale so neither arm starts with a larger search radius. |
| `router` | Spends `2 * probe_axes + 1` evaluations probing the landscape, then runs `lamarck-refine` or `cma-es` on the remaining budget. The probe is charged to the **same** budget. The chosen arm re-evaluates the start, one further counted evaluation. |
| AMM | **Missing.** Not implemented; see [Outstanding](#outstanding). |

The router picks CMA-ES iff the total budget is at least `minBudgetMultiple`
times the dimension **and** the probe's plateau share plus kink share is at most
`maxNonSmoothShare`; otherwise it picks the refinement.

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
* **Seeds.** Calibration seeds 1 to 8 choose the router rule; validation seeds
  1001 to 1012 are untouched by calibration (the manifest is rejected if the two
  overlap). Each validation seed runs all three arms on the same problem, so
  differences are paired.
* **Landscape probe** (`probes.rs`). Up to `ROUTER_PROBE_AXES = 8` evenly spaced
  axes, each evaluated at `x0 ± 0.05 e_i` (`2 x 8 + 1 = 17` evaluations). An
  axis is a **plateau** when both moves change the loss by at most
  `PLATEAU_REL_TOL = 1e-6` relative to the start loss. Otherwise it is a
  **kink** when the left and right slopes differ by more than
  `KINK_ASYMMETRY = 0.5` of the sum of their magnitudes.
* **Calibration.** Grid search on calibration seeds only: `minBudgetMultiple` in
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
  dropped from that pair (counted under `failures`); in calibration it counts as
  1.0, no improvement. This run had none.
* **Compute limits.** Each run's evaluation budget is enforced by
  `BudgetedObjective`, which refuses (fails loud with `BudgetExhausted`) any
  evaluation past the limit.
* **Cost accounting.** Total evaluations include preparation (145 per problem),
  the calibration probe passes, the router's 17 probe evaluations per run and 4
  confirmation evaluations per run (`CONFIRMATION_EVALUATIONS`). The probe and
  the start re-evaluation come out of the router's budget.

## How to run

```bash
# Predeclared synthetic experiment (about 14 minutes single-threaded, release build)
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

In corpus mode the screening records are the first N in file order, the held-out
records are the last M of the loaded window, and the full set is the loaded
window minus the held-out records. Seeds then vary only the arms' random
streams.

The run writes `manifest.json`, `runs.jsonl` and `summary.json` to `OUT_DIR`
(default `weight-router-bench`):

| File | Contents |
|------|----------|
| `manifest.json` | The manifest that was run. |
| `runs.jsonl` | One line per run, failures included: arm, routed-to arm, start and final screening, full and held-out losses, evaluations used, probe, preparation and confirmation evaluations, error. |
| `summary.json` | Calibrated rule, total evaluations, coverage (planned, completed, failed runs) and per-cell comparisons: paired differences, mean, median, 95% interval, wins, losses, ties, failures, for screening and held-out loss, plus the verdict. |

`runs.jsonl` (about 830 KB) is not committed. The committed
[`summary.json`](evidence/weight-router/summary.json) carries the per-cell paired
distributions.

## Results (synthetic suite only)

Run with `cargo run --release --example weight_router_bench -- /tmp/wrb-full`:
crate `neat_ai_lamarck` 0.1.46, neat-core tag v0.23.4, 13 min 34 s wall on a
shared, parallel 7-core container, so the time is indicative only. Coverage was
1404 of 1404 runs completed, 0 failed, and 3,704,685 evaluations in total.

All 324 validation router runs routed to CMA-ES. Verdicts over the 27 cells (12
paired validation seeds each):

| Comparison | First arm better | First arm worse | Inconclusive |
|------------|-----------------:|----------------:|-------------:|
| CMA-ES vs current refinement | 20 | 1 | 6 |
| Router vs CMA-ES | 0 | 13 | 14 |
| Router vs current refinement | 19 | 0 | 8 |

Per cell, differences in normalised screening loss (final over start); negative
means the first arm is better. The mean and interval are CMA-ES minus refinement
on screening loss, the held-out column is the same difference on held-out loss,
and the last two columns are the router verdicts.

| Family | D | Budget | CMA-ES − refine (screening mean, 95% CI) | held-out mean | verdict | router − CMA-ES verdict | router − refine verdict |
|---|---|---|---|---|---|---|---|
| tanh | 20 | 10D | -0.1485 [-0.1804, -0.1151] | -0.1572 | cma-es-better | router-worse | router-better |
| tanh | 20 | 30D | -0.0087 [-0.0155, -0.0008] | -0.0162 | cma-es-better | inconclusive | router-better |
| tanh | 20 | 100D | -0.0011 [-0.0019, -0.0006] | -0.0012 | cma-es-better | router-worse | router-better |
| tanh | 40 | 10D | -0.0899 [-0.12, -0.0607] | -0.1078 | cma-es-better | inconclusive | router-better |
| tanh | 40 | 30D | -0.0051 [-0.0134, 0.0021] | -0.0063 | inconclusive | router-worse | inconclusive |
| tanh | 40 | 100D | -0.0038 [-0.005, -0.0026] | -0.0024 | cma-es-better | router-worse | router-better |
| tanh | 100 (extrapolation) | 10D | -0.0459 [-0.0673, -0.0283] | -0.0515 | cma-es-better | inconclusive | router-better |
| tanh | 100 (extrapolation) | 30D | 0.0033 [0, 0.0068] | 0.0066 | cma-es-worse | inconclusive | inconclusive |
| tanh | 100 (extrapolation) | 100D | -0.0028 [-0.0041, -0.0015] | -0.0072 | cma-es-better | router-worse | router-better |
| relu | 20 | 10D | -0.1481 [-0.1856, -0.1102] | -0.1382 | cma-es-better | inconclusive | router-better |
| relu | 20 | 30D | -0.0205 [-0.0339, -0.0087] | -0.012 | cma-es-better | router-worse | router-better |
| relu | 20 | 100D | -0.0005 [-0.0007, -0.0002] | 0.0002 | inconclusive | router-worse | inconclusive |
| relu | 40 | 10D | -0.1395 [-0.163, -0.1164] | -0.136 | cma-es-better | router-worse | router-better |
| relu | 40 | 30D | -0.0158 [-0.0225, -0.0088] | -0.0136 | cma-es-better | inconclusive | router-better |
| relu | 40 | 100D | -0.0018 [-0.0044, -0] | -0.0023 | cma-es-better | inconclusive | inconclusive |
| relu | 100 (extrapolation) | 10D | -0.0964 [-0.1073, -0.0841] | -0.097 | cma-es-better | inconclusive | router-better |
| relu | 100 (extrapolation) | 30D | -0.0003 [-0.0071, 0.0064] | -0.0093 | inconclusive | inconclusive | inconclusive |
| relu | 100 (extrapolation) | 100D | -0.0011 [-0.0038, 0.0013] | -0.0039 | inconclusive | inconclusive | inconclusive |
| hard-tanh | 20 | 10D | -0.1653 [-0.2051, -0.1267] | -0.1877 | cma-es-better | router-worse | router-better |
| hard-tanh | 20 | 30D | -0.0177 [-0.0253, -0.0106] | -0.0179 | cma-es-better | router-worse | router-better |
| hard-tanh | 20 | 100D | -0.0009 [-0.0013, -0.0005] | -0 | cma-es-better | router-worse | router-better |
| hard-tanh | 40 | 10D | -0.0901 [-0.1237, -0.0562] | -0.0959 | cma-es-better | inconclusive | router-better |
| hard-tanh | 40 | 30D | 0.0022 [-0.0037, 0.0078] | -0.0006 | inconclusive | router-worse | inconclusive |
| hard-tanh | 40 | 100D | -0.0024 [-0.0032, -0.0016] | -0.0004 | cma-es-better | router-worse | router-better |
| hard-tanh | 100 (extrapolation) | 10D | -0.0624 [-0.0839, -0.0436] | -0.0743 | cma-es-better | inconclusive | router-better |
| hard-tanh | 100 (extrapolation) | 30D | 0.0017 [-0.0041, 0.0074] | 0.0034 | inconclusive | inconclusive | inconclusive |
| hard-tanh | 100 (extrapolation) | 100D | -0.0033 [-0.0055, -0.0014] | -0.0051 | cma-es-better | inconclusive | router-better |

## Findings

1. **CMA-ES beat the current refinement in 20 of 27 synthetic cells**, by the
   most at 10D (mean normalised-loss reduction of about 0.05 to 0.17, mostly
   0.09 to 0.17), with the held-out confirmation agreeing in sign in most cells.
   This is a hypothesis for the representative run. It is **not** evidence of a
   NEAT-AI benefit: a fixture built from a teacher network does not demonstrate
   one.
2. **Negative finding: the router never beat CMA-ES alone.** It was worse in 13
   cells and inconclusive in 14. With the calibrated rule it routed every
   validation run to CMA-ES, so it reduces to CMA-ES plus a 17-evaluation probe
   charge, which is why it cannot win. On this suite the landscape probes did
   not discriminate between families. Its wins over the refinement (19 cells)
   are CMA-ES's wins. Per the issue, the router and probes are therefore **not
   ported** to the production loop. If the representative run repeats this, the
   router and probe code should be removed rather than maintained.
3. **The CMA-ES margin shrinks with budget and is not uniform.** At 30D the
   margins are smaller and four cells are inconclusive (tanh D=40, relu D=100,
   hard-tanh D=40 and D=100). The one `cma-es-worse` cell is tanh, D=100, 30D
   (an extrapolation cell; mean difference +0.0033, interval `[0, 0.0068]`,
   held-out +0.0066). At 100D the mean differences are about 0.001 to 0.004 in
   size. The data do not say why.

## Outstanding

* **AMM reproduction.** The paper (<https://arxiv.org/abs/2611.06879>) could not
  be retrieved from the sandbox: the web fetch reported "proxy refused the
  connection" and a direct request to `export.arxiv.org` failed with "CONNECT
  tunnel failed, response 403". AMM's method is therefore not pinned, and
  comparing against a guess would be misleading, so it is not implemented.
* **Representative corpus run (owner step).** Run the corpus mode above on the
  private production creature and corpus, on a quiet machine. No such run has
  been made. Private results stay private.
* **Any port decision** waits on that run. Until then no default changes.
