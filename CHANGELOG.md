# Changelog

All notable changes to NEAT-AI-Lamarck are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- **The candidate generator's per-phase quotas can scale with the budget (Issue
  #108).** `--scale-candidate-quotas` keeps generating after the fixed opening
  quotas are spent, sweeping the ranked-source × weight-scale and
  ranked-source × squash grids a slice of every strategy family per round, so
  `--candidates N` binds until the generator is genuinely exhausted instead of
  topping out at ~29. Duplicate proposals are rejected rather than counted, and
  each batch reports whether the **budget**, the **fixed quota ceiling** or
  genuine **exhaustion** bound it — logged per experiment and journalled as
  `candidatesRequested` / `batchLimit`, with `report` summarising the achieved
  size in a `candidateBatch` bucket. On a production-shaped creature (2511
  inputs) the budget now binds at every count measured up to 240, at ~0.08 ms
  per candidate (`cargo run --release --example candidate_quota_bench`); the
  fixed quotas stop at 27, of which only 22 are distinct. The flag is
  **opt-in**: no default changes until the paired `candidate-quotas` arm of
  `scripts/run-followup-economics.sh` prices a bigger batch in promote-scores
  per scorer-minute — see `docs/followup-economics.md` Arm 5.

- **The per-experiment analysis scans fold record chunks across cores (Issue
  #107).** Both scans are read-only reductions, so they now run on
  `--analysis-threads` workers (default `4`, `0` aborts the run). Determinism
  comes from the partition rather than the schedule: the sample is cut into
  fixed 2048-record chunks — a function of the sample alone, never of the thread
  count or the host — and the per-chunk partials are merged in ascending chunk
  order, so 1, 2 and 8 threads produce **bit-identical** accumulators and
  `--seed` replay is unaffected. Every RNG draw (`select_sparse`) stays on the
  calling thread, ahead of the parallel region, and a creature that is not
  `forwardOnly` is folded as a single chunk because its activations carry state
  between records. Measured on the 10-core M4 host at production sample shape:
  the analysis phase is **1.9× faster at 2 threads, 3.1× at the default 4 and
  4.1× at 8** (`cargo run --release --example analysis_threads_bench`). The
  thread count in force is recorded in the journal `runHeader` as
  `analysisThreads`.

- **The three exclusive-box economics arms are wired up (Issue #96).**
  `scripts/run-followup-economics.sh` gains an `output-neuron` arm (pins
  `--focus-neuron output-0`, the slice `--focus-policy high-error` cannot reach
  on a fine-tuned creature) and a `backprop-cap` arm (a cap ladder on one seed).
  Neither is in the default arm set: like `multi-seed` they need the production
  creature and exclusive use of the scorer. The cap arm needed a knob that did
  not exist — `--backprop-max-bias-adjustment-scale` overrides
  `BackpropConfig::maximum_bias_adjustment_scale` (default `10`), mirroring
  `--backprop-learning-rate`: non-positive or non-finite aborts the run rather
  than silently reverting to the default, and the cap in force is recorded in
  the journal `runHeader` so an arm is identifiable from its journal alone. The
  measurements themselves remain unrun — they need ~4.5 hours of exclusive time
  on the production box.

- **The #8 baseline's follow-up economics experiments were run (Issue #75).**
  `docs/followup-economics.md` records 118 further experiments across three
  arms: an output-focus slice, a backprop step A/B and a batch-size A/B.
  `--backprop-learning-rate` is a new validated CLI knob (non-positive or
  non-finite aborts the run rather than silently reverting to `0.01`) and is
  recorded in the journal `runHeader`. `scripts/run-followup-economics.sh`
  drives the arms one at a time and `scripts/summarise-followup-economics.sh`
  turns their reports into the document's tables. **No strategy is disabled**;
  two findings explain why `backprop` and larger batches never paid: the
  backprop bias step saturates `maximum_bias_adjustment_scale` at any learning
  rate, and candidate generation has a fixed per-phase ceiling (~29 on the GRQ
  creature) so `--candidates` above it buys nothing.

- **Combo and graft wins are attributed to a strategy (Issue #74).** An
  acceptance now journals `comboMemberIndices` — the candidate indices behind
  the winner — so `report` can credit a merged `combo-NNN-kM` win to *every*
  member strategy instead of dropping it. Each `strategies[]` row carries the
  subset earned in a merge as `comboWins`, and a pre-#74 combo win (which names
  no members) is counted in `comboAcceptancesUnattributed` rather than silently
  ignored. Phase-G replay writes its own `graftReplay` journal line — grafts
  applied, baseline/after score, Δ, elapsed, scorer counters and any
  `replayError` — which `report` surfaces as a `graftReplay` bucket (`null` for
  journals without one), so a graft accept with no candidate stem is no longer
  invisible.

- **`--max-experiments` and graceful cancellation (Issue #72).** The loop now
  stops on the first of four rules instead of two: the wall-clock timeout, the
  new `--max-experiments N` cap (recorded in the journal `runHeader` as
  `maxExperiments`), `SIGINT`/`SIGTERM`, or three consecutive scorer failures.
  The signal handler only sets an `AtomicBool`; the loop polls it before the
  next experiment and again before the expensive scoring phase, so a signal
  during analysis abandons the in-flight experiment without leaving a
  `candidates-exp-N/` directory behind, and either way `best.json` is still
  re-stamped with the run-summary tag before the process exits `0`. A second
  signal force-quits with exit code `130`. `RunResult` carries the
  `stopReason`, which the run summary prints as `stopped on:`.

- **The focus neuron's structure, statistics and blame are journalled (Issue
  #70).** Every experiment record now carries an optional `focusStats` object —
  the focus scan the loop already computed: squash and incoming-connection
  count, pre/post activation statistics (mean, variance, min/max, near-zero and
  saturation fractions, records scanned), output residuals and the backprop
  blame attached to the focus. `report` aggregates it into `all` / `accepted` /
  `rejected` buckets (mean incoming count, saturation and near-zero fractions,
  post-activation variance, mean |blame| and per-squash counts), and the run
  summary prints the same split, so experimental questions 4 (are
  saturated/dead neurons good targets?) and 6 (does propagated blame predict a
  successful direction?) survive the process exiting. Journals written before
  the field omit it and report `focusStats: null`.

- **Distribution shape in `observations.statistics`, with a consumer (Issue
  #73).** Every input and target column now carries population `skewness` and
  `excessKurtosis`, computed in the same streaming pass (the Welford accumulator
  gained third/fourth central moments); a constant column reports `0` rather
  than `NaN`. The new `stats_skew_bias` candidate strategy consumes them: for an
  output focus whose target has `|skewness| ≥ 0.25`, it steps the bias a quarter
  of the way from the target's mean towards its median — the hypothesis that a
  squared-error fit centres on the wrong statistic under a skewed target —
  damped by the target's excess kurtosis and skipped when the neuron is
  saturated. `ALGORITHM_VERSION` is `1.1.0`, so a pre-existing cache is rejected
  as stale and regenerated. Covariances are **not** stored: they are exactly
  `r · σ_a · σ_b` from fields already written, and the new
  `ObservationsStatistics::input_covariance` /
  `input_target_covariance` accessors derive them on demand instead of
  duplicating them on disk.

- **Effective seed and run configuration are journalled (Issue #71).** When
  `--seed` is omitted Lamarck now draws an explicit `u64` seed, logs it
  (`replay this run with --seed …`) and uses it for both the main and
  per-experiment backprop RNGs, so an unseeded run is replayable. Every run
  writes a one-off `runHeader` line to `experiments.jsonl` carrying the effective
  seed, its source (`supplied` / `drawn`), the Lamarck version and the run knobs
  (candidate count, minimum improvement, screen rate/threshold, focus policy and
  pinned neuron, stats mode and quick sample size, `structuralOnly`, grafts path,
  creature and training-data paths). Experiment records now carry the effective
  seed instead of `null`. `report` skips the header line.

### Fixed

- **`report` anchored the opening baseline on a 5% screen sample (Issue #84).**
  An experiment whose candidate batch screened empty journals the subsample
  baseline, not a full-corpus score. With `--skip-phase0` that sampled number
  became `openingBaselineScore`, so `totalScoreImprovement` subtracted two
  different quantities and could report a negative total for a run that only
  ever accepted improvements (the `batch-020` arm reported `-4.473e-04` against
  two accepts of `+1.322e-6` and `+1.724e-6`). `openingBaselineScore` is now the
  `scores.baseline` of the first experiment that actually promoted — the score
  Phase-0 measures when it runs, since the incumbent cannot change before the
  first acceptance — and it, `totalScoreImprovement` and
  `relativeScoreImprovement` are `null` until such a full-corpus score exists.

- **README read as a project plan rather than the built system (Issue #40).**
  Rewritten in the present tense against the code: status section, full CLI flag
  tables (including the previously undocumented `--preserve-losers`,
  `--screen-promote-threshold`, `--grafts-path` and
  `--graft-replay-budget-seconds`), Phase-G graft memory, the eight candidate
  strategies with their journal tags, combo scoring, the journal fields actually
  written, and the real outputs. Gaps between the old spec and the code are now
  an **Outstanding work** table pointing at Issues #39, #69, #70-#75. New
  `lamarck/tests/readme_contract.rs` fails the build if README and `--help`
  drift apart again.

- **README input list called the scorer optional (Issue #37).** The scorer
  binary is mandatory — only the `--scorer` path override has a default
  (`rust_scorer` on `PATH`). The **Inputs** section now separates required
  positionals, required-but-defaulted settings (scorer, output directory,
  candidates, timeout, minimum improvement) and genuinely optional ones (seed,
  mutation-strategy flags), with each flag name and default matching `--help`.

- **Combo dampen / graft follow-ups after #63.** Scale by how many combo members
  contribute new synapses to a target (not raw new-edge count), so a multi-edge
  single candidate is not self-dampened when merged with an unrelated improver.
  Combo accepts record each structural member's solo-sized graft weights, not
  the dampened merge (`0.1.2`).

### Changed

- **README core-principle diagram is a Mermaid flowchart (Issue #86).** The
  mutate → screen → promote loop was a plain ASCII `text` block; it now renders
  as a colour-coded Mermaid flowchart (candidate sources, scoring stages and
  outcomes each get their own `classDef`), matching the README's other
  diagrams. Same steps and branches — nothing was dropped in the conversion.

- **Run-level `lamarck` check-in tag (Issue #35).** The creature tag used by GRQ
  check-in now summarises the whole run (`N accepts / M exps`, last strategy /
  focus, and `score: <%.6g> improved by <%.3g>` vs opening) instead of only the
  last accept’s micro-step. Score appears once at 6 significant figures so GRQ
  can take the tag as the commit subject without appending another score clause.
  Final `best.json` is restamped with the full experiment count at loop exit.

### Added

- **CI auto-increments `lamarck/Cargo.toml` patch on src PRs.** Matches the
  GRQ-taxation / `runlib.sh` contract so remote hosts rebuild when the crate
  version changes without relying on a manual bump every time.
- **Smart combo dampening (Issue #63).** After screened singles improve, combo
  merges (experiments + Phase-G grafts) scale newly stacked synapses into the
  same target by `k.powf(-STACK_DAMPEN_EXPONENT)` (default `1/√k`). Journal and
  run-summary fields (`combosScored`, `combosDampened`, `comboDampen`, …) expose
  enough signal to retune the exponent after production runs. Bump crate to
  `0.1.1` so remote `runlib`-style installs rebuild.
- **PR auto-format syncs `Cargo.lock` to latest neat-core (Issue #33).** New
  auto-format job runs `cargo fmt --all` and `cargo update -p neat-core` on
  every PR to `Develop`, committing lock drift against the checked-out
  NEAT-AI-core path dependency so workers stop reprinting
  `Updating neat-core vX -> vY` after `model_fetch`. Does not auto-bump
  `neat-core.expected-version`.
- Repository quality gates aligned with NEAT-AI-scorer (CI, gitleaks, deny,
  codespell, markdownlint, SBOM, and related workflows).
- Locked experiment contracts for score-based acceptance against GRQ-scale
  creatures.
- Algorithm spine modules: backprop learning signals, `observations.statistics`
  JSON cache, focus-neuron tracing, candidate generation, scorer integration
  (`score` + `1e-6` threshold, no `--gpu`), 45-minute optimisation loop with
  `experiments.jsonl`, and journal economics reporting.
- `--quick` mode writes/reuses `observations-quick.statistics` (default sample
  25k records) plus coloured stderr progress for interactive runs.
- Coloured end-of-run summary (experiments, score delta, analysis/scorer time,
  paths).
- `--focus-neuron <uuid>` locks experiments to a chosen non-input neuron (e.g.
  `output-0`); `--quick` also caps focus-stat scans to the sample size.
- Output focus scans measure mean/`mae`/squash-aware adjusted error; a
  `mean_error_bias` candidate applies `bias += mean((target-post)*deriv)`.
- Real focus `LearningSignal` accumulation wired into the loop; backprop
  candidates propose from accumulated bias/weight signals.
- Incoming-source stats, structural add/weaken candidates, focus policies,
  Phase-0 scorer gate, scorer-failure abort/logging, lean observations
  (correlations opt-in), and richer journal economics reporting.
