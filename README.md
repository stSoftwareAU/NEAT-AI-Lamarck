# NEAT-AI-Lamarck

> Experimental: teaching evolved NEAT-AI creatures that what they learn in life can be inherited. Adventurous mutations, sceptical scorer — Lamarck would be proud.

NEAT-AI-Lamarck is an experimental Rust optimiser for already-fit [NEAT-AI](https://github.com/stSoftwareAU/NEAT-AI) creatures.

It does **not** replace normal NEAT evolution. It takes the current fittest
creature, studies how that creature behaves across the training data, generates
small statistically informed / backpropagation-informed / exploratory variants,
and asks the existing [NEAT-AI-scorer](https://github.com/stSoftwareAU/NEAT-AI-scorer)
to decide whether any candidate is genuinely fitter.

The experiment is intentionally conservative: candidate generation may be
adventurous, but acceptance is not.

## Status

The optimiser described below is **built and running** against production
GRQ-scale creatures. The whole spine exists — Phase-0 parity gate, observations
cache, focus selection, backprop learning signals, eight candidate strategies,
two-phase screen/promote scoring, candidate combos, structural graft memory,
the experiment journal and the `report` subcommand.

Measured economics from a 45-minute production run live in
[`docs/baseline-economics.md`](docs/baseline-economics.md).

This document describes what the code does today. Known gaps are listed under
[Outstanding work](#outstanding-work), each with an issue.

## Core principle

```text
current fittest creature
        |
        v
analyse one selected neuron
        |
        +----------------+----------------+----------------+
        |                |                |                |
        v                v                v                v
 statistical         backprop        structural         random
 candidates          candidates      candidates         candidates
        |                |                |                |
        +----------------+--------+-------+----------------+
                                  |
                                  v
                         candidate population
                                  |
                                  v
                     NEAT-AI-scorer batch scoring
                                  |
                         improvement >= threshold?
                           /                 \
                         yes                 no
                          |                   |
                          v                   v
                   new incumbent        keep incumbent
                          \                   /
                           +--------+----------+
                                    |
                                    v
                                   repeat
```

Only the standard scorer may declare a winner.

## Why "Lamarck"?

The name is deliberately playful rather than biologically literal. The
experiment starts with an evolved creature, lets it "experience" its training
environment, and converts useful acquired information into heritable changes to
the creature.

## Related repositories

- [NEAT-AI](https://github.com/stSoftwareAU/NEAT-AI) — TypeScript evolutionary trainer and current backpropagation implementation.
- [NEAT-AI-core](https://github.com/stSoftwareAU/NEAT-AI-core) — shared Rust creature/network implementation used by this project.
- [NEAT-AI-scorer](https://github.com/stSoftwareAU/NEAT-AI-scorer) — authoritative Rust scorer. Its directory/batch scoring path evaluates Lamarck candidate populations.

This repository follows the Rust workspace/tooling conventions of
NEAT-AI-scorer where practical.

## Scope

Lamarck answers one practical question:

> Can information gathered from the training observations, the current
> creature's internal behaviour, and conventional backpropagation produce useful
> mutations faster than ordinary evolutionary search alone?

A successful mutation may come from statistics, backpropagation, a structural
hypothesis, or dumb luck. Lamarck only cares that the authoritative score
improves, and records enough to tell which strategies actually earn their cost.

It is **not**:

- a replacement for the normal NEAT evolutionary process;
- a wholesale rewrite of NEAT-AI training;
- an optimiser allowed to accept predicted improvements without full scoring;
- an online/live trading optimiser;
- an attempt to modify many unrelated areas of a creature at once.

## Runtime model

Lamarck runs alongside the normal evolutionary system on other machines, so the
supplied creature is **perishable**: while Lamarck works, evolution may discover
a new global champion elsewhere. The default wall-clock budget is therefore
**45 minutes** (`--timeout-seconds 2700`), and cheap repeatable experiments are
preferred over one expensive analysis that eats the window.

### Production scale target

The production creature is the GRQ champion (`../GRQ-cluster/network.json`):
about `2511` inputs, `1` output, `~1590` hidden neurons, `~21k` synapses,
`forwardOnly: true`. Streaming statistics, the 45-minute budget and cheap
candidate proposals all exist to stay viable at that scale.

## Usage

```bash
neat_ai_lamarck <creature.json> <training-data-dir> [OPTIONS]
neat_ai_lamarck report <experiments.jsonl>
```

A production-shaped invocation:

```bash
neat_ai_lamarck \
  ../GRQ-cluster/network.json \
  .lamarck/train-data \
  --scorer ../NEAT-AI-scorer/target/release/rust_scorer \
  --output-dir .lamarck \
  --timeout-seconds 2700 --candidates 100 --seed 1 \
  --focus-policy weighted --screen-sample-rate 0.05 \
  --grafts-path ~/.lamarck/grafts.json
```

### Required positional arguments

The run cannot start without them:

- current fittest creature JSON;
- training-data directory.

### Required, but defaulted

The run always uses these; the flag only overrides the value.

| Flag | Default | Purpose |
|------|---------|---------|
| `--scorer` | `rust_scorer` on `PATH` | NEAT-AI-scorer binary. Scoring is **mandatory** — safety invariant 3 lets only the scorer declare a candidate fitter, so a run that cannot spawn the binary aborts, at the Phase-0 gate or after 3 consecutive scorer failures when `--skip-phase0` is passed. |
| `--output-dir` | `.` | Holds `best.json`, `experiments.jsonl`, `winners/` and per-experiment working directories. |
| `--candidates` | `100` | Candidates generated per experiment. |
| `--timeout-seconds` | `2700` | Wall-clock budget (45 minutes). |
| `--min-improvement` | `1e-6` | Absolute score delta required to accept, strict `>`. |
| `--screen-sample-rate` | `0.05` | Scorer subsample for the screen phase. `1` (or `>= 1`) disables screening. |
| `--screen-promote-threshold` | `1e-6` | Minimum sample-score Δ before a candidate earns a full-corpus score. |
| `--focus-policy` | `weighted` | `weighted` \| `high-error` \| `random` \| `unsaturated`. |
| `--quick-sample-records` | `25000` | Record cap for `--quick` observations / focus / learning scans. |

### Genuinely optional

Leaving these unset changes behaviour.

| Flag | Effect when set |
|------|-----------------|
| `--seed` | Deterministic RNG seed. Unset means the run cannot be replayed (see [#71](https://github.com/stSoftwareAU/NEAT-AI-Lamarck/issues/71)). |
| `--focus-neuron` | Pin every experiment to one neuron UUID (debug / smoke); overrides `--focus-policy`. |
| `--structural-only` | Generate only synapse/neuron growth candidates. |
| `--quick` | Use the sampled `observations-quick.statistics` cache and cap focus/learning scans. Acceptance still uses the full corpus. |
| `--compute-correlations` | Compute the expensive input×input correlation matrix in observations. |
| `--skip-phase0` | Skip the Phase-0 parity gate. |
| `--preserve-losers` | Keep rejected candidate, promote, combo and graft working directories instead of deleting them. |
| `--grafts-path` | JSON store for structural graft memory; enables Phase-G replay and recording. Keep it outside any wiped work directory. |
| `--graft-replay-budget-seconds` | Wall-clock budget for Phase-G replay. Default: 10% of `--timeout-seconds`. |

## How a run works

```mermaid
flowchart TD
    P0[Phase 0: scorer baseline + parity gate] --> OBS[Phase 1: observations.statistics cache]
    OBS --> G[Phase G: replay stored structural grafts]
    G --> LOOP{wall-clock budget left?}
    LOOP -- no --> OUT[best.json + experiments.jsonl + winners/]
    LOOP -- yes --> LRN[accumulate creature learning signal]
    LRN --> F[Phase 2: select focus neuron]
    F --> AN[Phase 3: focus + incoming-source statistics]
    AN --> GEN[Phase 4: generate candidates]
    GEN --> SCR[Phase 5a: screen on scorer subsample]
    SCR --> PRO[Phase 5b: full-corpus score baseline + promoted]
    PRO --> CMB[Phase 5c: score combos of improving candidates]
    CMB --> ACC{score delta > min-improvement?}
    ACC -- yes --> NEW[new incumbent, stamp best.json + winners/]
    ACC -- no --> KEEP[keep incumbent]
    NEW --> J[append experiments.jsonl]
    KEEP --> J
    J --> LOOP
```

### Phase 0 — authoritative baseline and parity

Before optimisation starts Lamarck scores the supplied creature with the
scorer (finite `score`/`error` required), computes its own whole-creature mean
squared error over the same training directory through the compiled network, and
compares the overlapping quantities within documented epsilon
(`lamarck/src/parity.rs`):

- **error:** abs `1e-6` or rel `1e-5`;
- **unpenalized score** `1 - error` vs `scorer.score + complexityPenalty`:
  abs `1e-5` or rel `1e-4`.

Unexplained disagreement aborts the run, which stops Lamarck optimising a subtly
different metric. `--skip-phase0` disables the gate.

### Phase 1 — `observations.statistics`

The training-data directory holds a one-time statistics cache,
`observations.statistics`. If it is absent, Lamarck scans the complete corpus
and writes it. `--quick` builds/reuses a sampled
`observations-quick.statistics` from the first `--quick-sample-records` records
(default `25000`) so smoke runs finish in minutes rather than about an hour on
GRQ-scale data. Full caches remain the production default.

The file is human-readable **JSON** carrying semver format and algorithm
versions plus enough identity metadata to reject stale caches: input observation
count, output count, record count, a deterministic corpus identity, creation
timestamp, and the cache mode. A statistics file for different training data is
never silently reused.

For every raw input observation, and every target, it records: count; mean;
variance and standard deviation; minimum and maximum; zero, non-zero and
non-finite counts; mean absolute value; RMS; and approximate 1/5/25/50/75/95/99%
quantiles from a capped reservoir sample.

Relationships: observation/target Pearson correlations are always collected. The
expensive input×input correlation matrix is opt-in via `--compute-correlations`
(default off, after [#22](https://github.com/stSoftwareAU/NEAT-AI-Lamarck/issues/22)
removed fields nothing consumed).

### Phase G — structural graft memory

With `--grafts-path` set, Lamarck keeps a local JSON store of structural changes
that previously won: added synapses and added hidden-neuron bridges, each with
its attachment requirements and a helpful/harmful tally.

Before the experiment loop it classifies every stored graft against the opening
fittest (present / applicable / inapplicable), retires the inapplicable ones,
scores each applicable graft as a single, then scores dampened combinations of
the helpful ones, and applies the best accepted result. Persistently harmful
grafts are retired. New structural accepts are recorded back into the store —
for a combo accept, each structural member is recorded at its **solo** weights,
never the dampened merge, so replay cannot double-dampen.

### Phase 2 — select a focus neuron

Each iteration focuses on one non-input neuron. The default policy
(`--focus-policy weighted`) draws weighted-random by **error influence**: output
residual L1 mass, or hidden `|total blame|` decayed by distance to the nearest
output, so deep diluted neurons rarely win. Outputs are usually strongest but
are not chosen every time, and zero-signal neurons are never selected. The
selector also folds in each neuron's own accept history.

`high-error` always picks the single strongest neuron and sticks there — avoid
it in production. `random` and `unsaturated` remain available for A/B work, and
`--focus-neuron` pins a UUID for debug and smoke runs.

### Phase 3 — creature-specific analysis

Static observation statistics describe the dataset; hidden-neuron behaviour
depends on the incumbent and is measured against that creature. For the selected
neuron Lamarck collects streaming pre-activation and post-activation statistics
(mean, variance, standard deviation, min/max, near-zero and saturation
fractions) plus, for each incoming connection, source statistics and the
source's relationship with the neuron's learning/error signal. Raw observation
sources reuse `observations.statistics` where mathematically equivalent; hidden
sources are measured from the current creature.

### Backpropagation

Lamarck ports NEAT-AI backprop behaviour by wiring creatures through neat-core's
`propagate_topological_loop` (the TS/WASM reverse-topo contract), then folding
the result into an analyse-without-apply `LearningSignal`.

- Config / LR / limits / sparse ratio: `lamarck/src/backprop.rs`
- Creature → `PropagateInput` layout + sparse RNG: `lamarck/src/propagate_layout.rs`
- Apply (optional): `apply_learnings` clones the creature and writes proposed
  bias/weight updates — optimisation still accepts only via the scorer
- Defaults use fixed `generations: 1.0` and `sparse_ratio: 1.0` for
  deterministic full-network signals under a seeded RNG

A hidden neuron has no natural target — blame comes from the propagated learning
signal, never an invented `expected_hidden - actual_hidden`.

Parity fixtures live under `lamarck/tests/fixtures/backprop/` (tolerances about
`1e-9`–`1e-6`). Regenerate goldens:

```bash
LAMARCK_REGEN_BACKPROP_FIXTURES=1 cargo test -p neat_ai_lamarck --test backprop_parity
```

Optional Deno helper (sibling `../NEAT-AI`): `scripts/generate_backprop_parity_fixtures.ts`.

### Phase 4 — candidate generation

Every candidate is a descendant of the current incumbent carrying a small,
interpretable change. `--candidates` (default `100`) sets the population size,
sized to keep a ~10-core scorer box saturated. The generator front-loads
structural probes, then round-robins the remaining budget across the strategies
below; `--structural-only` restricts it to growth candidates.

| Journal tag | What it proposes |
|-------------|------------------|
| `backprop` | Bias, or the strongest incoming weight, stepped by the accumulated learning signal (absolute weight delta capped at `0.01`). |
| `mean_error_bias` | Output-focus `bias += mean((target - post) * derivative)`, damped to a tenth and skipped when the neuron is saturated. |
| `stats_weight` | Weight nudge on the best incoming source, direction from its correlation with the error, magnitude scaled by the source's standard deviation. |
| `stats_bias` | Bias step scaled by the measured pre-activation spread, pushed away from saturation. |
| `structural_add` | New upstream synapse from a residual-ranked unused source, weight from a fraction of the residual OLS coefficient. |
| `structural_add_neuron` | New hidden neuron bridging top residual sources into the focus, squash chosen from the residual sign and observation scale; falls back to splitting the strongest incoming synapse. |
| `structural_weaken` | Halves the weight of the smallest-magnitude incoming synapse. |
| `random` | Random bias or weight delta within ±0.05. |

There is no fixed quota for random controls — random accidents are valid
improvements and are accepted if they win. Candidate changes use measured source
scale and neuron behaviour rather than arbitrary absolute deltas, and each
strategy produces alternatives around its preferred direction instead of trusting
a single estimated optimum.

Every candidate records the strategy, a full-precision description of the exact
mutation, and the old/new value for scalar changes.

### Phase 5 — authoritative candidate scoring

The incumbent plus all candidates are written to a temporary directory:

```text
candidates-exp-7/
    baseline.json
    candidate-000.json
    candidate-001.json
    ...
```

Scoring runs in three steps:

1. **Screen** the full candidate directory with
   `rust_scorer --sample-rate 0.05 --sample-phase <n> …` (≈0.7–1s/creature on
   GRQ against ≈11s full). The sample phase rotates per experiment.
2. **Promote** only stems with sample Δ `> --screen-promote-threshold` into a
   full-corpus batch, so full-corpus time is not spent on sample noise. An empty
   screen ends the experiment without a full-corpus call.
3. **Combine** — when two or more promoted candidates each beat the baseline,
   their mutation deltas are merged into combination creatures (all pairs, then
   triples, budget-capped) and scored in one further full-corpus batch.
   Synapses newly stacked into the same target by `k` members are scaled by
   `k^-0.5` so a merge is not louder than the sum of its evidence. Conflicting
   edits to the same neuron or edge are skipped.

Scorer argv is the locked two-argument form plus the screen sampling flags only;
Lamarck never passes `--gpu` or `--cost`, so scorer defaults decide backend and
loss. Pass `--screen-sample-rate 1` to disable screening.

The incumbent is included in every scored batch, so a candidate is never
compared against a stale score. Acceptance uses the scorer JSON **`score`**
field (**larger-is-better**) from the **full-corpus** score only — never `error`
alone:

```text
candidate.score - baseline.score > 1e-6
```

(default absolute threshold, strict greater-than). GRQ `costOfGrowth` is `1e-7`,
so `1e-6` sits deliberately above growth noise.

An accepted winner becomes the incumbent immediately: `best.json` is rewritten
with the creature's `uuid`/`tags` re-attached and a run-summary `lamarck` tag,
a copy is kept under `winners/`, structural accepts are recorded into the graft
store, and creature-specific analysis is recomputed next iteration.

### Phase 6 — repeat until the budget expires

The loop continues selecting neurons and testing candidates until the
wall-clock timeout expires, or three consecutive scorer batches fail. A failed
experiment simply moves on to another attempt. A configured maximum experiment
count and explicit cancellation are **not** implemented — see
[#72](https://github.com/stSoftwareAU/NEAT-AI-Lamarck/issues/72).

The optimisation path is cumulative:

```text
C0 -> C1 -> C2 -> C3 -> ...
```

Every edge is an independently full-corpus-scored improvement.

## Outputs

Written under `--output-dir`:

```text
best.json           # best verified creature, tagged with the run summary
experiments.jsonl   # machine-readable experiment journal
winners/            # winner-NNNN.json per accepted improvement
```

Per-experiment working directories (`candidates-exp-N/`, `promote-exp-N/`,
`combos-exp-N/`, `phase0-baseline/`, `graft-replay/`) are deleted as the run
proceeds unless `--preserve-losers` is passed. The supplied creature file is
never modified: `best.json` starts as a verbatim copy of it.

## Experiment journal

`experiments.jsonl` is one JSON object per experiment, and is part of the
experiment rather than debug logging. Each record carries:

| Field | Meaning |
|-------|---------|
| `experimentNumber`, `timestampUnix` | Sequence and wall-clock position. |
| `seed` | Run seed, when `--seed` was supplied. |
| `incumbentId` | Incumbent shape identity (`in…-out…-n…-s…`). |
| `baselineScore` | Authoritative baseline for this experiment. |
| `focusNeuron` | Selected neuron UUID. |
| `candidates[]` | Per candidate: `strategy`, `focusNeuron`, `mutation`, `oldValue`, `newValue`. |
| `screenScores`, `scores` | Sample-phase and full-corpus scores by stem. |
| `winner`, `improvement`, `accepted` | Outcome of the experiment. |
| `analysisMs`, `scorerMs` | Where the time went. |
| `scorerError` | Present when the batch failed. |
| `comboMembers`, `combosScored`, `combosDampened`, `comboDampen` | Combination-scoring detail. |

Summarise strategy economics from a journal with the `report` subcommand:

```bash
neat_ai_lamarck report experiments.jsonl
# or: scripts/report-experiments.sh experiments.jsonl
```

It emits per-strategy appearances/wins/acceptance rate, focus history,
improvement series, candidates per scorer-minute and per screen-minute,
analysis-time fraction, projected batches per 45 minutes, and combo totals.

## Safety invariants

These rules are non-negotiable:

1. The supplied fittest creature is never modified in place.
2. Lamarck's custom analysis cannot declare a creature fitter.
3. Every accepted candidate must be scored against the normal complete training dataset by NEAT-AI-scorer.
4. The incumbent is included in each candidate scoring batch as a control.
5. A candidate must improve by more than the configured meaningful-improvement threshold.
6. A failed experiment leaves the incumbent unchanged.
7. Creature topology/serialization must remain compatible with NEAT-AI-core and NEAT-AI-scorer.
8. The original creature is always recoverable.

## Reproducibility

All Lamarck-controlled randomness comes from the `--seed` value: the main RNG is
seeded from it, and the per-experiment backprop RNG is derived from it
deterministically. Given an identical starting creature, training data,
`observations.statistics`, configuration, software version and seed, candidate
generation repeats.

Two caveats apply today. Without `--seed` the RNG is drawn from OS entropy and
the drawn value is not recorded, so the run cannot be replayed; and because the
experiment count is wall-clock bounded, a differently timed rerun may not
produce the same number of experiments. Both are tracked in
[#71](https://github.com/stSoftwareAU/NEAT-AI-Lamarck/issues/71).

## What we have learnt so far

From the 45-minute production-budget baseline in
[`docs/baseline-economics.md`](docs/baseline-economics.md) (single seed, GRQ
champion, 75 experiments):

- accepts are thin — 2 in 75 experiments, both from `random`, cumulative
  Δ ≈ `+2.11e-6`;
- scorer time dominates at ≈83% of the run, so full focused analysis before each
  batch is affordable;
- screening empties most batches (49/75), which is the point: expensive
  full-corpus promotes are skipped when the sample shows nothing;
- no strategy has earned removal — the sample is far too small to disable one.

The open experimental questions the journal is designed to answer:

1. Do statistically informed candidates beat ordinary random mutation often enough to justify their analysis cost?
2. How useful is conventional backpropagation when its proposed changes must survive whole-corpus evolutionary scoring?
3. Which mutation classes produce accepted improvements most often?
4. Are saturated/dead neurons particularly good targets?
5. Are observation correlations useful when adding or removing connections?
6. Does the propagated neuron blame/sensitivity predict successful mutation direction?
7. As the incumbent improves, how quickly does the hit rate fall?
8. Given the 45-minute useful-life constraint, how much analysis is economically justified before trying another candidate batch?

Questions 1–3 and 8 have a first single-seed answer; the follow-up runs needed
to answer the rest are tracked in
[#75](https://github.com/stSoftwareAU/NEAT-AI-Lamarck/issues/75).

## Outstanding work

| Issue | Gap |
|-------|-----|
| [#39](https://github.com/stSoftwareAU/NEAT-AI-Lamarck/issues/39) | The core-principle diagram predates two-phase screening. |
| [#69](https://github.com/stSoftwareAU/NEAT-AI-Lamarck/issues/69) | Unsuccessful candidates are re-scored across experiments instead of being remembered. |
| [#70](https://github.com/stSoftwareAU/NEAT-AI-Lamarck/issues/70) | The journal omits the focus neuron's squash, incoming count, statistics and blame. |
| [#71](https://github.com/stSoftwareAU/NEAT-AI-Lamarck/issues/71) | A run without `--seed` cannot be replayed, and the run configuration is not journalled. |
| [#72](https://github.com/stSoftwareAU/NEAT-AI-Lamarck/issues/72) | No maximum-experiment-count stopping rule and no graceful cancellation. |
| [#73](https://github.com/stSoftwareAU/NEAT-AI-Lamarck/issues/73) | `observations.statistics` has no skewness/kurtosis or covariances. |
| [#74](https://github.com/stSoftwareAU/NEAT-AI-Lamarck/issues/74) | `report` does not attribute combo or graft wins to a strategy. |
| [#75](https://github.com/stSoftwareAU/NEAT-AI-Lamarck/issues/75) | The follow-up economics experiments recommended by the #8 baseline are unrun. |

## Repository layout

```text
NEAT-AI-Lamarck/
├── Cargo.toml
├── rust-toolchain.toml
├── neat-core.expected-version
├── quality.sh
├── deny.toml
├── SECURITY.md
├── .github/workflows/   # scorer-aligned quality gates
├── scripts/
├── docs/
│   ├── architecture.md
│   └── baseline-economics.md
└── lamarck/src/
    ├── lib.rs
    ├── main.rs              # CLI (optimise + report subcommand)
    ├── config.rs            # defaults and run options
    ├── parity.rs            # Phase-0 scorer parity gate
    ├── observations.rs      # observations.statistics cache
    ├── focus.rs             # focus-neuron selection policies
    ├── backprop.rs
    ├── learning.rs
    ├── propagate_layout.rs
    ├── candidates.rs
    ├── structural.rs        # graph mutation primitives + residual ranking
    ├── combos.rs            # candidate merging and stacked-synapse dampening
    ├── grafts.rs            # structural graft store and phase-G replay
    ├── scorer.rs
    ├── run.rs
    ├── report.rs
    ├── tags.rs
    └── log.rs
```

## Development rules

- Rust edition 2024.
- Pin the Rust toolchain, matching NEAT-AI-scorer initially (`1.95.0`).
- Use TDD for behaviour changes.
- Keep dependencies modest and justified.
- Prefer streaming training-data analysis; the corpus is large.
- Preserve compatibility with NEAT-AI-core/NEAT-AI-scorer rather than duplicating their stable functionality.

## Build and quality gate

Clone **NEAT-AI-core** beside this repository:

```text
parent/
  NEAT-AI-core/
  NEAT-AI-Lamarck/
```

Local gate (mirrors CI):

```bash
./quality.sh < /dev/null
```

Requires **shellcheck**, **cargo-deny** (`cargo install cargo-deny --locked`),
and **codespell** (`pip install --user codespell`).

CI runs on pull requests to `Develop` and includes fmt/clippy/tests/docs,
cargo-deny, gitleaks, cargo-audit, dependency-review, Semgrep, markdownlint,
actionlint, SBOM, shellcheck, and codespell. Branch protection should require
the aggregator check **CI Required Checks**.

PRs also run an auto-format / housekeeping job
(`.github/workflows/auto-format.yml`, Issue #33). The job runs
`cargo fmt --all` and then `cargo update -p neat-core` so `Cargo.lock`
tracks the checked-out NEAT-AI-core path dependency (workers otherwise
rewrite the lock on every `cargo build` and `model_fetch` resets it). If the
working tree changes, the fix is committed and pushed back. The job
deliberately does **not** bump `neat-core.expected-version` — the
breaking-bump gate stays a human acknowledgement. The workflow is validated
by `scripts/check-auto-format-workflow.sh` (invoked from `quality.sh`).

A second PR job (`.github/workflows/version-increment.yml`) bumps the
`lamarck/Cargo.toml` patch version on source changes so remote `runlib`-style
installs rebuild; `scripts/check-version-increment-workflow.sh` validates it
from `quality.sh`.

### neat-core breaking-bump gate

The `neat-core` path dependency is unpinned. CI fails when the sibling
neat-core presents a breaking SemVer bump above
[`neat-core.expected-version`](./neat-core.expected-version). Clear the gate by
updating Lamarck for the change and bumping that baseline in the same PR.
