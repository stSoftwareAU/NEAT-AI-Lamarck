# NEAT-AI-Lamarck

> Experimental: teaching evolved NEAT-AI creatures that what they learn in life can be inherited. Adventurous mutations, sceptical scorer — Lamarck would be proud.

NEAT-AI-Lamarck is an experimental Rust optimiser for already-fit [NEAT-AI](https://github.com/stSoftwareAU/NEAT-AI) creatures.

It does **not** replace normal NEAT evolution. Instead, it takes the current fittest creature, studies how that creature behaves across the training data, generates small statistically informed / backpropagation-informed / exploratory variants, and asks the existing [NEAT-AI-scorer](https://github.com/stSoftwareAU/NEAT-AI-scorer) to decide whether any candidate is genuinely fitter.

The experiment is intentionally conservative: candidate generation may be adventurous, but acceptance is not.

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

The name is deliberately playful rather than biologically literal. The experiment starts with an evolved creature, lets it "experience" its training environment, and attempts to convert useful acquired information into heritable changes to the creature.

## Related repositories

- [NEAT-AI](https://github.com/stSoftwareAU/NEAT-AI) — TypeScript evolutionary trainer and current backpropagation implementation.
- [NEAT-AI-core](https://github.com/stSoftwareAU/NEAT-AI-core) — shared Rust creature/network implementation used by this project.
- [NEAT-AI-scorer](https://github.com/stSoftwareAU/NEAT-AI-scorer) — authoritative Rust scorer. Its directory/batch scoring path is used to evaluate Lamarck candidate populations.

This repository follows the Rust workspace/tooling conventions of NEAT-AI-scorer where practical.

## Goals

Version 1 should answer one practical question:

> Can information gathered from the training observations, the current creature's internal behaviour, and conventional backpropagation produce useful mutations faster than ordinary evolutionary search alone?

A successful mutation may come from statistics, backpropagation, a structural hypothesis, or dumb luck. Lamarck only cares that the authoritative score improves.

Secondary goals are to record enough information to learn which candidate-generation strategies are actually useful.

## Non-goals

Version 1 is not:

- a replacement for the normal NEAT evolutionary process;
- a wholesale rewrite of NEAT-AI training;
- an optimiser allowed to accept predicted improvements without full scoring;
- an online/live trading optimiser;
- an attempt to modify many unrelated areas of a creature at once.

## Runtime model

Lamarck is expected to run alongside the normal evolutionary system on other machines.

The supplied creature is therefore **perishable**: while Lamarck is working, evolution may discover a new global champion elsewhere.

The default wall-clock runtime is:

```text
45 minutes
```

The timeout must be configurable.

This constraint should influence implementation choices. A theoretically better analysis that consumes most of the 45-minute window may be less useful than several cheaper attempts.

## Inputs

**Required positional arguments** (no default — the run cannot start without them):

- current fittest creature JSON;
- training-data directory.

**Required, but defaulted** — the run always uses these; the flag only overrides
the value:

- a working NEAT-AI-scorer binary — `--scorer`, default `rust_scorer` resolved
  on `PATH`. Scoring is **mandatory**, not optional: safety invariant 3 lets
  only the scorer declare a candidate fitter, so a run that cannot spawn the
  binary aborts — at the Phase-0 gate, or after
  `DEFAULT_MAX_CONSECUTIVE_SCORER_FAILURES` (`3`) consecutive scorer failures
  when `--skip-phase0` is passed. The flag is optional only in the sense that
  the default path is used when it is omitted.
- output directory — `--output-dir`, default `.` (holds `best.json` and
  `experiments.jsonl`);
- candidate count — `--candidates`, default `100`;
- timeout — `--timeout-seconds`, default `2700` (45 minutes);
- minimum meaningful improvement — `--min-improvement`, default `1e-6`
  (absolute score delta, strict `>`).

**Genuinely optional** (unset changes behaviour):

- deterministic random seed — `--seed`; unset means the run is not reproducible;
- mutation-strategy configuration — e.g. `--structural-only`, `--focus-policy`,
  `--focus-neuron`, `--screen-sample-rate`.

### Production scale target

The intended production creature is the GRQ champion
(`../GRQ-cluster/network.json`): about `2511` inputs, `1` output,
`~1590` hidden neurons, `~21k` synapses, `forwardOnly: true`. Design choices
(streaming stats, 45-minute budget, cheap candidate proposals) should remain
viable at that scale.

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

## Phase 0 — authoritative baseline and parity

Before optimisation starts:

1. Score the supplied creature using NEAT-AI-scorer (finite `score` / `error` required).
2. Compute Lamarck's whole-creature mean squared error over the same training directory
   via the compiled network (same activation path used for focus analysis).
3. Compare overlapping quantities within documented epsilon (see `lamarck/src/parity.rs`):
   - **error:** abs `1e-6` or rel `1e-5`
   - **unpenalized score** `1 - error` vs `scorer.score + complexityPenalty`:
     abs `1e-5` or rel `1e-4`
4. Abort optimisation on unexplained disagreement (`--skip-phase0` disables the gate).

This prevents Lamarck from accidentally optimising a subtly different metric.

## Phase 1 — `observations.statistics`

The training-data directory contains, or will contain, a one-time statistics cache:

```text
observations.statistics
```

If it is absent, Lamarck scans the complete training corpus and creates it.

For smoke tests and short runs, pass `--quick` to build/reuse a sampled cache
instead:

```text
observations-quick.statistics
```

Quick mode scans only the first `--quick-sample-records` records (default
`25000`) so generation finishes in minutes rather than ~an hour on GRQ-scale
data. Full-mode caches remain the production default.

Because full generation is expected to be a one-time operation for a training
dataset, it should collect more than the first mutation strategy strictly
requires.

### Format

`observations.statistics` is human-readable **JSON** with **semver** format and
algorithm versions. Trust a matching file; regenerate or refuse on unsupported
version or corpus-identity mismatch. Prefer JSON for debuggability at GRQ
observation counts (~2511 inputs); compact binary only if profiling proves JSON
I/O dominates.

### Dataset identity

The file must contain enough metadata to reject stale statistics, including at least:

- format version (semver);
- statistics algorithm version (semver);
- input observation count;
- output count;
- record count;
- deterministic identity/checksum of the training corpus;
- creation timestamp.

A statistics file for different training data must never be silently reused.

### Per-observation statistics

For each raw input observation collect at least:

- count;
- mean;
- variance / standard deviation;
- minimum / maximum;
- zero and non-zero counts;
- non-finite count;
- quantiles: 1%, 5%, 25%, 50%, 75%, 95%, 99%;
- mean absolute value;
- RMS;
- skewness and kurtosis where practical.

### Relationships

For version 1 also collect, where memory/runtime permits:

- observation/observation covariance;
- Pearson correlation;
- observation/target covariance;
- observation/target correlation.

For the current observation count, a full symmetric correlation matrix is acceptable. The file format should be versioned so this can change later.

The expensive input×input covariance/correlation matrices are opt-in via
`--compute-correlations` (default off); observation/target relationships are
always collected.

## Phase 2 — select a focus neuron

Each optimisation iteration focuses on one non-input neuron.

Default policy (`--focus-policy weighted`, issue #25) draws weighted-random by
**error influence**: output residual L1 mass, or hidden `|total blame|` decayed by
distance to the nearest output (deep/diluted neurons rarely win). Outputs are
usually strongest but not chosen every time. Zero-signal neurons are never
selected. Avoid `high-error` in production — it sticks on one neuron.
`random` / `unsaturated` remain available. `--focus-neuron` pins a UUID for
debug/smoke.

## Phase 3 — creature-specific analysis

Static observation statistics describe the dataset. Hidden-neuron statistics depend on the current incumbent and must be measured against that creature.

For the selected neuron collect streaming statistics for at least:

### Pre-activation

- mean / variance / standard deviation;
- min / max;
- useful quantiles where practical.

### Post-activation

- mean / variance / standard deviation;
- min / max;
- near-zero fraction where relevant;
- activation saturation fraction where relevant.

### Incoming sources

For each incoming connection gather useful source statistics and relationships with the selected neuron's learning/error signal.

Raw observation sources may reuse `observations.statistics` where mathematically equivalent. Hidden sources must be measured from the current creature.

## Backpropagation

Lamarck ports NEAT-AI backprop behaviour by wiring creatures through neat-core’s
`propagate_topological_loop` (the TS/WASM reverse-topo contract), then folding
results into an analyse-without-apply [`LearningSignal`].

- Config / LR / limits / sparse ratio: `lamarck/src/backprop.rs`
- Creature → `PropagateInput` layout + sparse RNG: `lamarck/src/propagate_layout.rs`
- Apply (optional): `apply_learnings` clones the creature and writes proposed
  bias/weight updates (optimisation still accepts only via the scorer)
- Defaults use fixed `generations: 1.0` and `sparse_ratio: 1.0` for deterministic
  full-network signals under a seeded RNG

Parity fixtures live under `lamarck/tests/fixtures/backprop/` (tolerances about
`1e-9`–`1e-6`). Regenerate goldens:

```bash
LAMARCK_REGEN_BACKPROP_FIXTURES=1 cargo test -p neat_ai_lamarck --test backprop_parity
```

Optional Deno helper (sibling `../NEAT-AI`): `scripts/generate_backprop_parity_fixtures.ts`.

A hidden neuron has no natural target — blame comes from the propagated learning
signal, never an invented `expected_hidden - actual_hidden`.

## Phase 4 — candidate generation

Default candidate population size:

```text
100
```

This is configurable.

Candidates are descendants of the current incumbent. Version 1 should generally prefer small, interpretable changes and avoid changing many unrelated neurons in one candidate.

Candidate generators may include:

- conventional backprop-derived weight/bias changes;
- statistics-guided incoming-weight changes;
- statistics-guided bias changes;
- adding a plausible upstream connection;
- weakening/removing an apparently useless connection;
- random/exploratory mutations.

There is no fixed quota for random controls. Random accidents are valid improvements and should be accepted if they win.

Every candidate records the strategy and exact mutation that produced it so later analysis can compare approaches.

### Statistical mutation guidance

Candidate changes should use measured source scale and neuron behaviour rather than arbitrary absolute deltas where possible.

For example, for a source activation `x` and weight change `Δw`, estimate the induced pre-activation change from `Δw * x` and generate several conservative changes around a preferred direction/magnitude.

Do not trust a single estimated optimum. Produce alternatives around it and allow opposite-direction/exploratory candidates.

Bias proposals should similarly consider the selected neuron's measured pre-activation distribution and squash saturation.

## Phase 5 — authoritative candidate scoring

Write the incumbent plus all candidate creatures to a temporary directory, for example:

```text
candidates/
    baseline.json
    candidate-000.json
    candidate-001.json
    ...
```

Default production path (issue #24):

1. **Screen** the full candidate directory with
   `rust_scorer --sample-rate 0.05 …` (≈0.7–1s/creature on GRQ vs ≈11s full).
   Default batch size is **100** candidates (saturate a ~10-core GRQ box for
   improvement/hour).
2. **Promote** only stems with sample Δ `> 1e-6` (same bar as acceptance) so
   full-corpus time is not spent on sample noise.
3. **Full-corpus** score baseline + promoted creatures (two-arg scorer form).

Do **not** pass `--gpu` or `--cost`; scorer defaults decide backend and loss.
Pass `--screen-sample-rate 1` to disable screening.

The incumbent is included in every scored batch to avoid comparison against a stale score.

Acceptance uses the scorer JSON **`score`** field (**larger-is-better**) from the
**full-corpus** promote (or single) score only. Never accept on `error` alone.
A candidate is accepted only when:

```text
candidate.score - baseline.score > 1e-6
```

(default absolute threshold; strict greater-than). GRQ `costOfGrowth` is
`1e-7`; `1e-6` is deliberately above growth noise.

After acceptance, the winner becomes the new incumbent immediately. Creature-specific analysis from the old incumbent is then considered stale and is recomputed as required.

## Phase 6 — repeat until budget expires

Continue selecting neurons and testing candidates until:

- the default/configured wall-clock timeout expires;
- a configured maximum experiment count is reached;
- explicit cancellation occurs;
- another explicit stopping rule fires.

A failed neuron experiment simply moves on to another attempt.

The optimisation path is cumulative:

```text
C0 -> C1 -> C2 -> C3 -> ...
```

Every edge represents an independently full-corpus-scored improvement.

## Experiment journal

Write a machine-readable JSON Lines journal, proposed filename:

```text
experiments.jsonl
```

Record at least:

- experiment number;
- timestamp;
- random seed/state needed for reproduction;
- incumbent checksum/identifier;
- baseline authoritative score;
- selected neuron;
- selected neuron squash and incoming connection count;
- relevant neuron statistics;
- backprop/blame statistics;
- candidate strategy and exact mutation description;
- candidate scores;
- winning candidate, if any;
- absolute/relative improvement;
- accepted/rejected;
- analysis time;
- batch-scoring time.

This journal is part of the experiment, not merely debug logging.

Summarise strategy economics from a journal with the `report` subcommand
(`neat_ai_lamarck report experiments.jsonl`, or
`scripts/report-experiments.sh`).

## Outputs

At minimum:

```text
best.json
experiments.jsonl
```

Optionally preserve accepted intermediate creatures under:

```text
winners/
```

Failed candidates should normally be removed unless a debug/preserve flag is enabled.

## Reproducibility

All Lamarck-controlled randomness must come from a recordable seed.

Given identical:

- starting creature;
- training data;
- `observations.statistics`;
- configuration;
- software versions;
- random seed;

candidate generation should be reproducible.

## Experimental questions

The implementation should eventually let us answer:

1. Do statistically informed candidates beat ordinary random mutation often enough to justify their analysis cost?
2. How useful is conventional backpropagation when its proposed changes must survive whole-corpus evolutionary scoring?
3. Which mutation classes produce accepted improvements most often?
4. Are saturated/dead neurons particularly good targets?
5. Are observation correlations useful when adding or removing connections?
6. Does the propagated neuron blame/sensitivity predict successful mutation direction?
7. As the incumbent improves, how quickly does the hit rate fall?
8. Given the 45-minute useful-life constraint, how much analysis is economically justified before trying another candidate batch?

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
    ├── structural.rs
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

### neat-core breaking-bump gate

The `neat-core` path dependency is unpinned. CI fails when the sibling
neat-core presents a breaking SemVer bump above
[`neat-core.expected-version`](./neat-core.expected-version). Clear the gate by
updating Lamarck for the change and bumping that baseline in the same PR.
