# Changelog

All notable changes to NEAT-AI-Lamarck are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Fixed

- **Combo dampen / graft follow-ups after #63.** Scale by how many combo members
  contribute new synapses to a target (not raw new-edge count), so a multi-edge
  single candidate is not self-dampened when merged with an unrelated improver.
  Combo accepts record each structural member's solo-sized graft weights, not
  the dampened merge (`0.1.2`).

### Changed

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
