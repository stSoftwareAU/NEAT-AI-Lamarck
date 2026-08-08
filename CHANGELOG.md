# Changelog

All notable changes to NEAT-AI-Lamarck are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

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
