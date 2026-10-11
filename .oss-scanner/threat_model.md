# Threat model

Disclosure channels and scope are in [SECURITY.md](../SECURITY.md). The run
lifecycle is described under
[How a run works](../README.md#how-a-run-works) and the locked contracts under
[Safety invariants](../README.md#safety-invariants) in the README. This file is
the short brief for the scanner.

## What this project does and where untrusted input enters

NEAT-AI-Lamarck is an experimental, offline Rust CLI (`neat_ai_lamarck`) that
studies how an already-fit NEAT-AI creature behaves across its training data,
generates small statistical, backpropagation-informed and structural variants,
and keeps one only when the external NEAT-AI-scorer binary (`rust_scorer`) says
it is fitter. It opens no network connections and holds no credentials. Treat
as untrusted:

- creature JSON (the positional creature argument), which in production is
  pulled from a repository shared by many machines;
- the training-data directory of `.bin` files, and the observations statistics
  caches (`observations.statistics`, `src/observations.rs`) written beside it;
- state carried between runs: the structural graft memory (`--grafts-path`,
  `src/grafts.rs`), strategy priors (`--strategy-priors-path`) and the
  failed-candidate cache (`--failed-cache`, `src/failed_cache/`);
- `experiments.jsonl` journals read by `report`, and the scorer's output.

The scorer path and output directory are chosen by the operator and are
trusted.

## Components that matter most / least

Most important: creature loading and validation (`src/validate.rs`), corpus
and observations-cache parsing, graft and failed-cache replay, the code that
rewrites the creature (`src/candidates.rs`, `src/structural.rs`,
`src/backprop.rs`, `src/memetic.rs`) and every file name built from input data
under the output directory. Lower priority: `src/report.rs`, the economics and
calibration summaries, the benches under `lamarck/examples/` and the
`scripts/` research drivers.

## How to exercise it

From `/src`: `cargo test --workspace --all-features` runs the suite;
`lamarck/tests/common/` and `lamarck/tests/fixtures/` hold small creatures and
corpora, and several tests drive the built binary end to end.
`target/debug/neat_ai_lamarck --help` lists the options. `rust_scorer` is on
`PATH` when its optional install succeeded.

## How you rate severity

This is a research tool run by its own maintainers on their own hosts. The
crate has no `unsafe` code of its own.

- High: memory unsafety reachable from input, writing outside the output
  directory (path traversal), running a command other than the operator's
  scorer, or a cached or replayed entry that makes a run accept a candidate
  the scorer did not verify.
- Medium: panics, unbounded memory or CPU, or hangs on malformed creature
  JSON, corpus files, caches, graft memory or journals.
- Low: wrong statistics in reports or economics summaries, or issues that need
  the operator's own arguments to be hostile.

## Anything to leave alone

- Bugs in `neat-core` (NEAT-AI-core) or `rust_scorer` belong in those
  repositories.
- A candidate that fails to improve the score is the expected outcome the
  scorer gate exists for, not a vulnerability.
