# Architecture

NEAT-AI-Lamarck is intentionally a small orchestration/analysis layer around existing NEAT-AI components.

## Responsibilities

### NEAT-AI-Lamarck

- build/load `observations.statistics`;
- select a focus neuron;
- collect incumbent-specific neuron statistics;
- expose/port conventional backpropagation learning signals;
- generate candidate creatures;
- invoke authoritative batch scoring;
- accept only verified improvements;
- journal experiments and preserve the best verified creature.

### NEAT-AI-core

- creature/network representation and compilation;
- activation primitives and shared low-level network behaviour.

Generic Lamarck code may migrate here later only after the experiment proves useful and its interfaces stabilise.

### NEAT-AI-scorer

- authoritative full-corpus scoring;
- batch/directory candidate evaluation;
- final decision data used to accept/reject candidates.

Lamarck must not duplicate this authority.

## Iteration lifecycle

1. Start from the supplied incumbent.
2. Verify baseline score/parity.
3. Select one focus neuron.
4. Scan/measure the incumbent as needed.
5. Generate approximately 50 candidates from backpropagation, statistical, structural and random strategies.
6. Batch score baseline + candidates with NEAT-AI-scorer.
7. Accept the best candidate only if it clears the meaningful-improvement threshold.
8. Treat creature-specific statistics as stale after an acceptance.
9. Repeat until the wall-clock budget (45 minutes by default) expires.

## Locked contracts

- Accept only on authoritative scorer JSON **`score`** (larger-is-better).
- Default meaningful improvement: absolute **`1e-6`**, strict `>`.
- Scorer argv: `rust_scorer <candidates_dir> <training_data_dir>` — no `--gpu` / `--cost`.
- `observations.statistics` is versioned human-readable JSON (semver).
- Production scale target: GRQ `network.json` (~2511 inputs, ~1590 hidden, ~21k synapses).

## Design bias

The normal evolutionary system runs concurrently, so the starting champion has a limited useful lifetime. Prefer cheap, repeatable experiments over expensive analyses whose expected value does not justify consuming the run budget.
