# Architecture

NEAT-AI-Lamarck is intentionally a small orchestration/analysis layer around existing NEAT-AI components.

## Responsibilities

### NEAT-AI-Lamarck

- build/load `observations.statistics`;
- select a focus neuron;
- collect incumbent-specific neuron statistics;
- accumulate conventional backprop learning signals via neat-core
  `propagate_topological_loop` (analyse-without-apply);
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
3. Accumulate learning + output residuals, then select one focus neuron
   (default: `weighted` random by error-influence mass — output residual L1 and
   depth-decayed hidden blame; zero-signal neurons excluded). Prefer this over
   `high-error`, which sticks on a single neuron.
4. Scan/measure the chosen focus on the incumbent.
5. Generate ~100 candidates (default; keep the scorer CPU-saturated) from backpropagation, statistical, structural and random strategies.
6. **Screen** on a cheap scorer subsample (default 5% of rows); promote only sample Δ `> 1e-6`.
7. **Full-corpus** score baseline + promoted candidates; accept only if Δ score clears `1e-6`.
8. Treat creature-specific statistics as stale after an acceptance.
9. Repeat until the wall-clock budget (45 minutes by default) expires.

## Locked contracts

- Accept only on authoritative scorer JSON **`score`** (larger-is-better).
- Default meaningful improvement: absolute **`1e-6`**, strict `>`.
- Scorer argv: `rust_scorer <candidates_dir> <training_data_dir>` — no `--gpu` / `--cost`.
  Screen phase (issue #24) may add `--sample-rate` / `--sample-phase` only; acceptance
  still uses a full-corpus promote score.
- Phase-0 (issue #4): before optimising, Lamarck MSE must agree with scorer `error`
  (and unpenalized `1−error` with `score+complexityPenalty`) within
  `PHASE0_*_TOL` in `lamarck/src/parity.rs`; abort on unexplained disagreement.
- Focus analysis surfaces real backprop blame (`LearningSignal`) for hidden and
  output focuses; never invents a hidden-neuron target.
- `observations.statistics` is versioned human-readable JSON (semver).
- Production scale target: GRQ `network.json` (~2511 inputs, ~1590 hidden, ~21k synapses).

## Design bias

The normal evolutionary system runs concurrently, so the starting champion has a limited useful lifetime. Prefer cheap, repeatable experiments over expensive analyses whose expected value does not justify consuming the run budget.
