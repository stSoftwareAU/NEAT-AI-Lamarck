# Make the screen promote gate noise-aware, behind a flag (Issue #111)

## Summary

The screen phase promoted every candidate whose 5%-sample Δ beat a fixed
`--screen-promote-threshold` of `1e-6` — an absolute bar on a sampled
measurement, sitting at about **one** standard deviation of the screen's own
noise (`docs/screen-calibration.md`). Each promotion buys an ~11 s full-corpus
score on a coin flip: across the journals in hand, 244 promotions produced 3
scores over the accept bar and 207 that made the creature worse.

This PR adds `--screen-promote-gate noise-aware`, which prices each batch's own
spread before deciding: promote on `Δ > max(k · σ̂, --screen-promote-threshold)`,
with `k` from `--screen-promote-sigma-k` (default `3`) and σ̂ the lower quartile
of the batch's absolute screen deltas rescaled from a half-normal. **The default
is unchanged** — `absolute` is byte-for-byte the pre-#111 run, and moving it
needs benchmark evidence on accepts per wall-clock hour, not promotions avoided.
Closes #111.

Design points a reviewer should check:

- **Why the lower quartile.** A candidate batch is bimodal: structural
  proposals routinely move the score by `5e-2` while weight/bias nudges move it
  by `~1e-8`. On a real #75 batch the standard deviation is `2.5e-2` and the MAD
  `1.9e-2` — both measure *proposal dispersion*, not the screen's resolution
  floor, and a gate built on either promotes nothing for the wrong reason. The
  lower quartile sits inside the near-zero core and returns `~4e-6` on the same
  batch. `--screen-sample-rate` is deliberately not applied a second time: σ̂ is
  measured on scores produced at the run's own rate.
- **The gate can never be weaker than `absolute`.** The threshold is a `max`
  with the existing floor, so the promoted set is always a subset. Acceptance is
  untouched — full corpus, `--min-improvement`.
- **Degenerate batches fall back.** Fewer than four candidates, a non-finite
  delta, or a zero lower quartile yields no estimate and reverts to the absolute
  floor rather than dividing by zero or promoting everything.
- **Combo accepts are replayed through their members.** A merged combo has no
  screen score of its own, so the offline replay resolves `comboMemberIndices`
  and only counts it kept if every member still clears the gate; unknowable
  members (pre-#74 journals) count as **dropped**, so an unverifiable accept
  surfaces instead of passing silently.

Journalling and reporting so the change can be priced from a run:
`screenPromoteGate` / `screenPromoteSigmaK` in the `runHeader`, a per-experiment
`screenTiers` record (gate, screened, promoted, threshold, σ̂), and a
`promoteGateReplay` bucket in `report` that replays the gate over any journal.

## Evidence

Backend/CLI only — no web interface to screenshot.

### Offline replay over every journal in hand

`scripts/summarise-promote-gate.sh` over the #8 baseline and the seven #75 arms
(6805 screened candidates, 244 promotions, 2 accepts), at the default `k = 3`:

| Journal | Gate as run | Screened | Promoted as run | Promoted under gate | Avoided | Avoided share | Accepts kept |
|---------|-------------|----------|-----------------|---------------------|---------|---------------|--------------|
| `.lamarck-baseline-45` | none (pre-#111) | 1954 | 115 | 83 | 32 | 27.8% | 2 / 2 |
| `backprop-cap-tenth` | absolute | 462 | 10 | 0 | 10 | 100% | 0 / 0 |
| `backprop-lr-tenth` | absolute | 693 | 17 | 0 | 17 | 100% | 0 / 0 |
| `batch-100` | absolute | 924 | 26 | 0 | 26 | 100% | 0 / 0 |
| `batch-150` | absolute | 660 | 17 | 0 | 17 | 100% | 0 / 0 |
| `batch-40` | absolute | 594 | 14 | 0 | 14 | 100% | 0 / 0 |
| `output-focus` | absolute | 660 | 20 | 0 | 20 | 100% | 0 / 0 |
| `seed-2` | absolute | 858 | 25 | 0 | 25 | 100% | 0 / 0 |
| **pooled** | absolute | 6805 | 244 | 83 | 161 | **66%** | **2 / 2** |

Every accepted winner these journals contain, replayed against the gate:

| Experiment | Stem | Screen Δ | Gate demanded | σ̂ | Still promoted |
|------------|------|----------|---------------|-----|----------------|
| 3 | `candidate-025` | 4.74e-6 | 1e-6 | 9.37e-8 | yes |
| 5 | `candidate-018` | 5.38e-6 | 1e-6 | 2.31e-7 | yes |

Both accepts survive at every `k` from 1 to 5, asserted as a hard `cargo test`
failure in
`lamarck/tests/promote_gate_replay.rs::the_noise_aware_gate_would_still_have_promoted_both_historical_accepts`
over committed journal fixtures — so a gate that drops a real improver fails CI
rather than a production run.

### Paired benchmark

Production creature, 2 262 277-record GRQ corpus, seed 81, `--candidates 100`,
900 s per arm, run back to back via
`scripts/run-followup-economics.sh promote-gate` (the new arm in this PR):

| Metric | `absolute` (control) | `noise-aware` (k = 3) |
|--------|----------------------|-----------------------|
| Experiments completed | 23 | 24 |
| Wall duration | 864 s | 891 s |
| Experiments / hour | 95.8 | 97.0 |
| Full-corpus promotions | 46 | 48 |
| Promotions / experiment | 2.00 | 2.00 |
| Promote scores / scorer-minute | 3.25 | 3.26 |
| **Accepts / hour** | **0** | **0** |
| Load average before → after | 8.4 → 15.6 | 15.6 → 21.5 |

**The arms made the same decision in all 23 experiments they share** — identical
promotion counts, with the gate's threshold at the `1e-6` floor in 18 of them
and raised to `1.22e-6`–`3.75e-6` in the other 5 without changing the outcome.
σ̂ across the noise-aware arm's 24 batches ran `2.69e-9`–`1.25e-6`, median
`2.98e-8`, so `3σ̂` sits two orders of magnitude below the floor and
`max(3σ̂, 1e-6)` collapses to it. **At `k = 3` the gate is inert on this
creature with today's generator.** The 66% saving in the replay is real for the
archived journals but they were written by Lamarck `0.1.7`, whose batches
carried a far wider near-zero core (lower quartile of |Δ| ~`1.2e-6`, σ̂ ~`3.9e-6`)
than the generator produces after #105–#109. Replaying the control arm's own
journal, `k` has to reach ~30 before promotions fall (46 → 22).

Neither arm accepted anything in 15 minutes, so the benchmark cannot price
accepts per wall-clock hour, which is the metric that is supposed to decide the
default. Both measured `0`. Full write-up, with the σ̂ table and the caveats:
[`docs/promote-gate.md`](../../promote-gate.md).

### Decision

**The default stays `absolute`, and the flag ships opt-in.** The benchmark did
not support a change: the deciding metric (accepts per hour) was `0` on both
sides, and at the default `k` the gate is inert on the production creature, so
making it the default would change nothing there while risking the invisible
false negative on a differently shaped run. Moving it needs an arm long enough
to produce accepts on both sides — the #8 baseline took 75 experiments to find
2 — and a `k` calibrated on the current generator's σ̂. That is stated as the
decision in `docs/promote-gate.md` rather than left implicit.

## Test Plan

- `lamarck/src/promote_gate.rs` — σ̂ recovery on synthetic Gaussian batches at
  three scales; robustness to a contaminating tail and the honest inflation past
  the quartile's breakdown point; the degenerate cases the issue named
  (all-identical scores, identical positive deltas, a single candidate,
  all-negative deltas, a non-finite delta); "never weaker than absolute" over
  four batch shapes; monotonicity in `k`; mode parsing; the replay's combo
  handling and its loud failure on a screen map with no baseline.
- `lamarck/src/config.rs` — the default is the absolute gate at the pre-#111
  threshold; the noise-aware gate carries the threshold as its floor; a
  non-positive or non-finite `k` is rejected naming the flag; an unused `k` does
  not fail a default run.
- `lamarck/src/scorer.rs` — the absolute gate promotes exactly the pre-#111
  stems in the same order (default-drift guard); the noise-aware gate drops the
  wobble and keeps the improver; a batch with no baseline fails loudly.
- `lamarck/src/run.rs` — end to end on a bimodal fake scorer: the noise-aware
  run promotes the improver alone and journals its tiers and header knobs; the
  default run promotes every candidate over the bare threshold, with no σ̂ and no
  `screenPromoteSigmaK`.
- `lamarck/tests/promote_gate_replay.rs` — the accepts-survive gate above, the
  never-buys-more property on real batches, the pure-cost arm, and the report
  wiring.
- `lamarck/tests/promote_gate_doc.rs` — `docs/promote-gate.md` ↔ code contract:
  the tooling it names exists, the report fields it is built on are still
  emitted, and its "cannot support" and unchanged-default statements survive.
- `lamarck/tests/readme_contract.rs` — README ↔ CLI parity in both directions,
  covering the two new flags.
- `./quality.sh` — passes (fmt, clippy `-D warnings`, cargo-deny, codespell,
  shellcheck, 336 tests, rustdoc).
