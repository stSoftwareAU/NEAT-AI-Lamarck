## Summary

Measured the scorer's fixed per-call cost on the production creature and corpus,
journalled the per-call creature counts that make the measurement reproducible
from any run, and delivered the go/no-go. Closes #112.

The scoring path itself is unchanged — the only behaviour added is measurement.

**Result.** A sampled (screen) call costs **9 898 ms before it scores its first
creature**, then 452 ms per creature. A full-corpus (promote) call costs
**1 977 ms** fixed, then 5 490 ms per creature. Applied to the #75 baseline run's
own call mix (75 screen calls, 26 promote calls, 1 Phase-0 call, 2 236 s of
scorer time), the fixed per-call cost is **24–29% of a 45-minute run — 11–13
minutes of every 45**.

**Decision: Go**, filed as follow-up
[#123](https://github.com/stSoftwareAU/NEAT-AI-Lamarck/issues/123),
cross-referenced to NEAT-AI-scorer#536. #123 carries the cheaper lead the
measurement turned up: the sampled call's fixed cost is **5× the full-corpus
call's** (9.9 s vs 2.0 s) while doing a twentieth of the scoring, so the
`--sample-rate` path pays a per-call setup the streaming path does not — fixing
that inside the scorer would return ~20% of a run with **no** wire protocol, no
supervised process, and none of a persistent session's failure modes.

### What changed

- **`lamarck/src/scorer_cost.rs`** (new) — the per-call cost model: a
  `ScorerCallRecord` (phase, creature count, sample rate, elapsed ms, failed) and
  a least-squares fit whose intercept is the fixed per-call cost and slope the
  marginal per-creature cost. Fitted **per phase**: a sampled screen call and a
  full-corpus promote call have different marginal costs, so pooling them would
  fit one line through two populations. Failed calls are counted but never
  fitted — an aborted call would drag the intercept towards zero.
- **`RecordingScorer`** (`lamarck/src/scorer.rs`) — a `DirectoryScorer` wrapper
  that measures every call it forwards. Wrapping once inside
  `run_optimisation_cancellable` catches Phase-0, Phase-G graft replay, screen,
  promote **and** combo batches with no per-call-site changes. An unlistable
  batch directory is now an error rather than a call recorded as scoring nothing.
- **Journal** — `scorerCalls[]` on the experiment and graft-replay lines, plus a
  new `scorerCalls` line (`stage: phase0` / `trailing`) for calls that belong to
  no experiment. `scorerMs` sums calls of different sizes, so it cannot be
  regressed alone; the creature count per call is what recovers the split.
- **`report`** — a `scorerCallCost` bucket (`calls`, `failedCalls`,
  `creaturesScored`, `byPhase` → `fixedMs`, `marginalMsPerCreature`, `rSquared`,
  `fixedMsShareAtMean`), printed in the run summary. A phase measured at one
  batch size reports `null` rather than an intercept invented from one point; a
  pre-#112 journal reports an empty bucket.
- **Run summary scorer counts now come from the recorder.** They previously
  missed combo batches entirely — a combo batch was neither a success nor a
  failure. This is what makes the journal-completeness invariant testable.
- **Harness** — `scripts/measure-scorer-call-cost.sh` +
  `lamarck/examples/scorer_call_cost_bench.rs`, which sweep batch sizes at a
  chosen sample rate and record `loadBefore` / `loadAfter` around each run.
- **Write-up** — `docs/scorer-call-cost.md` with the method, the numbers, the
  conditions, the decision, and what the measurement cannot support.

### Evidence

Backend/CLI only — no web interface to screenshot. Raw measurement logs are
committed at
[`docs/evidence/scorer-call-cost/rate-0_05.log`](../../evidence/scorer-call-cost/rate-0_05.log)
and
[`docs/evidence/scorer-call-cost/rate-1.log`](../../evidence/scorer-call-cost/rate-1.log).

Measured on the GRQ champion (2511 inputs, 1605 neurons, 22 016 synapses) against
the 21 GiB, 522-file baseline corpus, 15 calls, 0 failures:

| Phase | Sample rate | Calls | `fixedMs` | `marginalMsPerCreature` | `rSquared` |
|-------|-------------|-------|-----------|-------------------------|------------|
| screen | `0.05` | 9 | **9 898** | 452 | 0.957 |
| promote | full corpus | 6 | **1 977** | 5 490 | 0.973 |

Scoring **one** creature on a 5% sample takes ≈10.3 s, of which ≈0.45 s is
scoring.

**Conditions, per the `docs/followup-economics.md` load caveat.** This was
**not** an idle box: a live production Lamarck run held 7–8 of the 10 cores
throughout, 1-minute load average 8.80 → 25.68. The write-up records this and
handles it two ways — a direct projection from the measured intercepts (35.6% of
scorer time) and a load-corrected one from the measured fixed *shares* applied to
the baseline campaign's own scorer time (≈29%) — which bracket the answer. A
share survives proportional inflation where an absolute intercept does not, and
contention inflates the CPU-bound slope at least as much as the I/O-bound
intercept, so the measured fixed share is a floor. The follow-up requires a
repeat on an idle box before the saving is quoted in a design.

```mermaid
flowchart LR
    P0["Phase-0 call"] --> REC["RecordingScorer<br/>phase + creatures + ms"]
    SCR["screen call"] --> REC
    PRO["promote call"] --> REC
    CMB["combo call"] --> REC
    GRF["graft-replay call"] --> REC
    REC --> J[["experiments.jsonl<br/>scorerCalls[]"]]
    J --> FIT["report: OLS per phase"]
    FIT --> OUT(["fixedMs = what a session removes<br/>marginalMsPerCreature = what it cannot"])

    classDef call fill:#dbeafe,stroke:#1d4ed8,stroke-width:2px,color:#0b2545
    classDef stage fill:#fef3c7,stroke:#b45309,stroke-width:2px,color:#451a03
    classDef out fill:#dcfce7,stroke:#15803d,stroke-width:2px,color:#052e16
    class P0,SCR,PRO,CMB,GRF call
    class REC,J,FIT stage
    class OUT out
```

### Test Plan

Added, all run by `./quality.sh` (green):

- `lamarck/src/scorer_cost.rs` — the fit recovers a known intercept and slope
  exactly (`ms = 5000 + 300 × creatures`); a pure per-creature cost fits a zero
  intercept; the fixed share is reported against an average call; **one** batch
  size reports no decomposition rather than a fabricated intercept; an empty
  input is not a zero intercept; phases are fitted separately; failed calls are
  counted but not fitted; phase labels round-trip through JSON.
- `lamarck/src/scorer.rs` — `RecordingScorer` records creature count, phase and
  sample rate (and ignores non-`*.json` files); a failed call is still recorded
  as `failed`; an unlistable batch directory is an error, not a zero-creature
  call.
- `lamarck/src/report.rs` — fixture journals with known call sizes and times
  recover the intercept and slope **exactly, per phase** (`screen` 800 + 40,
  `promote` 9000 + 11 000 — the wrong-decomposition guard); Phase-0 and
  graft-replay lines are folded into the model (the missing-calls guard); a
  pre-#112 journal reports an empty bucket.
- `lamarck/src/run.rs` — end to end with a fake scorer: the journalled call count
  equals the run's own `scorerSuccesses + scorerFailures`, every call names its
  creatures, Phase-0 / screen / promote all appear, a screen call carries the
  sample rate and a promote call does not, and `report` over that same journal
  regresses it.
- `lamarck/tests/scorer_call_cost_doc.rs` — the write-up names tooling that
  exists, quotes only report fields that are serialised, keeps `loadBefore` /
  `loadAfter` on the page, carries an explicit go/no-go cross-referenced to
  NEAT-AI-scorer#536, and keeps its limits section.

Modified: `run::tests::loop_accepts_winner_and_writes_journal` now expects the
Phase-0 `scorerCalls` line between the run header and the first experiment. The
assertion's intent (header first, experiments follow) is unchanged; the journal
has one more line.

Backwards compatibility checked against the real pre-#112 baseline journal:
`report` over it emits `"scorerCallCost": {"calls": 0, "failedCalls": 0,
"creaturesScored": 0, "byPhase": {}}` and every other figure unchanged.
