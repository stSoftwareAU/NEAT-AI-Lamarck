# Judge combos against the incumbent scored in their own call

## Summary

`rust_scorer` reports a creature's full-corpus score as a function of **which
other creatures shared the directory call** — `1.755e-7` relative on the
production creature between a directory of one and of two, and the incumbent and
a candidate do not move by the same amount. Lamarck accepts at
`--min-improvement 1e-6` and its measured full-corpus deltas live at `1e-7`–`1e-6`,
so a Δ taken across two calls of different sizes carries that artefact straight
into an accept decision. Closes #130.

Two halves shipped:

- **This repo — never subtract across calls.** The combo call already wrote and
  scored `baseline.json` and then threw that score away, comparing each combo
  against the *promote* call's baseline. That score is now the only baseline a
  combo is judged against, combos beat the best improving single on **Δ** rather
  than on a raw score from another call, a combo call that returns no `baseline`
  stem fails loudly instead of borrowing one, and `run.rs` gates the accept on
  `ComboSelection::delta` instead of re-subtracting the promote baseline from a
  winner scored elsewhere. The screen gate, the single-candidate Δ and
  `verify_accept_pair` were already same-call and are unchanged.
- **Upstream — remove the artefact.** The scorer fix is written, green and
  pushed on
  [`issue-lamarck-130-batch-invariant-partition`](https://github.com/stSoftwareAU/NEAT-AI-scorer/tree/issue-lamarck-130-batch-invariant-partition):
  the record partition is now fixed 64-record blocks decided by the corpus, so
  batch size, `activation_threads` and `NEAT_SCORER_WORKER_SPLIT` are invisible
  in the scores (bit-identically). This run could not open that PR —
  `WRITE_REPO_BLOCKED` confines its GitHub writes to this repository — so #143
  tracks the human hand-off and what is owed once the released scorer lands.

## Evidence

Backend/CLI change, so no screenshot. The behaviour is pinned by tests that
drive the real `select_best_with_combinations` with a scorer that reproduces the
artefact: every score in the combo call is shifted, exactly as a different batch
size shifts it in production.

Which subtractions are legal, before and after:

```mermaid
flowchart TD
    PROMOTE["promote call<br/>baseline + promoted candidates"] --> PD["single Δ<br/>candidate − promote baseline"]
    COMBO["combo call<br/>baseline + combo creatures"] --> CD["combo Δ<br/>combo − combo baseline"]
    COMBO -. "was: combo − promote baseline<br/>(different batch, not comparable)" .-> OLD["cross-call Δ"]
    PD --> GATE{"Δ > --min-improvement?"}
    CD --> GATE
    OLD -. removed .-> GATE
    GATE -- yes --> WIN(["new incumbent"])
    GATE -- no --> KEEP(["keep incumbent"])

    classDef call fill:#cffafe,stroke:#0e7490,stroke-width:2px,color:#083344
    classDef delta fill:#ede9fe,stroke:#6d28d9,stroke-width:2px,color:#2e1065
    classDef win fill:#dcfce7,stroke:#15803d,stroke-width:2px,color:#052e16
    classDef reject fill:#fee2e2,stroke:#b91c1c,stroke-width:2px,color:#450a0a

    class PROMOTE,COMBO call
    class PD,CD delta
    class WIN win
    class KEEP,OLD reject
```

`./quality.sh` passes (fmt, clippy `-D warnings`, cargo-deny, codespell, 314
library tests plus the integration suites, rustdoc).

Upstream benchmark evidence for the scorer branch — Criterion
`score_from_creature_dir`, 16 MiB corpus, this M-series host, medians with
Criterion's own change analysis:

| creatures | before | after | verdict |
| --- | --- | --- | --- |
| 1 | 37.96 ms | 46.12 ms | no change detected (p = 0.62) |
| 10 | 115.16 ms | 116.89 ms | no change detected (p = 0.44) |
| 50 | 332.45 ms | 265.87 ms | **−20.8%** (p = 0.00) |
| 200 | 830.70 ms | 677.29 ms | **−18.5%** (p = 0.00) |

## Test Plan

Added to `lamarck/src/combos.rs`:

- `combo_is_rejected_when_only_a_cross_call_baseline_would_accept_it` — the
  combo call's scores sit 3e-6 above the promote call's; a combo whose same-call
  Δ is 0.5e-6 must lose to the improving single, where the old rule accepted it
  on a 3.5e-6 cross-call "improvement". Fails against the unfixed code.
- `combo_wins_on_its_same_call_delta_even_when_its_raw_score_looks_worse` — the
  mirror case: the combo call sits 3e-6 below, so the combo's raw score never
  beats the single's, but its 5e-6 same-call Δ is the larger real improvement.
  Fails against the unfixed code.
- `combo_call_without_a_baseline_score_is_a_loud_error` — a combo call that
  returns no `baseline` stem is an error, not a silent fallback. Fails against
  the unfixed code.
- `selection_accepts_on_its_same_call_delta` — `ComboSelection::accepts` reads
  the Δ and the bar is strict.

Added `lamarck/tests/scorer_batch_composition_doc.rs` (5 tests): the accept gate
behaves as `docs/scorer-batch-composition.md` documents, and the document keeps
the measured evidence, the code paths it names, its "what this does not fix"
section, and the README link.

Upstream, on the scorer branch: `rust_scorer/tests/batch_composition_invariance.rs`
(3 tests) pins a creature's `error`/`score` as bit-identical alone versus in
batches of two and three, across `NEAT_SCORER_ACTIVATION_THREADS` of 1/2/5/13,
and across `NEAT_SCORER_WORKER_SPLIT` of 1/2/4/8. All three fail against the
unfixed scorer.

No existing test was removed or disabled. One upstream test was modified and the
reasoning recorded in it: `bin_lib_single_source` compared the binary's and the
library's floats with `assert_eq`, and its 4-record fixture now reaches the
4-record SIMD path where `Simd::reduce_sum` leaves float summation order
unspecified, so two separately code-generated artefacts differ by 1 ULP. Scored
floats are compared within `4 * f64::EPSILON`; `recordCount`,
`complexityPenalty` and `costName` stay exact.
