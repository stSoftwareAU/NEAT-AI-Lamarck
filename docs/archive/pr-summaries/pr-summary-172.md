# README Phase 5 scoring costs now defer to the measurement (Issue #172)

## Summary

`README.md`'s Phase 5 walkthrough stated the screen call costs "≈0.7–1s/creature
on GRQ against ≈11s full". Those figures were written in PR #76, before
issue #112's fixed/marginal cost decomposition existed, and are roughly double the
measured marginal costs in `docs/scorer-call-cost.md` — **452 ms/creature** for
the screen phase and **5 490 ms/creature** for the full-corpus promote phase —
the document the README itself cites as authoritative two sections later.

The inline figures are removed. Phase 5 now describes what the screen call *is*
(a sampled slice of the corpus rather than all of it) and links
`docs/scorer-call-cost.md` as the source of truth for what each call costs,
matching the "link, do not restate" convention the README follows for its other
measurement documents. Closes #172.

Out of scope and filed separately: `docs/promote-gate.md` carries the same
pre-#112 figures ("~1 s at 5%", "~11 s/creature") — stSoftwareAU/NEAT-AI-Lamarck#183.

## Evidence

This is a documentation change with no web interface, so there is no screenshot
to capture. The evidence is the new test, which fails against the old README
wording and passes against the new one.

Against the pre-fix README (the contradiction the issue reports):

```text
---- phase_five_states_no_timing_the_measurement_contradicts stdout ----
README Phase 5 states scorer timings docs/scorer-call-cost.md does not measure:
["`0.7` (= 700 ms)", "`1` (= 1000 ms)", "`11` (= 11000 ms)"]
 — measured ms: [9898.0, 452.0, 1977.0, 5490.0].
Cite the document rather than restating its numbers.

---- phase_five_cites_the_measured_scorer_call_cost_document stdout ----
README Phase 5 does not link docs/scorer-call-cost.md as the source of scoring costs

test result: FAILED. 9 passed; 2 failed
```

After the fix:

```text
cargo test --test readme_scorer_cost_consistency
test result: ok. 11 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

Full workspace suite: `cargo test --workspace --all-features -- --test-threads=2`
— all suites green (405 unit tests plus every integration suite). `cargo fmt
--check`, `cargo clippy` (with `-D warnings`), `cargo deny check` and
`RUSTDOCFLAGS="-D warnings" cargo doc` all pass. `./quality.sh` stops earlier
than that on this container because `codespell` is not installed and cannot be
installed (no `pip`/`pipx` present); CI runs that step for real, and the change
adds no unusual vocabulary.

Where the numbers come from, and what the test now enforces:

```mermaid
flowchart LR
    JOURNAL["run journal<br/>scorerCalls"] --> FIT["least-squares fit<br/>lamarck/src/scorer_cost.rs"]
    FIT --> DOC["docs/scorer-call-cost.md<br/>Result table<br/>452 ms screen · 5 490 ms promote"]
    DOC -->|"linked as source of truth"| README["README.md Phase 5"]
    DOC -->|"parsed at test time"| TEST["readme_scorer_cost_consistency"]
    README -->|"every timing it states"| TEST
    TEST -->|"unsupported figure"| FAIL["test fails"]

    classDef source fill:#1f3a5f,stroke:#8ab4f8,color:#ffffff
    classDef doc fill:#264d3b,stroke:#7ddba3,color:#ffffff
    classDef gate fill:#5c3a1e,stroke:#f0b37e,color:#ffffff
    class JOURNAL,FIT source
    class DOC,README doc
    class TEST,FAIL gate
```

## Test Plan

New `lamarck/tests/readme_scorer_cost_consistency.rs` (11 tests). It parses the
Result table out of `docs/scorer-call-cost.md` and the Phase 5 body out of
`README.md` at test time, so it keeps working whichever side changes next.

- `phase_five_states_no_timing_the_measurement_contradicts` — the regression
  test. Every duration Phase 5 states must be within 5% of a measured `fixedMs`
  or `marginalMsPerCreature` from the document's Result table. Reproduces the
  issue against the old wording (see Evidence above).
- `phase_five_cites_the_measured_scorer_call_cost_document` — Phase 5 links the
  measurement rather than leaving a reader to guess.
- `the_measured_result_table_still_parses_into_per_phase_fits` — guards the
  check above from passing vacuously if the table is renamed or restructured.
- Helper coverage (happy path, edges, failure): `section` on a requested,
  nested and missing heading; `measured_costs` on bolded space-grouped cells and
  on a table-less section; `durations` on the pre-fix Phase 5 wording (asserting
  it reads 700 / 1 000 / 11 000 ms), on both `ms` and `s` units mixed with bare
  numbers and percentages, and on text where a unit letter only starts a word
  (`0.05 sample`, `1e-6`) plus empty input.
