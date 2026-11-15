## Summary

Add an opt-in, offline benchmark of local weight optimisers under explicit
evaluation budgets (`lamarck/src/weight_bench/`, the `weight_router_bench`
example), an offline AMM reference driver pinned to the authors' release, and
synthetic evidence. A budget-aware router and its landscape probes were
measured, never beat CMA-ES alone, and were removed; AMM lost in every cell, so
nothing is ported. Nothing in the optimisation loop calls the benchmark and no
default changes. The representative comparison on production inputs is still
outstanding. Part of #266.

This branch is the earlier PR #271 work, squashed onto the current `Develop`
after #271 was closed for a `CHANGELOG.md` conflict. The conflict was two
adjacent `### Added` bullets (#267 and #266); both are kept. The CI gate
fixes that PR #271 carried are already on `Develop` through #270, so they are
not in this diff.

- [x] Read and pin the full AMM method; check release compatibility.
- [x] Run AMM, CMA-ES and the refinement surrogate on fixed synthetic screening records.
- [x] Correct comparison and accounting defects; validate failure handling.
- [x] Remove the router and probes after the negative finding; regenerate evidence from head.
- [x] Re-land the work on the current `Develop` and pass `./quality.sh`.
- [ ] Establish production-baseline equivalence and run representative inputs.
- [ ] Complete full authoritative confirmation on representative inputs.

## Spec

### Intent and Rationale

- Compare local weight search under explicit budgets without touching production acceptance, so a negative or positive finding costs nothing in the run loop.
- The squash re-lands reviewed work instead of rewriting it; only the changelog conflict needed resolving.

### Essential Design Decisions

- `lamarck-refine` is a random-axis surrogate that reuses `RANDOM_DELTA_HALF_WIDTH` (`candidates.rs`) and `STEP_FRACTIONS` (`followup.rs`); it is not `generate_candidate_batch`, and the docs claim no equivalence.
- The router and probes are removed, not kept: the issue ports only winning probes or routers. With no tuned parameter there is no calibration phase; `Manifest` refuses unknown fields, so a stale `calibrationSeeds` is rejected.
- AMM runs only in a separate, checksum-pinned scratch Python environment (preset `c46`); no dependency is added and it never runs in CI.
- Errored rows never enter paired distributions, while their spent evaluations stay counted.

### Undiscoverable Facts

- The full method in [arXiv:2611.06879v1](https://arxiv.org/pdf/2611.06879v1) was read (Section 3, Algorithms 1–2, Tables 2/6, Appendices A–B). The paper's `v0.3.0` resolves to `c4f0ab160aae912311328145aa5a426b26a06b26`; `v0.3.0` and the pinned `v0.7.0` gave 27 identical rows on one validation seed.
- No authorised production model and corpus pair was found on the host, so representative coverage is zero.
- The worker's pre-push gate reads a SHA-pinned `dtolnay/rust-toolchain` ref as a toolchain (`cargo +29eef336… --version` exits 101). `Develop` routes three jobs through `.github/actions/rust-toolchain`, but `ci.yml`'s `validation` job, `auto-format`, `cargo-audit`, `sbom` and `security` still call the pinned action directly.

## Evidence

CLI/backend only; no UI file is touched.

- `docs/evidence/weight-router/manifest.json`, `summary.json` and `reference-summary.json` come from the current benchmark code (`formatVersion` 2): 648/648 runs, 0 failed, 1,631,615 evaluations. CMA-ES beat the surrogate in 19 of 27 cells, lost 1, and 7 were inconclusive. Over the committed 324 AMM rows, AMM is worse in all 54 comparisons. These are synthetic results, not representative ones.
- `docs/evidence/weight-router/method-check.json` and `amm-runs.jsonl` are router-era output (crate 0.1.46); `docs/weight-optimiser-routing.md` labels them so.
- `Cargo.lock` moves `rustix` 1.1.4 → 1.1.6 for RUSTSEC-2026-0486, which the dependency audit flagged on #271.
- #266: Experiment: Benchmark budget-aware routing of local weight optimisers.

```mermaid
flowchart LR
    M[Manifest validation] --> V[Paired validation seeds]
    V --> B[Budgeted refine / CMA-ES arms and AMM server]
    B --> F{Run failed?}
    F -->|yes| E[Record error and spent costs; exclude pair]
    F -->|no| H[Full-window and held-out confirmation]
    H --> S[Paired summaries with bootstrap intervals]
```

## Acceptance Criteria

- **missing** — Use AMM as an offline reference against the current refinement method and CMA-ES on fixed deterministic screening records. — reason: AMM and CMA-ES ran offline on fixed synthetic records, but the refinement arm is a surrogate, not the current production refinement pipeline.
- **missing** — Test plateaus/kinks and activation families, then full/held-out confirmation. — reason: done on the synthetic suite only; full authoritative confirmation on representative inputs is not run.
- **missing** — Identify code/model/data versions; separate calibration from untouched validation and count all preparation/search costs. — reason: outputs record the crate version only; no representative model or data version is frozen.
- **missing** — Deliver a reproducible command/manifest and machine-readable results with paired distributions, uncertainty, failures and actual coverage. — reason: the synthetic side is complete, but representative coverage is 0 runs.

## Test Plan

- `timeout 1500 ./quality.sh < /dev/null` passed on `d044281` (exit 0, "All quality checks passed!"): shell and workflow checks, spelling, dependency audit, formatting, Clippy, workspace tests and rustdoc.
- `cargo test --test weight_bench --test weight_bench_reference < /dev/null` (from `lamarck/`) passed 19 and 13, with one debug-mode ignore; `cargo test --lib weight_bench` passed 2.
- Tests added by this PR include `lamarck/tests/weight_bench.rs::manifest_rejects_unknown_fields`, `manifest_rejects_duplicate_trial_identifiers` and `extrapolation_flag_marks_rows_and_cells_just_past_the_boundary`; `lamarck/tests/weight_bench_reference.rs::held_out_veto_turns_a_screening_win_inconclusive`, `failed_reference_with_completed_metrics_is_excluded_from_pairs` and `committed_evidence_parses_and_summarises`; `lamarck/src/weight_bench/experiment.rs::tests::failed_confirmations_keep_the_calls_already_spent`; and `lamarck/src/weight_bench/arms.rs::tests::arm_seeds_are_pinned`.
- `scripts/test-amm-reference.py` (offline integration checks against the real pinned AMM and Rust server) was not re-run here: no scratch AMM environment exists, and building one needs network access.
- No existing test assertion is removed relative to `Develop`; the benchmark module and its test files are new.

## Merge verification

- Pushed through the normal guarded `git push`. GitHub CI is not yet verified.

Not done: representative runs on production inputs, production-baseline
equivalence, full authoritative confirmation, and a re-run of the AMM campaign
(its committed rows are re-summarised against the current in-process runs).
These prevent any optimisation-benefit or issue-completion claim.
