# PR summary — issue #268

## Summary

Adds an opt-in, offline research harness (`lamarck/src/local_rules/`) that
tests whether sparse symbolic local update rules beat Lamarck's sparse backprop
refinement at an equal record budget, with the meta-search cost counted.
Nothing in `run_optimisation` or the CLI calls it. The predeclared synthetic run
is **negative**: both primary baselines (sparse backprop and a derivative-free
(1+1) search) beat the discovered rules at K = 5, 10 and 20 once the
meta-search is charged. The method, protocol and results are in
`docs/local-rules.md`.

This re-lands the harness from the closed PR #269 and fixes what that PR's last
review left open. A files run now refuses validation creatures that duplicate
each other: identical paired differences gave a zero-width interval, so one
number could decide the verdict. The example's usage header and
`docs/local-rules.md` section 8 now match the code. The paper's full method
could not be read from the sandbox, and the code stays until the owner's
representative run decides it, so some criteria remain partial.

Part of #268

## Spec

### Intent and Rationale

- The issue asks for a controlled comparison, not a feature. The harness is
  therefore a library module plus an example runner, and it never proposes
  creatures to the optimiser.
- Sparse backprop reuses Lamarck's own `LearningPlan`, `apply_learnings` and
  `calculate_learning_rate`. The baseline is the crate's real refinement path,
  not a re-implementation.

### Essential Design Decisions

- **Budget.** The cost unit is one forward pass over one record. A forward pass
  plus one update costs 2. The amortised meta-search share is added to both
  primary baselines' budgets. The two `-app-only` arms get the symbolic arm's
  application budget alone, are reported, and never decide the verdict.
- **Residual.** It is the output error on output neurons only (`local_residual`).
  Hidden neurons get no residual term, because output error is not local
  hidden-neuron information.
- **No success claim without the owner.** A synthetic report is always
  `not-claimable-synthetic`. A files run that earns its place is only
  `pending-owner-review`.
- **Failed and duplicate runs.** A comparison with fewer than `minPairedTasks`
  (8) pairs, or with any failed run behind it, is `inconclusive`. A files run
  refuses duplicate or calibration-equal validation creatures, and refuses to
  run without an explicit `OUT_DIR`, so a private run cannot overwrite the
  public evidence.

### Undiscoverable Facts

- arXiv 2611.12248 could not be fetched from the sandbox. This run's
  `WebFetch https://arxiv.org/abs/2611.12248` returned `proxy refused the
  connection`, and PR #269's attempt got `CONNECT tunnel failed, response 403`.
  `docs/local-rules.md` section 2 records only a secondary outline and marks the
  method **not pinned**.
- The worker's pre-push gate replays the CI cargo steps. It reads
  `dtolnay/rust-toolchain@<sha>` as `cargo +<sha>`, which exits 101 on a host
  without rustup. This run's push was refused with
  `[PRE_PUSH_BLOCKED] Rust toolchain unavailable here: cargo +29eef336… --version exited 101`.
  The gate's per-run spec (`pre-push-spec.json`) was built from the workflows
  at run start, so the local-action workaround in this diff cannot unblock this
  run's own push. It does change the workflows a fresh spec is built from. The
  open sibling PRs #270 and #271 carry the same workaround.

## Evidence

Backend/CLI research harness. No UI files are touched.

- `docs/evidence/local-rules/manifest.json` and `summary.json` hold the
  predeclared synthetic run. This run regenerated them with
  `cargo run --release --example local_rules_bench` at crate 0.1.47, and they
  are byte-identical to PR #269's (`cmp`).
- The result is `overallVerdict: negative`, with `successClaim:
  not-claimable-synthetic` and coverage of 360 planned, 360 completed and 0
  failed. In the `all` group (n = 24 paired tasks, held-out MSE, symbolic
  minus baseline, bootstrap 95% CI), both primary baselines are worse for the
  rules at every K:

| K | vs sparse-backprop | vs derivative-free |
|---|---|---|
| 5 | +0.256 [0.128, 0.395] worse | +0.268 [0.136, 0.411] worse |
| 10 | +0.149 [0.066, 0.245] worse | +0.161 [0.076, 0.260] worse |
| 20 | +0.088 [0.047, 0.132] worse | +0.100 [0.057, 0.146] worse |

- Without the meta share (the `-app-only` arms), the rules come out better, and
  the CI includes 0 for `derivative-free-app-only` at K = 20. So the negative
  verdict comes from the cost of discovering the rules, which is the cost the
  issue says to count.

```mermaid
flowchart LR
    M[Manifest::predeclared] --> C[Calibration tasks]
    C --> S[discover: seeded sparse-rule search, cost charged]
    S --> V[Untouched validation tasks: new seeds and widths, held-out records]
    V --> A1[symbolic K steps]
    V --> A2[sparse backprop: K budget + meta share]
    V --> A3[derivative-free: K budget + meta share]
    A1 & A2 & A3 --> P[paired differences + bootstrap CI]
    P --> O[overallVerdict + successClaim]
```

## Acceptance Criteria

<!-- vibe-spec-review inputs="diff+issue-body" -->

- **met** — Discover sparse polynomial rules from input, output, weight, local residual where legitimately available, and inexpensive products, per activation family — evidence: `lamarck/src/local_rules/rule.rs`, `lamarck/src/local_rules/search.rs`, `lamarck/tests/local_rules.rs::delta_rule_lowers_training_mse` — reviewer: met
- **met** — Explicitly define how any residual is obtained: output error is not automatically local hidden-neuron information — evidence: `lamarck/tests/local_rules.rs::local_residual_is_output_only`, `lamarck/tests/local_rules.rs::residual_only_rule_leaves_hidden_synapses_and_biases_alone`, `docs/local-rules.md` section 4 — reviewer: met
- **met** — Compare 5–20 update steps with sparse backpropagation and derivative-free correction under equal total budget including meta-search — evidence: `lamarck/tests/local_rules.rs::baselines_get_the_symbolic_budget_plus_the_meta_share` — reviewer: met
- **met** — Test transfer to new creatures/widths and held-out data — evidence: `lamarck/src/local_rules/manifest.rs` (disjoint seeds, widths 2/8 against calibration width 4), `docs/evidence/local-rules/summary.json` — reviewer: met
- **met** — Keep as isolated experiment; canonical implementations belong in the existing training engine — evidence: `lamarck/src/lib.rs` (`pub mod local_rules`, not called by `run_optimisation` or the CLI) — reviewer: met
- **missing** — Read and pin the full method before reproducing it — reviewer: partial — reason: the paper cannot be fetched from the sandbox (`proxy refused the connection`). Section 2 records a secondary outline and says the method is not pinned, so nothing is pinned and I rate it missing rather than partial.
- **partial** — First inspect current code, merged work and repository instructions. Reuse existing mechanisms — evidence: `lamarck/src/local_rules/baselines.rs` (sparse backprop through the crate's `LearningPlan` and `apply_learnings`) — reviewer: partial — reason: the derivative-free arm is a new (1+1) search, not the crate's signed-perturbation candidate path.
- **met** — Opt-in experiment only; preserve ordinary behaviour, interfaces, immutable parents and full authoritative acceptance. No automatic deployment or default change — evidence: `lamarck/tests/local_rules.rs::parent_is_unchanged_by_every_arm` — reviewer: met
- **met** — Predeclare controls, seeds, sample size, compute limits and success/failure criteria — evidence: `docs/evidence/local-rules/manifest.json`, `lamarck/tests/local_rules.rs::manifest_predeclares_the_minimum_paired_sample`, `lamarck/tests/local_rules.rs::files_run_refuses_duplicate_validation_creatures` — reviewer: met
- **partial** — Identify code/model/data versions — evidence: `versions` in `summary.json` (crate version, neat-core tag, task-generator version), plus the files-mode corpus key and fingerprints — reviewer: partial — reason: no commit SHA is recorded, so builds at the same crate version cannot be told apart.
- **met** — separate calibration from untouched validation and count all preparation/search costs — evidence: `lamarck/tests/local_rules.rs::baselines_get_the_symbolic_budget_plus_the_meta_share` — reviewer: met
- **met** — Provide deterministic positive/negative correctness tests for new behaviour and pass the repository's normal quality checks — evidence: `lamarck/tests/local_rules.rs` (29 tests), and `./quality.sh` passed on the final head — reviewer: met
- **met** — Research runs are separate from routine feature CI — evidence: `lamarck/examples/local_rules_bench.rs`; the tests run only a reduced manifest — reviewer: met
- **met** — Deliver a reproducible command/manifest and machine-readable results with paired distributions, uncertainty, failures and actual coverage — evidence: `docs/evidence/local-rules/summary.json` (`pairedDifferences`, CIs, `failures`, `coverage`), `lamarck/tests/local_rules.rs::reduced_manifest_is_reproducible_and_fully_covered` — reviewer: met
- **met** — Fixtures alone do not demonstrate an optimisation benefit — evidence: `lamarck/tests/local_rules.rs::synthetic_source_never_supports_a_success_claim` — reviewer: met
- **met** — Record negative findings — evidence: `docs/local-rules.md` section 7, `CHANGELOG.md`, `README.md` — reviewer: met
- **partial** — remove optional code that does not earn its maintenance cost — evidence: `docs/local-rules.md` section 8 decision rule — reviewer: partial — reason: the synthetic verdict is negative, but removal waits for the owner's representative files-mode run, which needs a private corpus.
- **met** — Do not claim success before representative runs — evidence: `lamarck/tests/local_rules.rs::files_report_claim_follows_its_source_and_verdict` — reviewer: met
- **met** — Keep public implementation and fixtures application-agnostic; private corpora/results and operational integration remain private — evidence: `lamarck/tests/local_rules.rs::out_dir_defaults_for_synthetic_mode_only`, `lamarck/tests/local_rules.rs::files_run_completes_and_names_its_source` — reviewer: met
- **unrequested** — Local `.github/actions/rust-toolchain` action used by `ci.yml` and `cargo-quality.yml`, a plain `rustup` toolchain step in `family-sync.yml`, a `CANONICAL_PATHS` fallback checked by rule 17 in `scripts/check-family-sync-workflow.sh` (two cases in `scripts/test-check-family-sync-workflow.sh`), and the CHANGELOG Fixed entry for them — reviewer: unrequested — reason: the worker's pre-push gate refuses every Rust push on these workflows (Undiscoverable Facts). CI still compiles with the `rust-toolchain.toml` pin.
- **unrequested** — Crate version 0.1.46 → 0.1.47 in `lamarck/Cargo.toml` and `Cargo.lock` — reviewer: unrequested — reason: the repository bumps on any `lamarck/src` change, and the committed evidence is stamped with 0.1.47.
- **unrequested** — The `-app-only` diagnostic arms and files mode — reviewer: unrequested — reason: the app-only arms show what the meta-search cost does to the verdict, and files mode is the private representative run the issue asks for. Neither decides the synthetic verdict.

## Standards Review

<!-- vibe-standards-review inputs="diff+CODING-STANDARDS.md" -->

- **clean** — No `CODING-STANDARDS.md` in this repository, so the reviewer used `CONTRIBUTING.md` and the README's development rules and safety invariants. It reported no violations. It checked the version bump (ahead of base), CHANGELOG under `[Unreleased]` matching `summary.json`, and the review-checked rules: SHA-pinned `uses:` (kept inside the local action), canonical `runlib.sh` and `family-pins.sh` untouched, TDD, no new dependency, streaming training-data reads, no duplication of core (backprop reuse), and parents never mutated. Optional notes: one over-long CHANGELOG line, and inconsistent blank-line spacing between the CHANGELOG entries.

## Test Plan

- Added `lamarck/tests/local_rules.rs` (29 tests, public API only, deterministic).
  It covers rule refusals, manifest refusals, the residual being output-only,
  parent immutability, derivative-free never raising training MSE, the verdict
  rule (`predeclared_verdict_follows_the_success_criterion`,
  `predeclared_verdict_ignores_other_groups_and_app_only_arms`), budget
  accounting, the minimum paired sample, failed runs
  (`a_failed_validation_run_is_recorded_and_never_decides_the_verdict`), the
  files-mode refusals and the success-claim rules.
- New this run: `lamarck/tests/local_rules.rs::files_run_refuses_duplicate_validation_creatures`.
  It refuses `[c2, c2]` ("validation creatures 0 and 1 share a fingerprint…")
  and `[c2, c3, c2]` ("0 and 2"), each with the minimum lowered to 2, and accepts
  `[c2, c3]` with n = 2. Before the fix it failed: `run_files` returned
  `Ok(Report …)` with identical fingerprints and `ci_low == ci_high`.
- `scripts/test-check-family-sync-workflow.sh` gains two rule-17 cases (missing
  fallback, stale fallback), and both expect exit 1. No existing assertion was
  removed or changed.
- `cargo test -p neat_ai_lamarck --test local_rules`: 29 passed.
- `./quality.sh < /dev/null` on the final head: passed ("All quality checks
  passed!").

**Owner step:** run files mode on a representative private corpus with at least
8 distinct validation creatures, keeping `OUT_DIR` private
(`docs/local-rules.md` section 8). If the run is not `earns-its-place`, remove
`lamarck/src/local_rules/`, the example, its tests and the evidence.

**Not done or not checked:** the paper's method is not pinned (no network
access to arXiv). No representative corpus was run. Baseline hyperparameters
(backprop learning rate 0.01, derivative-free sigma 0.05) were not tuned. This
run could not push its own branch because of the pre-push gate fault above.
