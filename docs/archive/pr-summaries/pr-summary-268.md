# PR summary — issue #268

## Summary

This PR adds an opt-in, offline experiment for issue #268. It finds sparse
symbolic local update rules for each activation family on calibration tasks,
then compares them with Lamarck's own sparse backprop and with a
derivative-free (1+1) search. Validation uses untouched tasks, and every arm
gets the same total record budget, including the share of the rule search. The
predeclared synthetic verdict is **`negative`**. Nothing in the optimiser, the
CLI or any default changes. Two parts are still open: the representative run on
a private creature and corpus, and pinning the paper's method. Part of #268.

## Spec

### Intent and Rationale

- The issue asks whether discovered local rules can stand in for current refinement. The harness answers that under a fixed protocol: seeds, sample size, limits and the verdict rule are set in `Manifest::predeclared()` before any run.
- The sparse-backprop arm drives Lamarck's existing `LearningPlan`, `apply_learnings` and `calculate_learning_rate` with `sparse_ratio`. Current refinement is therefore what is measured, not a second copy of it.

### Essential Design Decisions

- **Residual:** only an output neuron `k` has one, and it is `expected[k] − activation`. A hidden neuron has none, and features that use `r` are 0 there. Output error is never treated as local hidden-neuron information (`local_residual` in `lamarck/src/local_rules/rule.rs`).
- **Budget:** cost is counted in record units. One forward pass is 1 unit, a forward pass plus an update is 2, and held-out measurement is never charged. Every unit spent in the meta-search is charged and amortised over the 24 validation tasks. The primary arms get the symbolic arm's cost plus that share. The `-app-only` arms leave the share out and are diagnostics, not part of the verdict.
- **Backprop settings:** the sparse-backprop arm uses Lamarck's default `BackpropConfig` learning rate (0.01). The rule step uses `eta` = 0.05. The app-only "better" rows may partly reflect that difference, which is one reason they do not count.
- **Keeping the code:** the harness stays only as the reproducer until the owner's representative files-mode run. If that run is not `earns-its-place`, `lamarck/src/local_rules/`, the example, its tests and its evidence are removed. This is an assumption: the issue asks to remove code that does not earn its cost, but also says not to judge from fixtures alone.

### Undiscoverable Facts

- arXiv 2611.12248 could not be read from the sandbox on 14 November 2026. Both `curl https://export.arxiv.org/abs/2611.12248` and `curl https://arxiv.org/pdf/2611.12248` failed with `CONNECT tunnel failed, response 403`. What is tested is the issue's description of the method.
- The 13 November research shortlist the issue cites is not in this repository.

## Evidence

Backend/research change only; no UI files.

```mermaid
flowchart LR
    M[Manifest::predeclared] --> C[8 calibration tasks<br/>train records only]
    C --> D[discover: sparse rules per family<br/>meta cost charged]
    M --> V[24 untouched validation tasks<br/>same-width + new-width, held-out records]
    D --> S[symbolic K = 5, 10, 20]
    V --> S
    V --> B[sparse backprop<br/>equal total budget]
    V --> F[derivative-free<br/>equal total budget]
    S --> P[paired held-out MSE differences<br/>bootstrap 95% CI]
    B --> P
    F --> P
    P --> R[overallVerdict]
```

Command: `cargo run --release --example local_rules_bench`. It writes
`docs/evidence/local-rules/manifest.json` and `summary.json`. Coverage is 360 of
360 planned runs completed, with 0 failures. The meta-search cost 1,622,016
record units. Versions: crate 0.1.46, neat-core v0.23.4, task generator
`local-rules-synthetic-v1`. The rules found were: identity `R*1 + XR*2`;
logistic `R*2 + W*-2 + XYY*2`; tanh `XYY*-0.5 + YR*-1 + Y*-2`.

In the table, each difference is symbolic minus baseline held-out MSE, so a
negative number means symbolic did better. n = 24 paired tasks per row.

| Arm | K | Mean diff | 95% CI | Verdict |
|---|---|---|---|---|
| sparse-backprop (primary) | 5 | +0.2563 | [0.1276, 0.3948] | worse |
| derivative-free (primary) | 5 | +0.2684 | [0.1360, 0.4108] | worse |
| sparse-backprop (primary) | 10 | +0.1490 | [0.0661, 0.2453] | worse |
| derivative-free (primary) | 10 | +0.1610 | [0.0759, 0.2599] | worse |
| sparse-backprop (primary) | 20 | +0.0876 | [0.0472, 0.1316] | worse |
| derivative-free (primary) | 20 | +0.0996 | [0.0570, 0.1461] | worse |
| sparse-backprop-app-only | 5 / 10 / 20 | −0.3726 / −0.3781 / −0.3198 | all below 0 | better (diagnostic) |
| derivative-free-app-only | 5 / 10 / 20 | −0.1987 / −0.1655 / −0.0342 | last straddles 0 | better, better, inconclusive (diagnostic) |

Once the search cost is counted, the rules lose to both baselines at every K.
That is a negative finding on synthetic fixtures only, and it proves no benefit
or harm on real creatures. The method write-up is `docs/local-rules.md`. The
`CHANGELOG.md` and `README.md` entries link to it.

## Acceptance Criteria

- **missing** — "Do not claim success before representative runs" — reason: the representative run needs the private production creature and corpus, which are not in this repository. Owner step: run `cargo run --release --example local_rules_bench -- OUT_DIR --calibration-creature PATH --creature PATH --training-data DIR` on a quiet machine, keep the output private, and remove the harness unless the verdict is `earns-its-place`.
- **missing** — "Read and pin the full method before reproducing it" — reason: arXiv 2611.12248 could not be reached from the sandbox (CONNECT tunnel 403), so what is tested is the issue's description and the paper's method is not pinned.

## Test Plan

- Added `lamarck/tests/local_rules.rs` with 16 tests:
  - Positive: `delta_rule_lowers_training_mse`.
  - Hidden-residual negative, with an output-synapse positive control: `residual_only_rule_leaves_hidden_synapses_and_biases_alone`.
  - Immutable parents: `parent_is_unchanged_by_every_arm`.
  - Refusals, each with an accepted control: `sparse_rule_refusals_name_their_cause` and `manifest_refusals_name_their_cause`.
  - Baselines and statistics: `derivative_free_never_raises_training_mse`, `matched_count_arithmetic`, `verdict_from_confidence_interval` and `local_residual_is_output_only`.
  - Determinism and full coverage: `reduced_manifest_is_reproducible_and_fully_covered`.
  - Version pin: `neat_core_tag_matches_cargo_manifest`.
  - Files mode:
    - `files_run_completes_and_names_its_source`: no temp path is written to the summary.
    - `files_run_refuses_a_short_corpus_and_accepts_a_full_one`.
    - `files_run_refuses_the_calibration_creature_as_validation`.
    - `files_run_refuses_no_validation_creatures_and_mismatched_widths`.
  - Report contents: `synthetic_summary_carries_the_search_trace_and_source`.
- `cargo test --test local_rules`: 16 passed.
- `./quality.sh < /dev/null` on the head: exit 0, "All quality checks passed!". The first run failed `readme_contract::repository_layout_lists_every_lamarck_src_rs` because `local_rules` was missing from the README layout tree. The line was added and the gate re-run.
- No existing test was edited or removed.
- Not done: the representative private-corpus run, and pinning the paper's method (both above).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
