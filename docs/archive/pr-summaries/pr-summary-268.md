# PR summary — issue #268

## Summary

Documentation-only follow-up to the merged harness (PR #272). It re-lands the
record the closed PR #276 added to `docs/local-rules.md` section 2, and fixes the
factual error that PR's review found. The section now also records the 15 and 16
November attempts to fetch the paper. It adds a secondary outline of the method
taken from search snippets: monomial order 4, unrolled meta-optimisation, a
transfer test on Sliced MNIST, and example learned terms. It now gives the
paper's training horizon per task, as the PR #276 review reported it: 500 steps
for classification, 300 for reconstruction and 100 for self-supervised learning.
The old text gave the self-supervised schedule (100 steps over 6,000 episodes)
as if it held for the whole paper. A new comparison row and closing paragraph
say that the issue's 5–20 step window is shorter than the final horizon reported
for each of the paper's tasks. So the negative synthetic verdict neither speaks
to the paper's longer-horizon regimes nor refutes its method.

No code, test or evidence file changes. The paper still could not be fetched
from the sandbox, so the full method is not pinned. The keep-or-remove decision
on the harness waits on the owner's representative files-mode run. Those two
criteria therefore stay partial.

Part of #268

## Spec

### Intent and Rationale

- PR #276 was closed after its review fix stalled. Its review found that the
  horizon bullet applied the self-supervised schedule to the whole paper. A
  reader planning a faithful comparison would then pick the wrong horizon for
  classification or reconstruction.
- The per-task horizons are attributed to that review, which read
  <https://arxiv.org/html/2611.12248v1>. They are not presented as read in this
  run, because the fetch failed here.

### Essential Design Decisions

- The paper's monomial order (which it calls K) is named "monomial order 4" in
  the doc. That keeps it apart from this harness's step count K = 5, 10, 20.
- The outline stays labelled secondary and "not a pin". Section 2 still says
  this harness is not a faithful reproduction.
- The harness code is kept. Section 8's decision rule removes it only if the
  owner's representative run is also not `earns-its-place`. Removing it now
  would delete the reproducer that run needs.

### Undiscoverable Facts

- On 16 November 2026, `curl https://arxiv.org/html/2611.12248v1` gave
  `curl: (56) CONNECT tunnel failed, response 403`. WebFetch of the same URL
  gave "proxy refused the connection".
- The per-task schedules (sections 3.1–3.3 of the paper) come from the PR #276
  review comment on GitHub, not from this repository.

## Evidence

Docs-only change; no behaviour changed. Checked:

- `markdownlint-cli2 docs/local-rules.md`: 0 issues.
- `codespell docs/local-rules.md`: no findings.
- `calibration_steps` = 10 and K = 5, 10, 20 in the new table row match
  `lamarck/src/local_rules/manifest.rs` and
  `docs/evidence/local-rules/manifest.json` (`"calibrationSteps": 10`).
- Issue and PR numbers this diff cites: #276: Experiment: Test sparse symbolic
  local update rules against current refinement (Issue #268). Its review is
  the source of the per-task horizons.

## Acceptance Criteria

<!-- vibe-spec-review inputs="diff+issue-body" -->

- **partial** — Read and pin the full method before reproducing it — evidence: `docs/local-rules.md` section 2 — reviewer: partial — reason: the paper could not be fetched from the sandbox (proxy 403). The outline is secondary, from snippets and the PR #276 review. B, b, J, the variable set per degree, the outer optimiser, widths, depths and datasets are still not pinned.
- **met** — First inspect current code, merged work and repository instructions. Reuse existing mechanisms; if functionality already exists, this issue still owns the missing controlled comparison rather than a second implementation. — evidence: `lamarck/src/local_rules/baselines.rs` (uses `LearningPlan`, `apply_learnings`, `crate::mirror`) — reviewer: met
- **met** — Opt-in experiment only; preserve ordinary behaviour, interfaces, immutable parents and full authoritative acceptance. No automatic deployment or default change. — evidence: `lamarck/tests/local_rules.rs::parent_is_unchanged_by_every_arm`, `docs/local-rules.md` section 1 — reviewer: met
- **met** — Predeclare controls, seeds, sample size, compute limits and success/failure criteria. Identify code/model/data versions; separate calibration from untouched validation and count all preparation/search costs. — evidence: `docs/evidence/local-rules/manifest.json`, `lamarck/tests/local_rules.rs::baselines_get_the_symbolic_budget_plus_the_meta_share` — reviewer: met
- **met** — Provide deterministic positive/negative correctness tests for new behaviour and pass the repository's normal quality checks. Research runs are separate from routine feature CI. — evidence: `lamarck/tests/local_rules.rs`; `./quality.sh` on the head (see Test Plan) — reviewer: met
- **met** — Deliver a reproducible command/manifest and machine-readable results with paired distributions, uncertainty, failures and actual coverage. Fixtures alone do not demonstrate an optimisation benefit. — evidence: `lamarck/examples/local_rules_bench.rs`, `docs/evidence/local-rules/summary.json` — reviewer: met
- **partial** — Record negative findings and remove optional code that does not earn its maintenance cost. Do not claim success before representative runs. — evidence: `docs/local-rules.md` sections 7–8, `docs/evidence/local-rules/summary.json` — reviewer: met — reason: the negative finding is recorded and `successClaim` is `not-claimable-synthetic`. But the decision to keep or remove the harness waits on the owner's private representative run (`docs/local-rules.md` section 8), which has not happened. The reviewer accepted keeping the code as a defensible reading. The removal half is still undecided, so I mark it unfinished rather than met.
- **met** — Keep public implementation and fixtures application-agnostic; private corpora/results and operational integration remain private. — evidence: `lamarck/tests/local_rules.rs::out_dir_defaults_for_synthetic_mode_only`, `lamarck/tests/local_rules.rs::files_run_completes_and_names_its_source` — reviewer: met
- **met** — Leave unassigned until an implementer actively starts. `work-on` authorises the investigation, not skipping prerequisite verification. — evidence: issue #268 has no assignee; the worker's lifecycle comment records cohort `unassigned` — reviewer: missing — reason: the reviewer said "not checkable from the diff", a process item. I checked the issue directly.

## Standards Review

<!-- vibe-standards-review inputs="diff+CODING-STANDARDS.md" -->

- **clean** — The repository has no `CODING-STANDARDS.md`, so the reviewer used `CONTRIBUTING.md`. It checked version bumping (docs-only, no bump needed), the changelog (the existing #268 entry covers it) and build docs (not duplicated), and found no review-only rules. Every repository fact the diff states (`calibration_steps` = 10, K = 5, 10, 20, the 100–500 range) matches the head. It raised three optional wording points: the section pointer for K, one overlong line, and a duplicated "A negative result here" sentence. All three were fixed in this diff.

## Test Plan

- No test added or changed: the diff changes documentation only.
- `markdownlint-cli2 docs/local-rules.md` and `codespell docs/local-rules.md`
  were run on the final doc: 0 issues, no findings.
- `./quality.sh < /dev/null` was run after the last doc edit: exit 0, "All
  quality checks passed!" (clippy, tests, rustdoc, codespell, markdownlint).

## Not done

- The paper's full method is not pinned: the sandbox proxy refuses arXiv.
- The owner's representative files-mode run (section 8), and the
  keep-or-remove decision it drives, are still pending.
- The per-task horizons from the PR #276 review were not checked against the
  paper in this run.
