# PR summary — issue #268

## Summary

A documentation-only follow-up to the merged harness (PRs #272 and #277). It
adds paper facts from 16 November 2026 search snippets to `docs/local-rules.md`
section 2. The classification network has three layers. At degree 2, layer 1
sees only itself and layer 2, while layers 2 and 3 see the whole downstream
graph. The learned classification terms include the target-loss term `Ŷ·Y3`, a
logit-alignment term `Ŷᵀ·W3·Y2` and a squared cross-entropy term in layer 2. The
section also notes that the Papers-with-Code mirror's abstract claims accuracy
comparable to backprop, which the arXiv v1 abstract does not. A PDF snippet
adds that the outer loop runs until the global loss falls below a chosen
threshold. The section records the latest failed attempts to fetch the paper,
and the "Not pinned" list is narrowed to match, still labelled as known only
from snippets.

No code, test or evidence file changes. The paper still cannot be read from the
sandbox, so the full method stays unpinned (reviewer: missing). The
keep-or-remove decision on the harness still waits on the owner's representative
files-mode run (section 8), so that criterion stays partial.

Part of #268

## Spec

### Intent and Rationale

- The issue asks to read and pin the full method before reproducing it. The
  paper cannot be fetched here, so each fact that search snippets do give is
  recorded, which narrows what a later run must still pin.

### Essential Design Decisions

- The new facts sit in the existing "secondary, not a pin" outline, and each
  names its source inline. Section 2 still says the harness is not a faithful
  reproduction.
- The Papers-with-Code accuracy claim is recorded but not relied on, because
  the arXiv v1 abstract does not make it.
- The harness code is kept. Section 8's rule removes it only if the owner's
  representative run is also not `earns-its-place`, and removing it now would
  delete the reproducer that run needs.

### Undiscoverable Facts

- On 16 November 2026, `curl` of `https://arxiv.org/abs/2611.12248`,
  `https://arxiv.org/html/2611.12248v1` and the `export.arxiv.org` API query
  each gave `curl: (56) CONNECT tunnel failed, response 403`. WebFetch of
  `https://www.alphaxiv.org/abs/2611.12248` and
  `https://paperswithcode.co/paper/127548` gave "proxy refused the connection".
  A final retry the same day, `curl` and WebFetch of
  `https://arxiv.org/pdf/2611.12248v1`, failed the same two ways. Two further
  web searches for the outer optimiser, its learning rate, b and the widths
  found none of them in the indexed snippets.
- The new facts come from WebSearch result summaries of the paper's HTML and
  PDF versions, not from the paper itself. The loss-threshold stopping rule
  comes from a PDF snippet, and its threshold value is not given.

## Evidence

Docs-only change; no behaviour changed. Checked:

- `markdownlint-cli2 docs/local-rules.md`: 0 issues.
- `codespell docs/local-rules.md`: no findings.
- Issue and PR numbers this diff cites: #276: Experiment: Test sparse symbolic
  local update rules against current refinement (Issue #268). This is already
  cited in the section; the diff adds no new number.

## Acceptance Criteria

<!-- vibe-spec-review inputs="diff+issue-body" -->

- **missing** — Read and pin the full method before reproducing it — evidence: `docs/local-rules.md` section 2 ("Not pinned") — reviewer: missing — reason: the paper was never read. The sandbox proxy refuses arXiv and every mirror (403 / "proxy refused the connection", retried again on 16 November 2026), and section 2 says none of the outline is a pin. This diff only adds more secondary snippets. B, b, J, the loss threshold, the variable set per degree, the outer optimiser, widths and the other tasks' depths and datasets are still unknown.
- **met** — First inspect current code, merged work and repository instructions. Reuse existing mechanisms; if functionality already exists, this issue still owns the missing controlled comparison rather than a second implementation. — evidence: `lamarck/src/local_rules/baselines.rs` (calls `LearningPlan`, `apply_learnings`, `BackpropConfig`, `crate::mirror`) — reviewer: met
- **met** — Opt-in experiment only; preserve ordinary behaviour, interfaces, immutable parents and full authoritative acceptance. No automatic deployment or default change. — evidence: `lamarck/src/lib.rs` (`pub mod local_rules;` is the only reference outside the module), `lamarck/tests/local_rules.rs::parent_is_unchanged_by_every_arm` — reviewer: met
- **met** — Predeclare controls, seeds, sample size, compute limits and success/failure criteria. Identify code/model/data versions; separate calibration from untouched validation and count all preparation/search costs. — evidence: `docs/evidence/local-rules/manifest.json`, `lamarck/tests/local_rules.rs::baselines_get_the_symbolic_budget_plus_the_meta_share` — reviewer: met
- **met** — Provide deterministic positive/negative correctness tests for new behaviour and pass the repository's normal quality checks. Research runs are separate from routine feature CI. — evidence: `lamarck/tests/local_rules.rs`; `./quality.sh` on the head (see Test Plan) — reviewer: met
- **met** — Deliver a reproducible command/manifest and machine-readable results with paired distributions, uncertainty, failures and actual coverage. Fixtures alone do not demonstrate an optimisation benefit. — evidence: `lamarck/examples/local_rules_bench.rs`, `docs/evidence/local-rules/summary.json` — reviewer: met
- **partial** — Record negative findings and remove optional code that does not earn its maintenance cost. Do not claim success before representative runs. — evidence: `docs/local-rules.md` sections 7–8, `docs/evidence/local-rules/summary.json` (`overallVerdict: negative`, `successClaim: not-claimable-synthetic`) — reviewer: partial — reason: the negative finding is recorded and no success is claimed, but the optional code is not removed. Section 8's decision rule defers removal to the owner's private representative files-mode run, which has not happened, and removing the harness first would delete the reproducer that run needs.
- **met** — Keep public implementation and fixtures application-agnostic; private corpora/results and operational integration remain private. — evidence: `lamarck/tests/local_rules.rs::out_dir_defaults_for_synthetic_mode_only`, `lamarck/tests/local_rules.rs::files_run_completes_and_names_its_source` — reviewer: met
- **met** — Leave unassigned until an implementer actively starts. `work-on` authorises the investigation, not skipping prerequisite verification. — evidence: `gh issue view 268 --json assignees` shows only `VibeCoderST`, the worker that is actively implementing this run — reviewer: missing — reason: the reviewer said "not assessable from the diff", a process item with no code footprint; I checked the issue directly.

## Standards Review

<!-- vibe-standards-review inputs="diff+CODING-STANDARDS.md" -->

- **clean** — The repository has no `CODING-STANDARDS.md`, so the reviewer used `CONTRIBUTING.md` and `README.md`. It checked version bumping (docs-only, no bump needed), the changelog (the existing #268 entry covers it), markdownlint (MD013 off), codespell, and build docs (untouched). It found no review-only rules and no new claims about the repository. It raised optional points: the outline header's "from monomial order onwards" scope, and the unstated source of the mirror-abstract bullet. Both are fixed in this diff, and the spec reviewer's matching finding is fixed too.

## Test Plan

- No test added or changed: the diff changes documentation only.
- `markdownlint-cli2 docs/local-rules.md` and `codespell docs/local-rules.md`
  were run on the final doc: 0 issues, no findings.
- A spec reviewer sub-agent re-reviewed the final diff for the two open
  criteria (verdicts above). It also flagged the self-contradictory
  "aligned … opposite sign" wording, now reworded, and the missing PDF source
  and retry in Undiscoverable Facts, now added.
- `./quality.sh < /dev/null` was run after the last doc edit: exit 0, "All
  quality checks passed!" (clippy, tests, rustdoc, codespell, markdownlint).

## Not done

- The paper's full method is not pinned: the sandbox proxy refuses arXiv and
  its mirrors.
- The owner's representative files-mode run (section 8), and the
  keep-or-remove decision it drives, are still pending.
- The per-task horizons from the PR #276 review and the 16 November snippets
  were not checked against the paper.
