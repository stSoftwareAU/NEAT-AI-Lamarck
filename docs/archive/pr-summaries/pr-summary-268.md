# PR summary — issue #268

## Summary

The harness, its tests and its synthetic run were delivered by #272, which is
already merged (the synthetic verdict is negative). This PR covers what was
still open. It retries reading the paper (arXiv 2611.12248) and records what
newer web-search snippets say about the method in `docs/local-rules.md`
section 2. The paper itself is still unreachable from the sandbox, so the
method is still not pinned. The keep-or-remove decision on the harness still
waits for the owner's representative run. Documentation only; no code changes.

Part of #268

## Spec

### Intent and Rationale

- Two criteria were left open after #272: pinning the paper's full method, and
  removing optional code that does not earn its keep. This run could only
  narrow the first one, so it records what it found rather than claiming either.

### Essential Design Decisions

- Every new detail is marked secondary (from search snippets of the paper's
  HTML page), and the "Not pinned" paragraph says the paper is still unread.
  Search output is not a reading of the paper.
- The harness is kept. Its synthetic run is negative, but the representative
  files-mode run it exists for has not happened, and removing the harness now
  would take away the tool that run needs.

### Undiscoverable Facts

- On 15 November 2026 the sandbox proxy refused arXiv again (`curl`:
  `CONNECT tunnel failed, response 403`; WebFetch of the abstract, HTML,
  alphaxiv and Hugging Face pages: "proxy refused the connection"). Scholar
  Archive, ResearchGate, OpenReview, Papers with Code and the Wayback Machine
  also gave no HTTP response through the proxy (`000`). A later retry the same
  day also failed: `curl` of the arXiv abstract and PDF, `export.arxiv.org`, the
  Semantic Scholar page and API, and an `r.jina.ai` reader proxy gave `000`, and
  WebFetch of the PDF gave "proxy refused the connection". GitHub and PyPI were
  reachable. `gh search repos "symbolic local learning rules"` found no code
  released by the authors.
- New snippet facts: v1 (also listed among NeurIPS 2026 papers), monomial order
  K = 4, an unrolled outer loop, an inner horizon T that rises to 100 over 6,000
  episodes with jitter J, and a transfer test on held-out Sliced MNIST with
  different layer sizes. The issue's 5–20 step window is shorter than that
  horizon, and the doc now says so.

## Evidence

Documentation-only change. No UI, code, test or evidence file is touched.

- `docs/local-rules.md` section 2 now has: the 15 November retry and its
  errors, the new snippet-derived outline bullets, a narrowed "Not pinned"
  list, and a comparison-table row for the training horizon. It also notes that
  snippets suggest, without establishing, that short horizons are an
  unfavourable regime for the paper's rules. The row's
  `steps` [5, 10, 20] and `calibration_steps` 10 were checked against
  `docs/evidence/local-rules/manifest.json`.
- `docs/local-rules.md` section 2 also records the later 15 November retry
  (`export.arxiv.org`, Semantic Scholar, `r.jina.ai`, WebFetch of the PDF).
- Unused-code audit of the harness: every public item in
  `lamarck/src/local_rules/` is used by the module, the example or the tests,
  or appears in a public type or signature. `cargo clippy --all-targets` reports
  no dead code, so nothing smaller than the whole harness can be removed.
- Sources: web search results pointing at
  [arXiv abs](https://arxiv.org/abs/2611.12248),
  [HTML](https://arxiv.org/html/2611.12248) and
  [PDF](https://arxiv.org/pdf/2611.12248). Those pages themselves could not be
  fetched.

## Acceptance Criteria

<!-- vibe-spec-review inputs="diff+issue-body" -->

- **missing** — Read and pin the full method before reproducing it — reviewer: missing — reason: the sandbox proxy still refuses arXiv 2611.12248 and every mirror tried, so the paper was not read. `docs/local-rules.md` section 2 holds only a secondary outline and an explicit "Not pinned" list.
- **partial** — Record negative findings and remove optional code that does not earn its maintenance cost — evidence: `docs/local-rules.md` sections 7–8, `docs/evidence/local-rules/summary.json` — reviewer: partial — reason: the negative synthetic finding is recorded and the audit found no dead code inside the harness. Removing `lamarck/src/local_rules/` waits on the owner's representative files-mode run (section 8 decision rule), which needs a private corpus not available here.

## Test Plan

- No test is added or changed: the diff touches documentation only.
- `markdownlint-cli2 docs/local-rules.md` and `codespell docs/local-rules.md`
  reported no issues after the edits.
- `cargo clippy --all-targets` reported no dead code in the harness.
- `./quality.sh < /dev/null` passed on the final edit ("All quality checks
  passed!").
