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
  also gave no HTTP response through the proxy (`000`). GitHub and PyPI were
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
  short horizons are an unfavourable regime for the paper's rules. The row's
  `steps` [5, 10, 20] and `calibration_steps` 10 were checked against
  `docs/evidence/local-rules/manifest.json`.
- Sources: web search results pointing at
  [arXiv abs](https://arxiv.org/abs/2611.12248),
  [HTML](https://arxiv.org/html/2611.12248) and
  [PDF](https://arxiv.org/pdf/2611.12248). Those pages themselves could not be
  fetched.

## Acceptance Criteria

- **missing** — Read and pin the full method before reproducing it — reason: arXiv 2611.12248 and its mirrors are still refused by the sandbox proxy, so the paper was not read. `docs/local-rules.md` section 2 now holds a fuller secondary outline (K = 4, unrolled horizon up to 100, transfer setup) and lists what is still unpinned.
- **missing** — remove optional code that does not earn its maintenance cost — reason: the synthetic verdict is negative, but whether `lamarck/src/local_rules/` stays depends on the owner's representative files-mode run (`docs/local-rules.md` section 8), which needs a private corpus not available here.

## Test Plan

- No test is added or changed: the diff touches documentation only.
- `markdownlint-cli2 docs/local-rules.md` and `codespell docs/local-rules.md`
  reported no issues after the edit.
- `./quality.sh < /dev/null` passed on the final edit ("All quality checks
  passed!").
