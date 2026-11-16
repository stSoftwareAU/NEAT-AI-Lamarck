# PR summary — issue #268

## Summary

A documentation-only follow-up to the merged harness (PRs #272, #277 and #278).
It adds two paper facts from 16 November 2026 search snippets to
`docs/local-rules.md` section 2. A training episode stops as soon as the
max-norm of any greedily updated weight leaves chosen safety bounds. Each
learned rule is tested by training N_eval = 128 copies of the same network. The
"Not pinned" list is narrowed to match, still labelled as known only from
snippets, and the latest failed attempt to fetch the paper is logged.

No code, test or evidence file changes. The paper still cannot be read from the
sandbox, so the full method stays unpinned. The keep-or-remove decision on the
harness still waits on the owner's representative files-mode run (section 8):
no creature or corpus for it exists on this host.

Part of #268

## Spec

### Intent and Rationale

- The issue asks to read and pin the full method before reproducing it. The
  paper cannot be fetched here, so each fact that search snippets do give is
  recorded, which narrows what a later run must still pin.

### Essential Design Decisions

- The new facts sit in the existing "secondary, not a pin" outline, and each
  names its snippet source and date inline. Section 2 still says the harness is
  not a faithful reproduction.
- The evaluation-sample bullet contrasts the paper's 128 copies with this
  harness's one paired difference per validation task (`minPairedTasks` = 8),
  rather than changing the predeclared protocol after the results.
- The harness code is kept. Section 8's rule removes it only if the owner's
  representative run is also not `earns-its-place`, and removing it now would
  delete the reproducer that run needs.

### Undiscoverable Facts

- On 16 November 2026, `curl https://arxiv.org/abs/2611.12248` gave
  `curl: (56) CONNECT tunnel failed, response 403`. `curl` of the arXiv PDF,
  `export.arxiv.org`, `r.jina.ai`, Semantic Scholar and Hugging Face pages got
  no HTTP response (`000`), and WebFetch of
  `https://arxiv.org/html/2611.12248v1` gave "proxy refused the connection".
  `curl https://github.com` succeeded, so only the paper hosts are refused.
- The two new facts come from a WebSearch result summary, not from the paper.
  No bound values were given.
- A search of this host (`find / -xdev`, excluding build and VCS directories)
  found no creature JSON or training corpus. The public JSON files in the
  NEAT-AI family repositories (`NEAT-AI/test/data/`,
  `NEAT-AI-core/neat-core/tests/fixtures/`) are test fixtures. A files run on
  them would be the fixture evidence the issue says does not demonstrate a
  benefit, so none was run.

## Evidence

Docs-only change; no behaviour changed. Checked:

- `markdownlint-cli2 docs/local-rules.md`: 0 issues.
- `codespell docs/local-rules.md`: no findings.
- The `minPairedTasks` = 8 figure matches
  `docs/evidence/local-rules/manifest.json` and the section 6 table.
- Issue and PR numbers this diff cites: none new. The section already cites
  #276: Experiment: Test sparse symbolic local update rules against current
  refinement (Issue #268).

## Acceptance Criteria

The issue's "Execution and acceptance" list is not an acceptance-criteria
heading, so no reviewer ran. The parts still undone:

- **missing** — Read and pin the full method before reproducing it — reason: the sandbox proxy refuses arXiv and its mirrors (403, `000`, "proxy refused the connection", retried on 16 November 2026). The threshold and safety-bound values, B, b, J, the variable set per degree, the outer optimiser, the widths, and the other tasks' depths and datasets are still unknown.
- **missing** — Record negative findings and remove optional code that does not earn its maintenance cost. Do not claim success before representative runs. — reason: the synthetic negative finding is recorded and no success is claimed, but the owner's representative files-mode run has not happened. The section 8 removal decision depends on it, and no representative creature or corpus exists on this host.

## Test Plan

- No test added or changed: the diff changes documentation only.
- `markdownlint-cli2 docs/local-rules.md` and `codespell docs/local-rules.md`
  were run on the final doc: 0 issues, no findings.
- `./quality.sh < /dev/null` was run after the last doc edit: exit 0, "All
  quality checks passed!" (clippy, tests, rustdoc, codespell, markdownlint).

## Not done

- The paper's full method is not pinned: the sandbox proxy refuses arXiv and
  its mirrors.
- The owner's representative files-mode run (section 8), and the
  keep-or-remove decision it drives, are still pending.
- The per-task horizons from the PR #276 review were not checked against the
  paper.
