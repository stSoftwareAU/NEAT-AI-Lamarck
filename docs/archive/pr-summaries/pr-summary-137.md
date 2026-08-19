# Fix broken image links in archived PR summary 86 (Issue #137)

## Summary

`docs/archive/pr-summaries/pr-summary-86.md` embedded the two rendered-diagram
screenshots with repo-root-relative paths (`docs/evidence/…`). Markdown resolves
a relative target against the file's *own* directory, so on GitHub both pointed
at `docs/archive/pr-summaries/docs/evidence/…`, which does not exist — the
light/dark rendering evidence for the #86 diagram conversion showed as two
broken images. Both are now `../../evidence/core-principle-mermaid-light.png`
and `../../evidence/core-principle-mermaid-dark.png`.

The root cause is general — paths written for the PR-description context (which
renders from the repo root) break when the summary is archived one or two
directories deep — so the fix is guarded by a repo-wide test rather than a
one-off assertion about these two lines. Closes #137.

## Evidence

No web interface to screenshot: this is a documentation path fix. The
regression test is the evidence.

Before the path change (`cargo test --test docs_link_targets`):

```text
running 2 tests
test link_targets_ignores_fenced_examples_but_finds_real_links ... ok
test every_relative_markdown_link_resolves_from_its_own_file ... FAILED

markdown links that do not resolve relative to their own file:
  docs/archive/pr-summaries/pr-summary-86.md -> docs/evidence/core-principle-mermaid-light.png
  docs/archive/pr-summaries/pr-summary-86.md -> docs/evidence/core-principle-mermaid-dark.png
```

After:

```text
running 2 tests
test link_targets_ignores_fenced_examples_but_finds_real_links ... ok
test every_relative_markdown_link_resolves_from_its_own_file ... ok

test result: ok. 2 passed; 0 failed
```

Full workspace suite passes (`cargo test --workspace --all-features --
--test-threads=2`), as do `cargo fmt --all -- --check`, `cargo clippy
--workspace --all-targets --all-features -D warnings` and `cargo deny check`.
`./quality.sh` stops at its codespell preflight because `codespell` is not
installed in this container (`spell-check: codespell is not installed.`) and
there is no `pip`/`pipx` to install it — unrelated to this change; every other
gate in the script was run individually and passed.

## Test Plan

Added `lamarck/tests/docs_link_targets.rs`:

- `every_relative_markdown_link_resolves_from_its_own_file` — walks every `.md`
  file in the repo (skipping `target/`, `.git/`, `node_modules/`), extracts each
  `[...](target)` outside fenced code blocks, skips external URLs and anchors,
  and asserts the remainder exists relative to the containing file's directory.
  Fails on the unfixed tree with exactly the two `pr-summary-86.md` paths.
- `link_targets_ignores_fenced_examples_but_finds_real_links` — pins the
  extractor itself: link titles stripped, fenced blocks and inline code spans
  ignored, anchored and external targets still returned so the classifier (not
  the parser) decides what to skip.
