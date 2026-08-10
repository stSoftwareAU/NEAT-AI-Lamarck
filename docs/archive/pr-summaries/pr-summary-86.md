# Convert the README core-principle diagram to Mermaid (Issue #86)

## Summary

The README's `## Core principle` section drew the mutate → screen → promote loop
as a plain ASCII `text` block, while the rest of the README already renders
Mermaid. It is now a colour-coded Mermaid flowchart carrying exactly the same
steps and branches: fittest creature → analyse one neuron → four candidate
sources → candidate population → subsample screen (drop or promote) →
full-corpus promote → accept or keep incumbent → repeat.

Colour groups the diagram by role — creature/focus (blue), candidate sources
(cyan), pool and loop (violet), scoring stages (amber), accept (green), drop and
keep (red). Every `classDef` sets `fill`, `stroke` **and** `color`, so the nodes
read the same in GitHub's light and dark themes rather than inheriting theme
text colours.

Other `text` blocks in the README (directory tree, `C0 -> C1` path, output
listing) are untouched — they are not flow diagrams.

Closes #86.

## Evidence

Rendered locally with `@mermaid-js/mermaid-cli` v11 (the same Mermaid engine
GitHub uses), which also proves the block parses:

Light theme:

![Core principle flowchart, light theme](docs/evidence/core-principle-mermaid-light.png)

Dark theme (`-t dark -b #0d1117`):

![Core principle flowchart, dark theme](docs/evidence/core-principle-mermaid-dark.png)

`./quality.sh` passes (fmt, clippy, cargo-deny, codespell, full test suite), and
`markdownlint-cli2` reports 0 errors across the 16 markdown files.

## Test Plan

Added to `lamarck/tests/readme_contract.rs` (all four failed before the README
change, with `Core principle has no ```mermaid fenced block`):

- `core_principle_diagram_is_a_mermaid_flowchart` — the section contains a
  `mermaid` fenced block starting with `flowchart`, and no `text` block remains.
- `core_principle_diagram_keeps_every_candidate_source` — statistical, backprop,
  structural and random sources all survive the conversion.
- `core_principle_diagram_shows_both_outcomes_and_repeats` — new incumbent, keep
  incumbent and the repeat edge are all present.
- `core_principle_diagram_colours_and_applies_every_class` — at least three
  `classDef`s, each declaring `fill:#`, `stroke:#` and `color:#`, and each
  actually applied to a node.

Supporting helper tests: `fenced_block_returns_only_the_requested_language`,
`fenced_block_is_none_for_a_missing_or_unterminated_block`.

The pre-existing contract tests that guard the diagram's meaning
(`core_principle_diagram_shows_two_phase_screening`,
`core_principle_diagram_shows_screened_out_candidates_are_dropped`) still pass
unmodified — that is the regression check that the conversion lost nothing.
