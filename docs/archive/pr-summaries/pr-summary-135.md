# Define NEAT, GRQ and squash on first use (Issue #135)

## Summary

Three terms of art were used in `README.md` with no first-use definition: **NEAT**
was never expanded (only the NEAT-AI repo was linked), **GRQ** framed the production
scale target, the corpus, the scorer host and several operational warnings without
being explained anywhere in this repository, and **squash** appeared in the candidate
table and as a journal field without being identified as the neuron's activation
function.

Each term is now glossed where it is first used:

- Intro — NEAT is expanded to
  [NeuroEvolution of Augmenting Topologies](https://en.wikipedia.org/wiki/Neuroevolution_of_augmenting_topologies)
  with a one-line statement of what the algorithm does and what NEAT-AI is.
- `## Status` — GRQ is glossed as the private production system whose evolved
  trading models this optimiser tunes, plus the note that nothing in this repository
  needs access to it.
- `#### Aggregate neurons` (the first use of `squash`) — a squash is a neuron's
  activation ("squashing") function, e.g. `TANH`, and the `MINIMUM` / `MAXIMUM` /
  `IF` aggregates are squashes too.

Three README contract tests keep the glosses in place: they locate the first use of
each term in the README as committed and assert the surrounding text defines it, so
deleting or moving a gloss fails the suite rather than reaching a reader.

Closes #135.

## Evidence

Documentation-only change — no web interface to screenshot. Verified by the test
suite and the markdown/lint gates.

New tests fail against the unfixed README (run before the README edits):

```text
---- readme_expands_neat_before_the_first_section stdout ----
README.md never expands the NEAT acronym in its opening section

---- readme_glosses_grq_where_it_is_first_used stdout ----
the paragraph first using GRQ omits "private" — a reader is left to guess what GRQ is

---- readme_glosses_squash_where_it_is_first_used stdout ----
the paragraph first using `squash` omits "squashing" — the term is never glossed

test result: FAILED. 37 passed; 3 failed
```

After the README edits:

```text
cargo test --test readme_contract
test result: ok. 40 passed; 0 failed

cargo test --workspace --all-features -- --test-threads=2
all suites ok (405 unit tests + integration suites, 0 failed)

markdownlint-cli2 → Summary: 0 issues in 0 files
cargo fmt --check / cargo clippy -D warnings / cargo doc → clean
cargo deny check → advisories ok, bans ok, licenses ok, sources ok
```

**Gate not run locally:** `./quality.sh` stops at the codespell preflight because
`codespell` is not installed in this container and there is no `pip`/`pipx` to
install it (`/bin/bash: line 1: pip: command not found`, `python3: No module named
pip`). Every other step of `quality.sh` was run individually and passed, as listed
above; CI runs the codespell job for real.

## Test Plan

Added to `lamarck/tests/readme_contract.rs`:

- `readme_expands_neat_before_the_first_section` — the preamble must expand NEAT
  and link the reference page.
- `readme_glosses_grq_where_it_is_first_used` — the paragraph holding the first
  `GRQ` mention must say it is a private production system.
- `readme_glosses_squash_where_it_is_first_used` — the paragraph holding the first
  `squash` mention must gloss it as the activation ("squashing") function.
- Helper unit tests for the two new helpers: `preamble_stops_at_the_first_section`,
  `preamble_returns_everything_when_there_are_no_sections`,
  `paragraph_containing_returns_only_the_matching_paragraph`,
  `paragraph_containing_panics_when_the_term_is_absent`.

No existing tests were modified or removed.
