## Summary

`docs/baseline-economics.md:32` told the reader to "see `.run-baseline-economics.sh`" — an uncommitted local helper from the #8 baseline run that exists nowhere in the repository. The parenthetical is dropped and the operational advice is now stated inline in the command fence: keep a private train-data copy (or hold `.in-use.lock`) because GRQ `node.sh` can delete `.trainData-binary_*` mid-run. That is the same substance already recorded at `docs/baseline-economics.md:126`, so nothing is lost by removing the pointer.

The script was not recoverable — `git ls-files` and a full-tree grep find only the reference itself — so committing it under `scripts/` was not an option. Closes #134.

## Evidence

Backend/docs change with no web interface to screenshot. Evidence is the new doc contract test, written first and failing against the unfixed document:

```text
running 4 tests
test the_document_names_reporting_tooling_that_exists ... ok
test the_document_keeps_the_operational_recommendation ... ok
test the_command_block_states_the_private_copy_advice_inline ... FAILED
test every_repo_script_the_document_names_exists ... FAILED

---- every_repo_script_the_document_names_exists stdout ----
docs/baseline-economics.md points at .run-baseline-economics.sh, which is not in the repository
```

After the doc fix:

```text
running 4 tests
test the_command_block_states_the_private_copy_advice_inline ... ok
test the_document_keeps_the_operational_recommendation ... ok
test the_document_names_reporting_tooling_that_exists ... ok
test every_repo_script_the_document_names_exists ... ok

test result: ok. 4 passed; 0 failed
```

`cargo test --workspace --all-features`, `cargo clippy -D warnings`, `cargo fmt --check`, `cargo deny check` and `cargo doc` all pass. `quality.sh` stops at its codespell preflight because `codespell` is not installed in this container (`spell-check: codespell is not installed.`) and no installer (`pip`/`pipx`) is available; every other gate in `quality.sh` was run individually and passed. CI runs the spell check for real.

## Test Plan

Added `lamarck/tests/baseline_economics_doc.rs`, matching the existing `*_doc.rs` contract-test pattern:

- `every_repo_script_the_document_names_exists` — the general guard: scans the document for every `.sh` token spelt as a repo path (contains `/`) or a repo-root dotfile (leading `.`) and asserts each resolves on disk. This is the regression test for #134; it fails on the unfixed document. A bare tool name such as GRQ's `node.sh` names someone else's script and is not treated as a repo claim.
- `the_document_names_reporting_tooling_that_exists` — `scripts/report-experiments.sh` is still named and still present.
- `the_command_block_states_the_private_copy_advice_inline` — the reproduction fence carries the train-data warning itself rather than deferring to a helper.
- `the_document_keeps_the_operational_recommendation` — the closing recommendation still names the private copy, `.in-use.lock` and `node.sh`.
