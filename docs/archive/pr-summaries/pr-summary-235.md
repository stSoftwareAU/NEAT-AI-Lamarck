# Pin neat-core to a release tag and refresh the pin on every PR

## Summary

`neat-core` is no longer a sibling `path` dependency that tracks core's head.
`lamarck/Cargo.toml` now carries a git dependency pinned to core's latest
release tag, and the `family-sync` job moves that pin to the newest release on
every PR — so the workspace builds in a bare clone and core releases arrive
through this repository's own PR, compiled and tested where they land.
Closes #235.

- **`lamarck/Cargo.toml`** — `neat-core = { git = "https://github.com/stSoftwareAU/NEAT-AI-core", tag = "v0.22.5" }`,
  on one line (the form `family-pins.sh` rewrites); `Cargo.lock` regenerated and
  now records the resolved commit `771ad136…`.
- **`scripts/family-pins.sh`** — a byte-identical copy of NEAT-AI-core
  `Develop`'s canonical script (NEAT-AI-core#681), never edited here.
- **`.github/workflows/family-sync.yml`** — syncs *both* canonical scripts,
  re-lints and re-runs their contract tests where refreshed bytes arrive, then
  runs `family-pins.sh`; a moved pin is compiled and tested in the same job, and
  the refreshed scripts, the moved pin and the updated `Cargo.lock` travel in
  one commit. `version-increment.yml` is paths-filtered on
  `lamarck/Cargo.toml` and `Cargo.lock`, so the pin never moves at an unchanged
  crate version.
- **Sibling checkout retired** — `.github/actions/setup-neat-core` is deleted
  and removed from `ci.yml`, `auto-format.yml`, `cargo-quality.yml`,
  `security.yml` and `sbom.yml`.
- **Baseline gate removed** — `scripts/check-neat-core-version.sh` and
  `neat-core.expected-version` guarded an unpinned path dependency that no
  longer exists. Its job (catching a breaking core release) is now done where
  the pin moves: the family-sync job compiles and tests it.
- **`deny.toml`** — allows the one `stSoftwareAU/NEAT-AI-core` git source;
  every other git source stays denied (`cargo deny check` fails otherwise).
- **`auto-format.yml`** — no longer runs `cargo update -p neat-core`; a second
  mover would race the commit carrying the matching `Cargo.lock` bump, and
  `check-auto-format-workflow.sh` now fails CI if it reappears.

## Evidence

Backend/CI change — no web interface to screenshot. The evidence is command
output and tests.

**The workspace builds with no `NEAT-AI-core` checkout beside it.** This
container has no sibling clone (`ls ../../` → `pr-branch-update  s1  s2`):

```text
$ cargo build --workspace
   Compiling neat-core v0.22.5 (https://github.com/stSoftwareAU/NEAT-AI-core?tag=v0.22.5#771ad136)
   Compiling neat_ai_lamarck v0.1.37 (…/NEAT-AI-Lamarck/lamarck)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 8.80s
```

**A pin behind core's latest release is moved, and `Cargo.lock` follows.** The
shipped pin was temporarily set back to `v0.22.0` and the real script run:

```text
$ ./scripts/family-pins.sh
[family-pins] neat-core v0.22.0 → v0.22.5 (lamarck/Cargo.toml)
[family-pins] 1 pin(s) moved; Cargo.lock updated
$ grep -n 'neat-core = ' lamarck/Cargo.toml
30:neat-core = { git = "https://github.com/stSoftwareAU/NEAT-AI-core", tag = "v0.22.5" }
$ ./scripts/family-pins.sh   # idempotent: already latest, exits 0 changing nothing
```

**The full gate passes**: `./quality.sh < /dev/null` → `All quality checks
passed!` (shellcheck, both canonical-script contract suites, every workflow
validator, codespell, `cargo deny check`, fmt, clippy with `-D warnings`, the
whole test suite, rustdoc). `actionlint` is clean and `markdownlint-cli2` finds
0 issues.

```mermaid
flowchart TD
    PR([PR opened / synchronised]) --> Fetch["Fetch scripts/runlib.sh and<br/>scripts/family-pins.sh from<br/>NEAT-AI-core Develop"]
    Fetch -->|fetch error or empty| Fail["Job fails<br/>(never reported as in sync)"]
    Fetch -->|fetched| Cmp{"cmp against the<br/>committed copies"}
    Cmp -->|identical| Pins
    Cmp -->|differs| Refresh["Rewrite the copy,<br/>shellcheck + contract tests"]
    Refresh --> Pins["Run scripts/family-pins.sh"]
    Pins -->|pin already latest| Clean{"anything changed?"}
    Pins -->|pin moved| Build["cargo test the moved pin"]
    Build --> Clean
    Clean -->|no| Pass([No commit — already in sync])
    Clean -->|yes| Rebase["Rebase onto origin/&lt;head&gt;"]
    Rebase --> Push["Push 'chore: sync from<br/>NEAT-AI-core Develop'"]
    Push --> Bump["version-increment.yml sees<br/>lamarck/Cargo.toml + Cargo.lock<br/>→ patch bump"]
    Bump --> Review([PR carries the canonical copies<br/>and core's latest release])
```

Reviewer note: the pushed sync commit triggers `version-increment.yml` only
when the push is made with the configured App token — a push made with the
default `GITHUB_TOKEN` starts no new workflow run. That is why a moved pin is
compiled and tested inside the family-sync job itself rather than left to the
CI run of the resulting commit.

## Test Plan

- **Added `scripts/test-family-pins.sh`** (15 assertions, hermetic — no
  network, no cargo): `--help` and usage errors exit 0/2; a manifest with no
  family pin, a commented-out pin and a non-family git pin are left
  byte-for-byte alone; a multi-line family pin fails loud naming the file and
  the line rather than reading as "already current". Wired into `quality.sh`
  and the CI **Shell Script Quality** job, and re-run by `family-sync.yml`
  whenever the copied script is refreshed.
- **Extended `scripts/check-family-sync-workflow.sh`** with four new rules
  (both canonical scripts fetched; `family-pins.sh` actually run;
  `lamarck/Cargo.toml` + `Cargo.lock` staged; a moved pin compiled) and
  **`scripts/test-check-family-sync-workflow.sh`** with a fixture per rule —
  each mutates the shipped workflow one way and asserts the validator exits 1.
  All four fixtures were observed failing the validator, and the shipped
  workflow passing it.
- **Inverted the `neat-core` rule in
  `scripts/check-auto-format-workflow.sh`** (auto-format must *not* move the
  pin) with a new fixture in `scripts/test-check-auto-format-workflow.sh` that
  reintroduces `cargo update -p neat-core` and asserts exit 1.
- **`lamarck/tests/contributing_contract.rs`** — the README-owned-facts list
  named the deleted `neat-core.expected-version`; it now names
  `family-sync.yml`.
- Existing suites unchanged and green, including `readme_contract.rs` and the
  repository-layout contract that tracks the top-level file list.
