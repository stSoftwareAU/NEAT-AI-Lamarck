# Contributing to NEAT-AI-Lamarck

Thanks for improving **NEAT-AI-Lamarck** — an experimental optimiser for
already-fit NEAT-AI creatures. This guide summarises how to build, test, and
submit changes.

## Repository layout

Clone **NEAT-AI-core** and **NEAT-AI-Lamarck** as siblings:

```text
parent/
  NEAT-AI-core/
  NEAT-AI-Lamarck/
```

The `neat-core` path dependency in [`lamarck/Cargo.toml`](./lamarck/Cargo.toml)
resolves to `../../NEAT-AI-core/neat-core`.

## Prerequisites

- **Rust** — pinned in [`rust-toolchain.toml`](./rust-toolchain.toml)
- **shellcheck** — lints bash scripts
- **cargo-deny** — `cargo install cargo-deny --locked`
- **codespell** — `pip install --user codespell`

## Local gate

```bash
./quality.sh < /dev/null
```

This mirrors CI: shellcheck, the auto-format workflow validator (Issue #33),
codespell, cargo-deny, fmt `--check`, clippy with warnings denied, tests, and
rustdoc.

On each PR the **Auto Format** workflow
([`.github/workflows/auto-format.yml`](./.github/workflows/auto-format.yml))
runs `cargo fmt --all` and `cargo update -p neat-core`, then pushes any
tracked-tree changes back to the PR branch. It does **not** bump
`neat-core.expected-version` — acknowledge breaking neat-core bumps
deliberately in the same PR that updates Lamarck for them.

## Version bumping

**Every binary-affecting change must bump the patch version in
[`lamarck/Cargo.toml`](./lamarck/Cargo.toml)** (and keep `Cargo.lock` in sync).
Remote GRQ runners use the same pattern as
[`runlib.sh`](https://github.com/stSoftwareAU/GRQ-taxation/blob/Develop/scripts/runlib.sh):
they compare the installed `neat_ai_lamarck` version marker against
`Cargo.toml` and skip rebuilding when they match. Forgetting to bump leaves
stale binaries on remote machines.

CI also runs a **Version Increment** workflow
([`.github/workflows/version-increment.yml`](./.github/workflows/version-increment.yml))
that auto-increments the patch on a pull request when `lamarck/src/` has
changed — but only if the PR branch has not already bumped it (same approach
as GRQ-taxation). Bumping locally when your change touches `lamarck/src/` keeps
the version correct and avoids an extra bot commit.

**Never ship a crate version behind `origin/Develop`.** A merge conflict that
silently takes Develop's older `lamarck/Cargo.toml` version must fail CI, not
look like an intentional bump. Equal versions may still auto-patch-bump; ahead
versions are accepted without another bump
(`scripts/check-lamarck-version-no-downgrade.sh`, Issue #152).

Docs-only or CI-config-only changes do not need a bump. Record notable changes
under **[Unreleased]** in [`CHANGELOG.md`](./CHANGELOG.md).
