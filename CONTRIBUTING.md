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
