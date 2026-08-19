# Contributing to NEAT-AI-Lamarck

Thanks for improving **NEAT-AI-Lamarck** — an experimental optimiser for
already-fit NEAT-AI creatures. This guide covers the habits specific to raising
a change here; the build mechanics are documented once, in the README, and are
not repeated (Issue #136).

## Building and the local gate

The sibling-clone repository layout, the prerequisites, the cargo profiles, the
gate you must run before opening a PR, and the two PR workflows that maintain
formatting and the crate version are all described in
[README › Build and quality gate](./README.md#build-and-quality-gate). Read
that section before your first build, and edit **it** — not this file — when
any of it changes.

The [breaking-bump gate](./README.md#neat-core-breaking-bump-gate) is the part
of that section you meet as a contributor rather than as a reader: clearing it
is a deliberate acknowledgement, made in the same PR that updates Lamarck for
the change.

## Version bumping

**Every binary-affecting change must bump the patch version in
[`lamarck/Cargo.toml`](./lamarck/Cargo.toml)** (and keep `Cargo.lock` in sync).
Remote GRQ runners use the same pattern as
[`runlib.sh`](https://github.com/stSoftwareAU/GRQ-taxation/blob/Develop/scripts/runlib.sh):
they compare the installed `neat_ai_lamarck` version marker against
`Cargo.toml` and skip rebuilding when they match. Forgetting to bump leaves
stale binaries on remote machines.

Bumping locally whenever your change touches `lamarck/src/` keeps the version
correct and saves an extra bot commit from the PR workflow that would otherwise
do it for you.

**Never ship a crate version behind `origin/Develop`.** A merge conflict that
silently takes Develop's older `lamarck/Cargo.toml` version must fail CI, not
look like an intentional bump. Equal versions may still auto-patch-bump; ahead
versions are accepted without another bump
(`scripts/check-lamarck-version-no-downgrade.sh`, Issue #152).

Docs-only or CI-config-only changes do not need a bump.

## Changelog

Record notable changes under **[Unreleased]** in
[`CHANGELOG.md`](./CHANGELOG.md).
