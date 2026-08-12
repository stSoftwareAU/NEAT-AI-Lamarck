//! `docs/baseline-reuse.md` ↔ code contract (Issue #113).
//!
//! The document's value is that every figure in it is reproducible and that the
//! guards it promises are the ones the code has. These tests guard the ways
//! that decays: the tooling it points at going away, the flags and journal
//! fields it quotes being renamed, and — the review-time gate against a
//! confident recommendation drawn from one host — its limits section being
//! deleted.

use std::path::{Path, PathBuf};

fn repo_path(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join(relative)
}

fn doc() -> String {
    let path = repo_path("docs/baseline-reuse.md");
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()))
}

/// The document tells the reader to run these, so they have to exist.
#[test]
fn the_document_names_tooling_that_exists() {
    let doc = doc();
    for tool in [
        "lamarck/examples/promote_baseline_bench.rs",
        "lamarck/src/baseline.rs",
        "lamarck/src/run.rs",
        "docs/evidence/baseline-reuse/paired-sweeps.log",
    ] {
        assert!(doc.contains(tool), "docs/baseline-reuse.md drops {tool}");
        assert!(
            repo_path(tool).exists(),
            "docs/baseline-reuse.md points at a missing {tool}"
        );
    }
}

/// Every knob and journal field the document quotes must still be spelt that
/// way — a renamed flag makes the document quietly wrong.
#[test]
fn the_document_quotes_the_current_flag_and_journal_names() {
    let doc = doc();
    for name in [
        "--baseline-reverify-interval",
        "--baseline-drift-epsilon",
        "baselineSource",
        "rememberedVerified",
        "netCreatureScoresSaved",
    ] {
        assert!(doc.contains(name), "docs/baseline-reuse.md drops {name}");
    }
}

/// The three guards are the whole argument for the change being safe. A
/// document that lost one would be recommending an unguarded saving.
#[test]
fn the_document_states_all_three_guards() {
    let doc = doc().to_lowercase();
    for phrase in [
        "content fingerprint",
        "every accept is verified before the swap",
        "drift aborts the run",
    ] {
        assert!(
            doc.contains(phrase),
            "docs/baseline-reuse.md no longer states the {phrase:?} guard"
        );
    }
}

/// A measurement from one host on one afternoon must keep saying so.
#[test]
fn the_document_keeps_its_limits_section() {
    let doc = doc();
    assert!(
        doc.contains("## What this measurement cannot support"),
        "docs/baseline-reuse.md lost its limits section"
    );
    let limits = doc
        .split("## What this measurement cannot support")
        .nth(1)
        .expect("limits section located above");
    assert!(
        limits.lines().filter(|l| l.starts_with("- ")).count() >= 3,
        "the limits section no longer lists what the benchmark cannot support"
    );
}

/// The README is where an operator meets the flag, so it must carry the pointer
/// to the measurement rather than leaving the number unsourced.
#[test]
fn the_readme_links_the_measurement() {
    let readme = std::fs::read_to_string(repo_path("README.md")).expect("README.md is readable");
    assert!(
        readme.contains("docs/baseline-reuse.md"),
        "README.md does not link docs/baseline-reuse.md"
    );
}
