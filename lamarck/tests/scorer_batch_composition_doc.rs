//! `docs/scorer-batch-composition.md` ↔ code contract (issue #130).
//!
//! The document's value is that the rule it states — a Δ is only valid inside
//! one scorer call — is the rule the code actually applies, and that the
//! evidence and the upstream hand-off survive editing. These tests guard both:
//! the accept gate behaves as documented, and the sections a future editor is
//! most likely to trim are still there.

use std::path::{Path, PathBuf};

use neat_ai_lamarck::{ComboSelection, ScoreResult, StackDampenReport, accepts_improvement_delta};

fn repo_path(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join(relative)
}

fn doc() -> String {
    let path = repo_path("docs/scorer-batch-composition.md");
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()))
}

fn selection(delta: f64, score: f64) -> ComboSelection {
    ComboSelection {
        creature_path: PathBuf::from("combo-000-k2.json"),
        stem: "combo-000-k2".into(),
        result: ScoreResult {
            score,
            error: 1.0 - score,
            complexity_penalty: 0.0,
        },
        delta,
        member_indices: vec![0, 1],
        dampen: StackDampenReport::default(),
        combos_scored: 1,
        combos_dampened: 0,
    }
}

/// The documented rule: the gate reads the winner's same-call Δ and nothing
/// else. A raw score dragged up or down by another call's batch composition
/// must not change the verdict.
#[test]
fn the_accept_gate_reads_the_same_call_delta_only() {
    let min_improvement = 1e-6;
    // Same Δ, wildly different raw scores → same verdict.
    assert!(selection(2e-6, 0.5).accepts(min_improvement));
    assert!(selection(2e-6, 0.1).accepts(min_improvement));
    // A raw score far above the promote-call baseline cannot rescue a Δ that
    // does not clear the bar.
    assert!(!selection(5e-7, 0.9).accepts(min_improvement));
    // The bar is strict.
    assert!(!selection(min_improvement, 0.5).accepts(min_improvement));
    assert!(accepts_improvement_delta(1.000_001e-6, min_improvement));
}

/// The document tells the reader where the rule lives, so those files have to
/// exist.
#[test]
fn the_document_names_code_that_exists() {
    let doc = doc();
    for path in [
        "lamarck/src/combos.rs",
        "lamarck/src/run.rs",
        "docs/followup-economics.md",
    ] {
        assert!(
            doc.contains(path),
            "docs/scorer-batch-composition.md drops {path}"
        );
        assert!(
            repo_path(path).exists(),
            "docs/scorer-batch-composition.md points at a missing {path}"
        );
    }
}

/// The measured evidence is the whole reason the rule exists — a document that
/// lost it would read as an unmotivated restriction.
#[test]
fn the_document_keeps_the_measured_evidence() {
    let doc = doc();
    for figure in [
        "0.347854391736837",
        "0.347854330680259",
        "1.755e-7",
        "6.7e-8",
        "--min-improvement 1e-6",
    ] {
        assert!(
            doc.contains(figure),
            "docs/scorer-batch-composition.md drops the measured {figure}"
        );
    }
}

/// The artefact is upstream. Losing that section would leave the document
/// claiming Lamarck fixed a scorer bug it only routed around.
#[test]
fn the_document_states_what_is_not_fixed_here() {
    let doc = doc();
    for phrase in [
        "What this does not fix",
        "NEAT-AI-scorer",
        "multi_score.rs",
        "workers_per_creature_split",
    ] {
        assert!(
            doc.contains(phrase),
            "docs/scorer-batch-composition.md drops {phrase}"
        );
    }
}

/// The README section this document backs must keep pointing at it.
#[test]
fn the_readme_points_at_the_document() {
    let readme = std::fs::read_to_string(repo_path("README.md")).expect("read README.md");
    assert!(
        readme.contains("docs/scorer-batch-composition.md"),
        "README.md drops the link to docs/scorer-batch-composition.md"
    );
    assert!(
        readme.contains("Same-call deltas"),
        "README.md drops the Same-call deltas section"
    );
}
