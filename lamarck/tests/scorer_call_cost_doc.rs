//! `docs/scorer-call-cost.md` ↔ code contract (Issue #112).
//!
//! The document's whole value is that its numbers can be re-measured and that
//! the conditions they were measured under are on the page. These tests guard
//! the ways that decays: the tooling it points at going away, the report fields
//! it quotes being renamed, the load conditions being dropped, and the go/no-go
//! it exists to deliver being softened into a suggestion.

use neat_ai_lamarck::report::JournalReport;
use std::path::{Path, PathBuf};

fn repo_path(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join(relative)
}

fn doc() -> String {
    let path = repo_path("docs/scorer-call-cost.md");
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()))
}

/// The document tells the reader to run these, so they have to exist.
#[test]
fn the_document_names_tooling_that_exists() {
    let doc = doc();
    for tool in [
        "scripts/measure-scorer-call-cost.sh",
        "lamarck/examples/scorer_call_cost_bench.rs",
        "lamarck/src/scorer_cost.rs",
        "docs/evidence/scorer-call-cost/rate-1.log",
        "docs/evidence/scorer-call-cost/rate-0_05.log",
    ] {
        assert!(doc.contains(tool), "docs/scorer-call-cost.md drops {tool}");
        assert!(
            repo_path(tool).exists(),
            "docs/scorer-call-cost.md points at a missing {tool}"
        );
    }
}

/// Every JSON field the document quotes is one `report` actually emits.
#[test]
fn every_report_field_the_document_quotes_is_serialised() {
    let doc = doc();
    let report = serde_json::to_value(empty_report()).expect("report serialises");
    let cost = report
        .get("scorerCallCost")
        .expect("report carries the scorerCallCost section");
    assert!(
        doc.contains("scorerCallCost"),
        "docs/scorer-call-cost.md no longer says which report section it is built from"
    );
    for field in ["calls", "failedCalls", "creaturesScored", "byPhase"] {
        assert!(
            cost.get(field).is_some(),
            "`report` no longer emits `scorerCallCost.{field}`, which the document is built on"
        );
        assert!(
            doc.contains(field),
            "docs/scorer-call-cost.md drops the {field} field"
        );
    }
    // The per-phase fit is the measurement itself.
    for field in ["fixedMs", "marginalMsPerCreature", "fixedMsShareAtMean"] {
        assert!(
            doc.contains(field),
            "docs/scorer-call-cost.md drops the {field} field"
        );
    }
}

/// A fixed cost measured beside a live scorer run is meaningless, so the
/// conditions are part of the result — the review-time gate from issue #112.
#[test]
fn the_document_records_the_load_conditions() {
    let doc = doc();
    for marker in ["loadBefore", "loadAfter"] {
        assert!(
            doc.contains(marker),
            "docs/scorer-call-cost.md dropped {marker} — the measurement conditions are the result"
        );
    }
}

/// The deliverable is a decision, not a recommendation for somebody else.
#[test]
fn the_document_carries_an_explicit_go_no_go_with_an_estimated_saving() {
    let doc = doc();
    assert!(
        doc.contains("## Decision"),
        "docs/scorer-call-cost.md has no decision section"
    );
    let decided = doc.contains("**Go**") || doc.contains("**No-go**");
    assert!(
        decided,
        "docs/scorer-call-cost.md states no explicit go/no-go on a persistent scoring session"
    );
    assert!(
        doc.contains("NEAT-AI-scorer#536"),
        "the decision must be cross-referenced to the scorer-side survey"
    );
}

/// The over-claim gate: the document must keep saying what it cannot support.
#[test]
fn the_document_states_what_the_measurement_cannot_support() {
    let doc = doc();
    assert!(
        doc.contains("What this measurement cannot support"),
        "the limits section is the review gate against a decision drawn from a loaded box"
    );
    for claim in [
        // The box carried a competing production scorer throughout.
        "not an idle box",
        // Lamarck cannot see inside the scorer process.
        "cannot attribute the fixed cost",
    ] {
        assert!(
            doc.contains(claim),
            "docs/scorer-call-cost.md dropped the limit: {claim}"
        );
    }
}

/// A `report` over no experiments at all — the shape check above needs one.
fn empty_report() -> JournalReport {
    let journal = tempfile::NamedTempFile::new().expect("temp journal");
    neat_ai_lamarck::report_from_journal(journal.path()).expect("an empty journal reports")
}
