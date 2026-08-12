//! `docs/screen-calibration.md` ↔ code contract (Issue #110).
//!
//! The document's whole value is that every figure in it is reproducible from
//! a journal and carries the sample it came from. These tests guard the two
//! ways that decays: the report field names the document quotes going away,
//! and the "what this cannot support" section — the review-time gate against a
//! confident recommendation drawn from two accepts — being deleted.

use neat_ai_lamarck::report::JournalReport;
use neat_ai_lamarck::screen_calibration::ScreenCalibrationAccumulator;
use std::path::{Path, PathBuf};

fn repo_path(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join(relative)
}

fn doc() -> String {
    let path = repo_path("docs/screen-calibration.md");
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()))
}

/// The document tells the reader to run these, so they have to exist.
#[test]
fn the_document_names_tooling_that_exists() {
    let doc = doc();
    for tool in [
        "scripts/summarise-screen-calibration.sh",
        "lamarck/src/screen_calibration.rs",
    ] {
        assert!(
            doc.contains(tool),
            "docs/screen-calibration.md drops {tool}"
        );
        assert!(
            repo_path(tool).exists(),
            "docs/screen-calibration.md points at a missing {tool}"
        );
    }
}

/// Every JSON field the document quotes is one `report` actually emits.
#[test]
fn every_report_field_the_document_quotes_is_serialised() {
    let doc = doc();
    let report = serde_json::to_value(empty_report()).expect("report serialises");
    let calibration = report
        .get("screenCalibration")
        .expect("report carries the screenCalibration section");
    for field in [
        "screenEnabled",
        "screenOnlyCandidates",
        "fullOnlyCandidates",
        "distinctPairs",
    ] {
        assert!(
            doc.contains(field),
            "docs/screen-calibration.md no longer explains `{field}`"
        );
        assert!(
            calibration.get(field).is_some(),
            "`report` no longer emits `{field}`, which the document quotes"
        );
    }
}

/// The over-claim gate: the document must keep saying what it cannot support.
#[test]
fn the_document_states_what_the_sample_cannot_support() {
    let doc = doc();
    assert!(
        doc.contains("What this sample cannot support"),
        "the limits section is the review gate against a two-accept recommendation"
    );
    for claim in [
        // The sample rate never varied, so it cannot be recommended on.
        "cannot price `--screen-sample-rate`",
        // A rejected candidate is never full-scored.
        "cannot establish a false-negative rate",
        // Six arms share one seed and replay one experiment stream.
        "not 222 independent samples",
    ] {
        assert!(
            doc.contains(claim),
            "docs/screen-calibration.md dropped the limit: {claim}"
        );
    }
}

/// A `report` over no experiments at all — the shape check above needs one.
fn empty_report() -> JournalReport {
    let calibration = ScreenCalibrationAccumulator::default().finish();
    assert!(
        !calibration.screen_enabled,
        "an empty journal has no screen phase"
    );
    let journal = tempfile::NamedTempFile::new().expect("temp journal");
    neat_ai_lamarck::report_from_journal(journal.path()).expect("an empty journal reports")
}
