//! `docs/screen-thresholds.md` ↔ code contract (Issue #220).
//!
//! The document describes a mechanism that decides which candidates are worth a
//! full-corpus score, so a reader has to be able to trust every knob, field and
//! default it quotes. These tests guard the ways that decays: the tooling it
//! points at going away, the report fields it tabulates being renamed, the
//! defaults it states drifting from the code, the guardrail wording being
//! deleted, and — the honesty gate — its "not yet run" status outliving the
//! absence of a production A/B.

mod common;

use common::{read, repo_path};
use neat_ai_lamarck::report::JournalReport;
use neat_ai_lamarck::screen_calibration::StrategyScreenCalibration;
use neat_ai_lamarck::screen_thresholds::{
    CALIBRATION_WINDOW, DEFAULT_SCREEN_CONTROL_RATE, MAX_THRESHOLD_ADJUSTMENT,
    MIN_CALIBRATION_PAIRS, SCREEN_THRESHOLD_MODEL_VERSION, StrategyThresholdRow,
};

fn doc() -> String {
    read("docs/screen-thresholds.md")
}

/// The document tells the reader to run these, so they have to exist.
#[test]
fn the_document_names_tooling_that_exists() {
    let doc = doc();
    for tool in [
        "scripts/run-screen-threshold-ab.sh",
        "scripts/summarise-screen-thresholds.sh",
        "lamarck/src/screen_thresholds.rs",
        "lamarck/tests/screen_thresholds.rs",
    ] {
        assert!(doc.contains(tool), "docs/screen-thresholds.md drops {tool}");
        assert!(
            repo_path(tool).exists(),
            "docs/screen-thresholds.md points at a missing {tool}"
        );
    }
}

/// Every report field the document tabulates is one `report` actually emits.
#[test]
fn every_report_field_the_document_quotes_is_serialised() {
    let doc = doc();
    let mut base = empty_report();
    base.screen_calibration.by_strategy = vec![StrategyScreenCalibration::default()];
    base.screen_threshold_replay.strategies = vec![StrategyThresholdRow::default()];
    let report = serde_json::to_value(base).expect("report serialises");

    let row = report
        .get("screenCalibration")
        .and_then(|calibration| calibration.get("byStrategy"))
        .and_then(|rows| rows.get(0))
        .expect("the report carries per-strategy calibration rows");
    for field in [
        "screened",
        "paired",
        "promotionPrecision",
        "meanScreenDelta",
        "fullDelta",
        "controlPromotions",
        "controlFalseNegatives",
        "promoteMs",
        "recommendedMultiplier",
        "recommendedThreshold",
        "basis",
    ] {
        assert!(
            row.get(field).is_some(),
            "`report` no longer emits `screenCalibration.byStrategy[].{field}`"
        );
        assert!(
            doc.contains(field),
            "docs/screen-thresholds.md drops the `{field}` column"
        );
    }

    let replay = report
        .get("screenThresholdReplay")
        .expect("the report carries the screenThresholdReplay bucket");
    for field in [
        "promotedAsRun",
        "promotedUnderCalibration",
        "promotionsAvoided",
        "promotionsAdded",
        "controlPromotions",
        "acceptsKept",
        "acceptsDropped",
        "improvementDropped",
        "promoteMsPerCreature",
        "promoteSecondsSaved",
        "scoreImprovementPerWallHourAsRun",
        "projectedScoreImprovementPerWallHour",
    ] {
        assert!(
            replay.get(field).is_some(),
            "`report` no longer emits `screenThresholdReplay.{field}`"
        );
        assert!(
            doc.contains(field),
            "docs/screen-thresholds.md drops the `{field}` field"
        );
    }
}

/// Every constant the document states is the constant the code ships.
#[test]
fn the_documented_defaults_are_the_shipped_defaults() {
    let doc = doc();
    assert!(
        doc.contains(&format!("`{DEFAULT_SCREEN_CONTROL_RATE}`")),
        "docs/screen-thresholds.md no longer states the default control rate"
    );
    assert!(
        doc.contains(&format!("most recent {CALIBRATION_WINDOW}")),
        "docs/screen-thresholds.md no longer states the calibration window"
    );
    assert!(
        doc.contains(&format!(
            "Fewer than {MIN_CALIBRATION_PAIRS} paired observations"
        )),
        "docs/screen-thresholds.md no longer states the evidence minimum"
    );
    assert!(
        doc.contains(&format!(
            "`[1/{}, {}]`",
            MAX_THRESHOLD_ADJUSTMENT as u32, MAX_THRESHOLD_ADJUSTMENT as u32
        )),
        "docs/screen-thresholds.md no longer states the clamp"
    );
    assert!(
        doc.contains(SCREEN_THRESHOLD_MODEL_VERSION),
        "docs/screen-thresholds.md no longer names the journalled model version"
    );
}

/// The guardrail is the reason this feature is allowed to exist: the screen
/// stays advisory, and the control sample is what keeps it falsifiable.
#[test]
fn the_document_states_the_guardrails() {
    let doc = doc();
    for claim in [
        "**Calibration never makes the screen authoritative.**",
        "cannot establish a false-negative rate",
        "is refused at startup",
    ] {
        assert!(
            doc.contains(claim),
            "docs/screen-thresholds.md dropped the guardrail: {claim}"
        );
    }
}

/// The whole safety argument is that nothing about an untouched run changed:
/// the document must keep saying so, and the CLI must keep it true.
#[test]
fn the_document_states_that_the_default_threshold_is_unchanged() {
    let doc = doc();
    assert!(
        doc.contains("`--screen-threshold-mode shared` stays the default"),
        "docs/screen-thresholds.md no longer states that the default is unchanged"
    );
    assert!(
        !neat_ai_lamarck::LamarckConfig::default()
            .screen_threshold_mode
            .is_per_strategy(),
        "the shipped default is no longer the shared threshold the document promises"
    );
}

/// No production A/B has been run, and the document must say so rather than
/// implying a measured verdict — or a measured false-negative rate — it does
/// not have.
#[test]
fn the_document_states_that_the_production_ab_is_unrun() {
    let doc = doc();
    assert!(
        doc.contains("### Status: not yet run"),
        "docs/screen-thresholds.md dropped its status section"
    );
    assert!(
        doc.contains("**No production A/B has been run for this feature.**"),
        "docs/screen-thresholds.md no longer states that the A/B is unrun"
    );
    assert!(
        doc.contains("improvement per wall hour, or about a false-negative rate"),
        "docs/screen-thresholds.md no longer states what it has not measured"
    );
    assert!(
        doc.contains("scoreImprovementPerWallHour"),
        "docs/screen-thresholds.md no longer names the gate metric the A/B compares"
    );
}

/// A report with no journal behind it — every field at its serialised default.
fn empty_report() -> JournalReport {
    let journal = tempfile::NamedTempFile::new().expect("temp journal");
    neat_ai_lamarck::report_from_journal(journal.path()).expect("an empty journal reports")
}
