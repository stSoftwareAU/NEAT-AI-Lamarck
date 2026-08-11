//! Offline replay of the noise-aware promote gate (Issue #111).
//!
//! The gate can cost accepts, and a lost accept is invisible in production:
//! nothing in the journal records the acceptance that never happened. So the
//! gate is replayed here against journals that already exist, and the two
//! accepts the optimiser has ever earned (issue #8, both barely over the
//! `1e-6` bar) are asserted to survive it. **A gate that drops either of them
//! fails this test rather than a production run.**
//!
//! The fixtures are verbatim journal lines from the campaigns
//! `docs/screen-calibration.md` analyses, trimmed to the experiments that
//! matter and with the run header's local paths neutralised:
//!
//! - `baseline-45-first-six.jsonl` — experiments 1–6 of the #8 baseline run,
//!   which contains both accepts (experiments 3 and 5).
//! - `followup-75-batch-40-first-three.jsonl` — the head of a #75 arm, whose
//!   batches are dominated by catastrophic structural proposals. It accepted
//!   nothing across its whole run, so it is the pure-cost case.

use neat_ai_lamarck::promote_gate::{
    DEFAULT_SCREEN_PROMOTE_SIGMA_K, PromoteGateReplay, PromoteGateReplayAccumulator,
};
use neat_ai_lamarck::report::report_from_journal;
use neat_ai_lamarck::run::JournalLine;
use std::path::{Path, PathBuf};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/promote-gate")
        .join(name)
}

/// Replay one journal at a chosen σ̂ multiplier.
fn replay(name: &str, sigma_k: f64) -> PromoteGateReplay {
    let text = std::fs::read_to_string(fixture(name))
        .unwrap_or_else(|e| panic!("failed to read fixture {name}: {e}"));
    let mut accumulator = PromoteGateReplayAccumulator::with_sigma_k(sigma_k);
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        match JournalLine::parse(line).expect("fixture line parses") {
            JournalLine::Header(header) => accumulator.push_header(&header.config),
            JournalLine::Experiment(record) => accumulator
                .push_experiment(&record)
                .expect("fixture experiment replays"),
            JournalLine::GraftReplay(_) => {}
        }
    }
    accumulator.finish()
}

/// The hard gate: neither of the #8 accepts may be lost, at the default `k`
/// or at any multiplier a run might plausibly be given.
#[test]
fn the_noise_aware_gate_would_still_have_promoted_both_historical_accepts() {
    for sigma_k in [1.0, 2.0, DEFAULT_SCREEN_PROMOTE_SIGMA_K, 4.0, 5.0] {
        let replayed = replay("baseline-45-first-six.jsonl", sigma_k);
        assert_eq!(
            replayed.accepts.len(),
            2,
            "the fixture must carry both #8 accepts"
        );
        assert_eq!(
            replayed.accepts_dropped, 0,
            "k={sigma_k} would have discarded a real accept: {:#?}",
            replayed.accepts
        );
        for accept in &replayed.accepts {
            assert!(
                accept.would_promote,
                "k={sigma_k} drops experiment {} winner {} (screen Δ {:?}, gate demanded {:e})",
                accept.experiment_number, accept.stem, accept.screen_delta, accept.threshold
            );
        }
    }
}

/// The gate may never buy *more* full-corpus scores than the run it replays:
/// it is a strict subset of the absolute gate, on real batches too.
#[test]
fn the_noise_aware_gate_never_buys_more_than_the_run_did() {
    for name in [
        "baseline-45-first-six.jsonl",
        "followup-75-batch-40-first-three.jsonl",
    ] {
        for sigma_k in [1.0, DEFAULT_SCREEN_PROMOTE_SIGMA_K, 5.0] {
            let replayed = replay(name, sigma_k);
            assert!(
                replayed.promoted_under_gate <= replayed.promoted_as_run,
                "{name} at k={sigma_k}: gate promoted {} against the run's {}",
                replayed.promoted_under_gate,
                replayed.promoted_as_run
            );
            assert_eq!(
                replayed.promotions_avoided,
                replayed.promoted_as_run - replayed.promoted_under_gate
            );
        }
    }
}

/// The saving the gate is for. This #75 arm bought a full-corpus score and
/// accepted nothing across its whole run; the gate declines to buy it. The
/// screened count is untouched — the screen tier itself still runs.
#[test]
fn a_pure_cost_arm_loses_its_promotion_and_no_accepts() {
    let replayed = replay(
        "followup-75-batch-40-first-three.jsonl",
        DEFAULT_SCREEN_PROMOTE_SIGMA_K,
    );
    assert_eq!(replayed.accepts.len(), 0, "this arm accepted nothing");
    assert_eq!(replayed.accepts_dropped, 0);
    assert_eq!(
        replayed.screened, 99,
        "the screen tier still scores the batch"
    );
    assert_eq!(replayed.promoted_as_run, 1);
    assert_eq!(replayed.promoted_under_gate, 0);
    assert_eq!(replayed.promotions_avoided, 1);
}

/// The replayed floor comes from the journal's own header when it has one, and
/// the gate the run itself used is reported beside the replay so the two are
/// never confused.
#[test]
fn the_replay_reports_the_gate_the_run_actually_used() {
    let arm = replay(
        "followup-75-batch-40-first-three.jsonl",
        DEFAULT_SCREEN_PROMOTE_SIGMA_K,
    );
    assert_eq!(
        arm.gate_as_run.as_deref(),
        Some("absolute"),
        "a pre-#111 journal ran the absolute gate"
    );
    assert_eq!(arm.replay_floor, 1e-6, "floor comes from the run header");
    assert_eq!(arm.replay_sigma_k, DEFAULT_SCREEN_PROMOTE_SIGMA_K);

    // The #8 baseline predates the run header entirely (issue #71), so there is
    // no gate to report and the floor falls back to the documented default.
    let baseline = replay(
        "baseline-45-first-six.jsonl",
        DEFAULT_SCREEN_PROMOTE_SIGMA_K,
    );
    assert_eq!(baseline.gate_as_run, None);
    assert_eq!(baseline.replay_floor, 1e-6);
}

/// `report` carries the replay, so the evidence is reproducible from the CLI
/// on any journal rather than only from this test.
#[test]
fn report_from_journal_carries_the_replay() {
    let report =
        report_from_journal(&fixture("baseline-45-first-six.jsonl")).expect("fixture reports");
    let replayed = &report.promote_gate_replay;
    assert_eq!(replayed.accepts_dropped, 0);
    assert_eq!(replayed.replay_sigma_k, DEFAULT_SCREEN_PROMOTE_SIGMA_K);
    assert_eq!(
        replayed.promoted_as_run, report.screen_calibration.paired_candidates,
        "promotions replayed must be the promotions the calibration paired"
    );

    let json = serde_json::to_value(&report).expect("report serialises");
    let section = json
        .get("promoteGateReplay")
        .expect("report carries promoteGateReplay");
    for field in [
        "gateAsRun",
        "replaySigmaK",
        "screened",
        "promotedAsRun",
        "promotedUnderGate",
        "acceptsKept",
        "acceptsDropped",
    ] {
        assert!(
            section.get(field).is_some(),
            "promoteGateReplay omits `{field}`"
        );
    }
}
