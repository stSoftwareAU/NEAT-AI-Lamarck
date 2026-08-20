//! README Phase 5 ↔ `docs/scorer-call-cost.md` cost contract (Issue #172).
//!
//! Phase 5 used to quote per-creature scoring costs written before the
//! fixed/marginal decomposition of issue #112 existed ("≈0.7–1s/creature on GRQ
//! against ≈11s full"), roughly double the measured 452 ms screen and 5 490 ms
//! promote marginal costs the README itself cites as authoritative two sections
//! later. These tests re-derive the measured figures from the document's own
//! Result table and fail on any Phase 5 timing that contradicts them, whichever
//! side of the contradiction drifts next.
//!
//! The same contract over `docs/promote-gate.md` (Issue #183) lives in
//! `lamarck/tests/promote_gate_doc.rs`; the parsing both use is in
//! `lamarck/tests/common/mod.rs`.

mod common;

use common::{measured_costs, measured_scorer_costs_ms, read, section, timings_contradicting};

fn readme_phase_five() -> String {
    section(&read("README.md"), "\n### Phase 5").to_string()
}

/// The measurement the README defers to has to stay machine-readable, or the
/// consistency check below would pass by finding nothing to compare against.
#[test]
fn the_measured_result_table_still_parses_into_per_phase_fits() {
    let measured = measured_costs(&read("docs/scorer-call-cost.md"));
    let phases: Vec<&str> = measured.iter().map(|m| m.phase.as_str()).collect();
    for phase in ["screen", "promote"] {
        assert!(
            phases.contains(&phase),
            "docs/scorer-call-cost.md `## Result` no longer fits the {phase} phase: {phases:?}"
        );
    }
    for row in &measured {
        assert!(
            row.fixed_ms > 0.0 && row.marginal_ms > 0.0,
            "the {} row reports a non-positive fit: {row:?}",
            row.phase
        );
    }
}

/// Every timing Phase 5 states must be one `docs/scorer-call-cost.md` measured.
#[test]
fn phase_five_states_no_timing_the_measurement_contradicts() {
    let supported = measured_scorer_costs_ms();
    let contradicted = timings_contradicting(&readme_phase_five(), &supported);

    assert!(
        contradicted.is_empty(),
        "README Phase 5 states scorer timings docs/scorer-call-cost.md does not measure: \
         {contradicted:?} — measured ms: {supported:?}. Cite the document rather than \
         restating its numbers."
    );
}

/// Phase 5 is where a reader decides screen-versus-promote economics, so it has
/// to point at the measurement instead of leaving them to guess.
#[test]
fn phase_five_cites_the_measured_scorer_call_cost_document() {
    assert!(
        readme_phase_five().contains("docs/scorer-call-cost.md"),
        "README Phase 5 does not link docs/scorer-call-cost.md as the source of scoring costs"
    );
}
