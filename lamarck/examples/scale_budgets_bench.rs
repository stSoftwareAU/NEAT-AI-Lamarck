//! Paired benchmark for creature-scale budgets (issue #223).
//!
//! Runs the post-focus analysis scan over the same creature and the same
//! corpus twice — once under `--scale-budgets fixed` (the pre-#223 literals)
//! and once under `--scale-budgets derived` — and reports the resolved budgets
//! and what the derived ones cost. Repeated across materially different
//! creature sizes, it answers the question the fixed literals cannot: what a
//! shortlist calibrated against a historical creature is worth on a creature
//! that has since grown.
//!
//! It measures the **analysis**, not the run economics: what a wider shortlist
//! is worth in accepts per wall-clock hour needs the real creature, the real
//! scorer and exclusive box time. No paired A/B script exists for
//! `--scale-budgets` yet, and the README's outstanding-work table records that.
//!
//! It also holds the focus count at one, so the derived arm's *other* budget is
//! not priced here — see the warning in `docs/scale-sensitivity.md`.
//!
//! Usage (release build — debug timings are meaningless):
//!
//! ```text
//! cargo run --release --example scale_budgets_bench -- [RECORDS] [REPEATS] [INPUTS:HIDDEN:FANIN,...]
//! ```
//!
//! Each shape is `INPUTS:HIDDEN:FANIN`; the default set spans a small creature,
//! a historical production shape and a materially larger current one.

use std::path::Path;
use std::time::Instant;

use neat_ai_lamarck::analysis::{ScanBudget, scan_post_focus};
use neat_ai_lamarck::scale::{CreatureScale, ResidualLimits, ScaleBudgetMode};
use neat_ai_lamarck::structural::RankedSource;
use neat_core::{CreatureExport, compile_creature, parse_creature_json};

mod support;
use support::{creature_json_with_fan_in, write_sample};

const FOCUS: &str = "o1";

/// Rank every input and hidden as an unused source, so the residual pass has
/// the whole creature to shortlist from.
fn prior_sources(creature: &CreatureExport) -> Vec<RankedSource> {
    let inputs = (0..creature.input).map(|i| format!("input-{i}"));
    let hiddens = creature
        .neurons
        .iter()
        .filter(|n| n.uuid != FOCUS)
        .map(|n| n.uuid.clone());
    inputs
        .chain(hiddens)
        .map(|from_uuid| RankedSource {
            from_uuid,
            score: 0.0,
            direction: 0.0,
            weight_scale: 1.0,
            ols_weight: None,
        })
        .collect()
}

/// Time one post-focus scan under `limits`, returning its wall-clock ms.
fn scan_ms(
    creature: &CreatureExport,
    data: &Path,
    records: u64,
    prior: &[RankedSource],
    limits: ResidualLimits,
) -> u128 {
    let mut network = compile_creature(creature).unwrap();
    let start = Instant::now();
    let scan = scan_post_focus(
        creature,
        &mut network,
        data,
        FOCUS,
        ScanBudget::serial(Some(records)).with_residual(limits),
        None,
        prior,
    )
    .unwrap();
    let elapsed = start.elapsed().as_millis();
    std::hint::black_box(scan);
    elapsed
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let arg = |i: usize, default: &str| -> String {
        args.get(i).cloned().unwrap_or_else(|| default.to_string())
    };
    let records: usize = arg(0, "2000").parse().expect("RECORDS must be a number");
    let repeats: usize = arg(1, "3").parse().expect("REPEATS must be a number");
    let shapes = arg(2, "64:16:4,2511:1590:12,2511:4852:9");

    println!("records={records} repeats={repeats}");
    println!(
        "{:<22} {:>9} {:>9} {:>12} {:>12} {:>9}",
        "shape (in:hidden:fan)", "neurons", "synapses", "fixed ms", "derived ms", "ratio"
    );

    for shape in shapes.split(',') {
        let dims: Vec<usize> = shape
            .split(':')
            .map(|d| d.parse().expect("shape must be INPUTS:HIDDEN:FANIN"))
            .collect();
        let [inputs, hidden, fan_in] = dims[..] else {
            panic!("shape must be INPUTS:HIDDEN:FANIN, got {shape}");
        };

        let dir = tempfile::tempdir().unwrap();
        write_sample(dir.path(), records, inputs);
        let creature =
            parse_creature_json(&creature_json_with_fan_in(inputs, hidden, fan_in)).unwrap();
        let scale = CreatureScale::from_creature(&creature);
        let prior = prior_sources(&creature);
        let fixed = ResidualLimits::resolve(ScaleBudgetMode::Fixed, scale);
        let derived = ResidualLimits::resolve(ScaleBudgetMode::Derived, scale);

        // Alternate the arms so page-cache state and machine drift hit both.
        let mut fixed_ms = u128::MAX;
        let mut derived_ms = u128::MAX;
        for _ in 0..repeats {
            fixed_ms = fixed_ms.min(scan_ms(
                &creature,
                dir.path(),
                records as u64,
                &prior,
                fixed,
            ));
            derived_ms = derived_ms.min(scan_ms(
                &creature,
                dir.path(),
                records as u64,
                &prior,
                derived,
            ));
        }

        println!(
            "{:<22} {:>9} {:>9} {:>12} {:>12} {:>9}",
            shape,
            scale.neurons(),
            scale.synapses,
            fixed_ms,
            derived_ms,
            format!("{:.2}x", derived_ms as f64 / fixed_ms.max(1) as f64)
        );
        println!(
            "  shortlist fixed={} (+{} hidden), derived={} (+{} hidden); ranked sources={}",
            fixed.shortlist,
            fixed.hidden_extra,
            derived.shortlist,
            derived.hidden_extra,
            prior.len()
        );
    }
}
