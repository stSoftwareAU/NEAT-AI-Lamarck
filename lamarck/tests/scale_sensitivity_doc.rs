//! `docs/scale-sensitivity.md` ↔ code contract (issue #223).
//!
//! The document is an inventory: it tells a reader which constants move with
//! the creature, what each one resolves to, and what the derived arm was
//! measured to cost. All three decay silently. These tests pin the tooling it
//! points at, the literals and derived budgets it quotes, and — the honesty
//! gate — its "opt-in" status outliving the default it describes.

mod common;

use common::{read, repo_path};
use neat_ai_lamarck::scale::{CreatureScale, ResidualLimits, ScaleBudgetMode};
use neat_ai_lamarck::{LamarckConfig, ResolvedBudgets};
use neat_core::parse_creature_json;

fn doc() -> String {
    read("docs/scale-sensitivity.md")
}

/// The bench shapes the document tabulates: inputs, hiddens, fan-in.
const MEASURED_SHAPES: [(usize, usize, usize); 2] = [(2_511, 1_590, 12), (2_511, 4_852, 9)];

/// `n` with the document's thin-space digit grouping (`4102` → `4 102`).
fn grouped(n: usize) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, ch) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(' ');
        }
        out.push(ch);
    }
    out
}

/// Creature of the given shape, matching the bench fixture's topology.
fn creature(inputs: usize, hidden: usize, fan_in: usize) -> String {
    let mut neurons = String::new();
    let mut synapses = String::new();
    for h in 0..hidden {
        neurons.push_str(&format!(
            r#"{{"type":"hidden","uuid":"h{h}","bias":0.01,"squash":"TANH"}},"#
        ));
        for k in 0..fan_in {
            synapses.push_str(&format!(
                r#"{{"fromUUID":"input-{}","toUUID":"h{h}","weight":0.3}},"#,
                (h * 4 + k) % inputs
            ));
        }
        synapses.push_str(&format!(
            r#"{{"fromUUID":"h{h}","toUUID":"o1","weight":0.2}},"#
        ));
    }
    neurons.push_str(r#"{"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY"}"#);
    synapses.push_str(r#"{"fromUUID":"input-0","toUUID":"o1","weight":0.1}"#);
    format!(
        r#"{{"semanticVersion":"4.0.0","forwardOnly":true,"input":{inputs},"output":1,
           "neurons":[{neurons}],"synapses":[{synapses}]}}"#
    )
}

/// The document tells the reader to run these, so they have to exist.
#[test]
fn the_document_names_tooling_that_exists() {
    let doc = doc();
    for tool in [
        "lamarck/src/scale.rs",
        "lamarck/examples/scale_budgets_bench.rs",
        "lamarck/tests/scale_budgets.rs",
        "lamarck/tests/scale_sensitivity_doc.rs",
    ] {
        assert!(doc.contains(tool), "docs/scale-sensitivity.md drops {tool}");
        assert!(
            repo_path(tool).exists(),
            "docs/scale-sensitivity.md points at a missing {tool}"
        );
    }
}

/// The fixed literals the inventory quotes are the ones the code ships.
#[test]
fn the_inventory_quotes_the_fixed_literals_the_code_uses() {
    let doc = doc();
    let fixed = ResidualLimits::FIXED;
    for (label, value) in [
        ("residual shortlist head", fixed.shortlist),
        ("extra hidden sources", fixed.hidden_extra),
        ("synthetic probe rows", fixed.synthetic_probes),
    ] {
        assert!(
            doc.contains(&format!("`{value}` ")) || doc.contains(&format!("| `{value}` ")),
            "docs/scale-sensitivity.md does not quote the {label} literal {value}"
        );
    }
    // The derivation shape itself, so a reader can reproduce any row.
    assert!(
        doc.contains("clamp(round(gain × √population), floor, ceiling)"),
        "docs/scale-sensitivity.md drops the derivation formula"
    );
}

/// Every derived shortlist the measured table states is the one the code
/// resolves for that shape.
#[test]
fn the_measured_table_states_the_budgets_the_code_resolves() {
    let doc = doc();
    for (inputs, hidden, fan_in) in MEASURED_SHAPES {
        let parsed = parse_creature_json(&creature(inputs, hidden, fan_in)).expect("shape parses");
        let scale = CreatureScale::from_creature(&parsed);
        let derived = ResidualLimits::derived(scale);
        let row = format!("`{inputs}:{hidden}:{fan_in}`");
        assert!(
            doc.contains(&row),
            "docs/scale-sensitivity.md drops the measured shape {row}"
        );
        assert!(
            doc.contains(&format!(
                "{} (+{})",
                derived.shortlist, derived.hidden_extra
            )),
            "docs/scale-sensitivity.md states a derived shortlist for {row} that the code does not resolve \
             (expected {} (+{}))",
            derived.shortlist,
            derived.hidden_extra
        );
        assert!(
            doc.contains(&format!("| {} |", grouped(scale.neurons()))),
            "docs/scale-sensitivity.md states a neuron count for {row} that the shape does not have \
             (expected {})",
            grouped(scale.neurons())
        );
    }
}

/// Honesty gate: the document calls `derived` opt-in, so the default must be
/// `fixed` until a production A/B moves it — and then the document moves too.
#[test]
fn the_opt_in_status_matches_the_shipped_default() {
    let doc = doc();
    assert_eq!(
        LamarckConfig::default().scale_budgets,
        ScaleBudgetMode::Fixed,
        "the shipped default changed; docs/scale-sensitivity.md still calls derived opt-in"
    );
    assert!(
        doc.contains("stays **opt-in**"),
        "docs/scale-sensitivity.md drops the opt-in status of --scale-budgets"
    );
    assert!(
        doc.contains("run-economics question"),
        "docs/scale-sensitivity.md must keep stating what the analysis benchmark does not measure"
    );
}

/// The header contract the document promises: both arms record the budgets.
#[test]
fn both_arms_resolve_the_budgets_the_document_says_are_journalled() {
    let parsed = parse_creature_json(&creature(2_511, 1_590, 12)).expect("shape parses");
    let scale = CreatureScale::from_creature(&parsed);
    for mode in [ScaleBudgetMode::Fixed, ScaleBudgetMode::Derived] {
        let config = LamarckConfig {
            scale_budgets: mode,
            ..LamarckConfig::default()
        };
        let budgets = ResolvedBudgets::resolve(&config, scale);
        assert_eq!(budgets.mode, mode);
        assert!(
            budgets.graft_replay_ms > 0,
            "{} must resolve a Phase-G budget to journal",
            mode.label()
        );
        assert!(budgets.focus_count >= 1);
    }
    assert!(
        doc().contains("journalled on **both** arms"),
        "docs/scale-sensitivity.md drops the both-arms journalling contract"
    );
}
