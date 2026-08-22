//! Output gate for every creature Lamarck builds (issues #189 / #192).
//!
//! `neat_core::creature_validate` is the single definition of a valid
//! creature, shared with NEAT-AI over WASM and called natively here. Lamarck
//! applies grafts and structural edits, so it is one of the consumers that must
//! certify its own output: validate **after** the structure is final and
//! **before** the value escapes, so a violation is attributed to the graft or
//! edit that caused it rather than to a later consumer.
//!
//! The gate is deliberately *not* wired into deserialise or host load. An
//! externally supplied creature is not this repo's bug to report, and
//! load-time validation would add cost on every read.

use neat_core::{CreatureExport, ValidateOptions, ValidationFailure, creature_validate};

/// The options every Lamarck output is validated with.
///
/// Chosen deliberately rather than accepting [`ValidateOptions::default`]:
///
/// * `forward_only: true` — Lamarck only ever emits feed-forward creatures.
///   [`crate::structural::is_forward_edge`] already refuses a backward edge at
///   the point of insertion, so pinning the option closes the same rule at the
///   output boundary and additionally rejects self connections. It also forces
///   `feedback_loop` to `Some(false)`, which is why that field stays `None`
///   here — setting it would be redundant, and any other value would contradict
///   what Lamarck actually produces.
/// * `neurons: None` and `connections: None` — Lamarck *grows* structure, so it
///   has no fixed expected counts to assert. A pinned count would fail every
///   neuron-bridge graft by construction.
pub const LAMARCK_VALIDATE_OPTIONS: ValidateOptions = ValidateOptions {
    neurons: None,
    connections: None,
    feedback_loop: None,
    forward_only: true,
};

/// Certify `creature` against the shared rules.
///
/// `context` names what produced the creature — a graft id, or the structural
/// edit — and is prefixed to the message so the failure points at its cause.
///
/// # Errors
///
/// Returns the first violated rule rendered as `context: Class(REASON):
/// message [neuron N] [synapse N]`, folded into the `Result<_, String>` channel
/// the graft and structural entry points already use. The failure is never
/// swallowed or degraded to a log line: a creature that breaks a rule means the
/// edit that produced it is wrong.
pub fn validate_creature(creature: &CreatureExport, context: &str) -> Result<(), String> {
    creature_validate(creature, &LAMARCK_VALIDATE_OPTIONS)
        .map(|_stats| ())
        .map_err(|failure| describe_failure(context, &failure))
}

/// Render a [`ValidationFailure`] with every field it carries.
fn describe_failure(context: &str, failure: &ValidationFailure) -> String {
    let mut text = format!("{context}: {failure}");
    if let Some(index) = failure.neuron_index {
        text.push_str(&format!(" [neuron {index}]"));
    }
    if let Some(index) = failure.synapse_index {
        text.push_str(&format!(" [synapse {index}]"));
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use neat_core::parse_creature_json;

    fn valid_creature() -> CreatureExport {
        parse_creature_json(
            r#"{
              "semanticVersion":"4.0.0","forwardOnly":true,"input":2,"output":1,
              "neurons":[
                {"type":"hidden","uuid":"h1","bias":0.0,"squash":"IDENTITY"},
                {"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY"}
              ],
              "synapses":[
                {"fromUUID":"input-0","toUUID":"h1","weight":1.0},
                {"fromUUID":"h1","toUUID":"o1","weight":1.0}
              ]
            }"#,
        )
        .unwrap()
    }

    #[test]
    fn valid_creature_is_certified() {
        validate_creature(&valid_creature(), "unit").expect("a sound creature passes");
    }

    #[test]
    fn unsorted_synapses_are_refused_with_the_synapse_index() {
        let mut creature = valid_creature();
        creature.synapses.swap(0, 1);
        let err = validate_creature(&creature, "unit").expect_err("rule 25 rejects this");
        assert!(err.starts_with("unit: "), "context prefix: {err}");
        assert!(err.contains("SORT_FAILURE"), "reason: {err}");
        assert!(err.contains("[synapse 1]"), "synapse index: {err}");
    }

    #[test]
    fn self_connection_is_refused_under_forward_only() {
        let mut creature = valid_creature();
        creature.synapses[1].to_uuid = "h1".into();
        let err = validate_creature(&creature, "unit").expect_err("h1 -> h1 is a self connection");
        assert!(err.contains("SELF_CONNECTION"), "reason: {err}");
    }
}
