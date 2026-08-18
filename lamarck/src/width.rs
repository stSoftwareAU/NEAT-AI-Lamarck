//! Observation-width guard for creature JSON (issue #165).
//!
//! The top-level `input` / `output` integers of a `CreatureExport` are the
//! observation width. They are authoritative and **cannot be re-derived** —
//! `neurons` lists only the non-input neurons — so a creature whose width is
//! missing or `< 1` is refused everywhere: on load, before any training-record
//! framing, and on every write. There is no fallback and no default.
//!
//! This is a local guard at Lamarck's own boundary. It stays in place after
//! NEAT-AI-core#550 tightens `parse_creature_json` / `creature_to_json`; the
//! wording mirrors NEAT-AI TS `CreatureValidate.ts` so the fleet fails the same
//! way.

use neat_core::{CreatureExport, creature_to_json, creature_to_json_pretty, parse_creature_json};
use serde_json::Value;

/// Observation width of a creature: `(input, output)`.
pub type Width = (usize, usize);

/// Refuse a creature whose `input` or `output` is `< 1`.
///
/// Wording matches NEAT-AI TS `CreatureValidate.ts` so a fleet operator sees
/// one message whichever binary caught it.
pub fn validate_creature_width(creature: &CreatureExport) -> Result<(), String> {
    validate_width((creature.input, creature.output))
}

/// [`validate_creature_width`] on a bare `(input, output)` pair.
pub fn validate_width((input, output): Width) -> Result<(), String> {
    if input < 1 {
        return Err(format!("Must have at least one input neurons was: {input}"));
    }
    if output < 1 {
        return Err(format!(
            "Must have at least one output neurons was: {output}"
        ));
    }
    Ok(())
}

/// Parse creature JSON and refuse it unless `input >= 1` and `output >= 1`.
///
/// Every load path in this crate goes through here so an `input: 0` creature
/// can never reach `TrainingDataConfig::new(creature.input, creature.output)`
/// and mis-frame the corpus (`chunks.rs`, `analysis.rs`, `parity.rs`, …).
pub fn load_creature(text: &str) -> Result<CreatureExport, String> {
    let creature = parse_creature_json(text).map_err(|e| e.to_string())?;
    validate_creature_width(&creature)?;
    Ok(creature)
}

/// Assert `written` carries the same observation width as `source`, and that
/// the width is `>= 1`. Call before any creature text reaches disk.
pub fn assert_written_width(source: &CreatureExport, written: Width) -> Result<(), String> {
    validate_creature_width(source)?;
    validate_width(written)?;
    if written != (source.input, source.output) {
        return Err(format!(
            "refusing to write creature: input/output {}/{} do not match the source's {}/{}",
            written.0, written.1, source.input, source.output
        ));
    }
    Ok(())
}

/// Assert `candidate` has the same observation width as `incumbent` (and that
/// both are `>= 1`) — a candidate is a mutation of the incumbent, never a
/// different-shaped creature.
pub fn assert_same_width(
    incumbent: &CreatureExport,
    candidate: &CreatureExport,
) -> Result<(), String> {
    assert_written_width(incumbent, (candidate.input, candidate.output))
}

/// Compact creature JSON, refused unless the written text carries the source's
/// (`>= 1`) width. Use in place of `neat_core::creature_to_json` on write.
pub fn checked_creature_json(creature: &CreatureExport) -> Result<String, String> {
    validate_creature_width(creature)?;
    let json = creature_to_json(creature).map_err(|e| e.to_string())?;
    assert_written_width(creature, written_width(&json)?)?;
    Ok(json)
}

/// Pretty creature JSON, refused unless the written text carries the source's
/// (`>= 1`) width. Use in place of `neat_core::creature_to_json_pretty` on write.
pub fn checked_creature_json_pretty(creature: &CreatureExport) -> Result<String, String> {
    validate_creature_width(creature)?;
    let json = creature_to_json_pretty(creature).map_err(|e| e.to_string())?;
    assert_written_width(creature, written_width(&json)?)?;
    Ok(json)
}

/// Read the top-level `input` / `output` of a JSON creature *as written*.
///
/// `Width` names both fields with no `#[serde(default)]`, so a document that
/// omits either is an error, never a zero. The full document is scanned so the
/// answer does not depend on where the serialiser put the keys.
pub fn written_width(json: &str) -> Result<Width, String> {
    #[derive(serde::Deserialize)]
    struct WrittenWidth {
        input: usize,
        output: usize,
    }
    let w: WrittenWidth = serde_json::from_str(json)
        .map_err(|e| format!("written creature has no usable input/output width: {e}"))?;
    Ok((w.input, w.output))
}

/// [`written_width`] over an in-memory JSON value (the check-in writer builds
/// one to re-attach `uuid` / `tags`).
pub fn value_width(value: &Value) -> Result<Width, String> {
    let read = |key: &str| -> Result<usize, String> {
        value
            .get(key)
            .and_then(Value::as_u64)
            .map(|n| n as usize)
            .ok_or_else(|| format!("written creature has no usable `{key}` width"))
    };
    Ok((read("input")?, read("output")?))
}

#[cfg(test)]
mod tests {
    use super::*;

    const ONE_BY_ONE: &str = r#"{
      "input": 1,
      "output": 1,
      "forwardOnly": true,
      "neurons": [{"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY"}],
      "synapses": [{"fromUUID":"input-0","toUUID":"o1","weight":1.0}]
    }"#;

    fn with_width(input: usize, output: usize) -> String {
        ONE_BY_ONE
            .replacen("\"input\": 1", &format!("\"input\": {input}"), 1)
            .replacen("\"output\": 1", &format!("\"output\": {output}"), 1)
    }

    #[test]
    fn load_rejects_zero_input_with_ts_wording() {
        let err = load_creature(&with_width(0, 1)).unwrap_err();
        assert_eq!(err, "Must have at least one input neurons was: 0");
    }

    #[test]
    fn load_rejects_zero_output_with_ts_wording() {
        let err = load_creature(&with_width(1, 0)).unwrap_err();
        assert_eq!(err, "Must have at least one output neurons was: 0");
    }

    #[test]
    fn load_rejects_zero_input_before_zero_output() {
        // Both bad: report the input first, like CreatureValidate.ts.
        let err = load_creature(&with_width(0, 0)).unwrap_err();
        assert!(err.contains("input"), "{err}");
    }

    #[test]
    fn load_rejects_missing_width_no_default() {
        let missing_input = ONE_BY_ONE.replacen("\"input\": 1,", "", 1);
        assert!(load_creature(&missing_input).is_err());
        let missing_output = ONE_BY_ONE.replacen("\"output\": 1,", "", 1);
        assert!(load_creature(&missing_output).is_err());
    }

    #[test]
    fn load_rejects_negative_width() {
        assert!(
            load_creature(&with_width(1, 1).replacen("\"input\": 1", "\"input\": -1", 1)).is_err()
        );
    }

    #[test]
    fn load_accepts_valid_width() {
        let creature = load_creature(&with_width(3, 2)).unwrap();
        assert_eq!((creature.input, creature.output), (3, 2));
    }

    #[test]
    fn checked_json_round_trips_width_byte_identically() {
        let creature = load_creature(&with_width(2511, 3)).unwrap();
        let compact = checked_creature_json(&creature).unwrap();
        assert!(
            compact.starts_with("{\"input\":2511,\"output\":3,"),
            "{compact}"
        );
        assert_eq!(written_width(&compact).unwrap(), (2511, 3));
        let pretty = checked_creature_json_pretty(&creature).unwrap();
        assert_eq!(written_width(&pretty).unwrap(), (2511, 3));
        assert_eq!(load_creature(&pretty).unwrap(), creature);
    }

    #[test]
    fn checked_json_refuses_zero_width_source() {
        let mut creature = load_creature(ONE_BY_ONE).unwrap();
        creature.input = 0;
        assert!(checked_creature_json(&creature).is_err());
        assert!(checked_creature_json_pretty(&creature).is_err());
        let mut creature = load_creature(ONE_BY_ONE).unwrap();
        creature.output = 0;
        assert!(checked_creature_json(&creature).is_err());
    }

    #[test]
    fn written_width_rejects_missing_or_zero() {
        assert!(written_width("{}").is_err());
        assert!(written_width(r#"{"input":1}"#).is_err());
        assert_eq!(written_width(r#"{"input":0,"output":1}"#).unwrap(), (0, 1));
        let source = load_creature(ONE_BY_ONE).unwrap();
        assert!(assert_written_width(&source, (0, 1)).is_err());
        assert!(assert_written_width(&source, (1, 0)).is_err());
    }

    #[test]
    fn assert_written_width_rejects_mismatch() {
        let source = load_creature(&with_width(4, 2)).unwrap();
        assert!(assert_written_width(&source, (4, 2)).is_ok());
        let err = assert_written_width(&source, (3, 2)).unwrap_err();
        assert!(err.contains("3/2") && err.contains("4/2"), "{err}");
        assert!(assert_written_width(&source, (4, 1)).is_err());
    }

    #[test]
    fn assert_same_width_rejects_reshaped_candidate() {
        let incumbent = load_creature(&with_width(4, 2)).unwrap();
        let mut candidate = incumbent.clone();
        assert!(assert_same_width(&incumbent, &candidate).is_ok());
        candidate.input = 5;
        assert!(assert_same_width(&incumbent, &candidate).is_err());
    }

    #[test]
    fn value_width_reads_reattached_document() {
        let value: Value = serde_json::from_str(&with_width(7, 1)).unwrap();
        assert_eq!(value_width(&value).unwrap(), (7, 1));
        assert!(value_width(&serde_json::json!({"output": 1})).is_err());
    }
}
