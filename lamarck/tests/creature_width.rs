//! Issue #165: the binary refuses a creature whose `input` / `output` is `< 1`.
//!
//! The top-level width is the observation width and cannot be re-derived from
//! `neurons` (which lists only non-input neurons), so an `input: 0` creature
//! must stop the run before anything is written — non-zero exit, the TS
//! `CreatureValidate.ts` wording on stderr, no `best.json`, no journal.

use std::fs;
use std::process::Command;

const ONE_BY_ONE: &str = r#"{
  "input": 1,
  "output": 1,
  "forwardOnly": true,
  "neurons": [{"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY"}],
  "synapses": [{"fromUUID":"input-0","toUUID":"o1","weight":1.0}]
}"#;

fn run_with_width(input: usize, output: usize) -> (std::process::Output, std::path::PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let creature = dir.path().join("creature.json");
    fs::write(
        &creature,
        ONE_BY_ONE
            .replacen("\"input\": 1", &format!("\"input\": {input}"), 1)
            .replacen("\"output\": 1", &format!("\"output\": {output}"), 1),
    )
    .unwrap();
    let training = dir.path().join("data");
    fs::create_dir_all(&training).unwrap();
    fs::write(
        training.join("0.bin"),
        [1.0f32, 0.5f32]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let out = dir.path().join("out");
    let output = Command::new(env!("CARGO_BIN_EXE_neat_ai_lamarck"))
        .arg(&creature)
        .arg(&training)
        .arg("--output-dir")
        .arg(&out)
        .arg("--timeout-seconds")
        .arg("1")
        .arg("--max-experiments")
        .arg("1")
        .arg("--skip-phase0")
        // A scorer that cannot exist: a run that got past the width guard
        // would fail on it, so a *width* error proves the guard fired first.
        .arg("--scorer")
        .arg(dir.path().join("no-such-scorer"))
        .output()
        .expect("failed to run neat_ai_lamarck");
    // Keep the tempdir alive until the caller has inspected `out`.
    let out_path = out.clone();
    std::mem::forget(dir);
    (output, out_path)
}

#[test]
fn cli_rejects_zero_input_with_no_output_written() {
    let (output, out) = run_with_width(0, 1);
    assert!(!output.status.success(), "input: 0 must exit non-zero");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("Must have at least one input neurons was: 0"),
        "stderr: {stderr}"
    );
    assert!(
        !out.exists(),
        "no output directory may be created: {stderr}"
    );
}

#[test]
fn cli_rejects_zero_output_with_no_output_written() {
    let (output, out) = run_with_width(1, 0);
    assert!(!output.status.success(), "output: 0 must exit non-zero");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("Must have at least one output neurons was: 0"),
        "stderr: {stderr}"
    );
    assert!(
        !out.exists(),
        "no output directory may be created: {stderr}"
    );
}

#[test]
fn cli_accepts_valid_width_and_gets_past_the_guard() {
    // Same harness, valid width: the run reaches the (missing) scorer, so the
    // failure is the scorer's, not the width guard's.
    let (output, _out) = run_with_width(1, 1);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("Must have at least one"),
        "a 1x1 creature must pass the width guard: {stderr}"
    );
}
