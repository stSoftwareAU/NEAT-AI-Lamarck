//! Campaign-runner ↔ CLI contract for the #96 exclusive-box arms.
//!
//! Each arm below costs exclusive box time on the production creature, so the
//! flags it passes have to be right *before* the run starts — a rejected flag
//! discovered 20 minutes in wastes the whole slot. These tests drive
//! `scripts/run-followup-economics.sh` against a stub optimiser that records
//! its argv, then check the real binary accepts what the stub was handed.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use tempfile::tempdir;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .canonicalize()
        .expect("repository root resolves")
}

/// Stub `neat_ai_lamarck`: appends its argv to `$ARGV_LOG`, then produces the
/// journal / report the runner insists on so the arm completes.
const STUB_LAMARCK: &str = r#"#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >>"$ARGV_LOG"
if [[ "${1:-}" == "report" ]]; then
  echo '{"experiments":0}'
  exit 0
fi
out="."
prev=""
for arg in "$@"; do
  if [[ "$prev" == "--output-dir" ]]; then out="$arg"; fi
  prev="$arg"
done
mkdir -p "$out"
echo '{"record":"runHeader"}' >"$out/experiments.jsonl"
"#;

struct Campaign {
    dir: tempfile::TempDir,
    argv_log: PathBuf,
}

impl Campaign {
    /// Runner harness: stub optimiser + stub scorer + a creature and data dir.
    fn new() -> Self {
        let dir = tempdir().expect("temp dir");
        let root = dir.path();
        let stub = root.join("lamarck-stub.sh");
        fs::write(&stub, STUB_LAMARCK).expect("write stub");
        let scorer = root.join("rust_scorer-stub.sh");
        fs::write(&scorer, "#!/usr/bin/env bash\nexit 0\n").expect("write scorer stub");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            for path in [&stub, &scorer] {
                fs::set_permissions(path, fs::Permissions::from_mode(0o755)).expect("chmod stub");
            }
        }
        fs::write(root.join("creature.json"), "{}").expect("write creature");
        fs::create_dir_all(root.join("train-data")).expect("create training dir");
        Self {
            argv_log: root.join("argv.log"),
            dir,
        }
    }

    /// Run one arm of the campaign runner; returns the stub's recorded argv.
    fn run_arm(&self, arm: &str, extra_env: &[(&str, &str)]) -> Result<Vec<String>, String> {
        let root = self.dir.path();
        let mut command = Command::new("bash");
        command
            .arg(repo_root().join("scripts/run-followup-economics.sh"))
            .arg(arm)
            .current_dir(root)
            .env("LAMARCK", root.join("lamarck-stub.sh"))
            .env("SCORER", root.join("rust_scorer-stub.sh"))
            .env("CREATURE", root.join("creature.json"))
            .env("TRAIN_DATA", root.join("train-data"))
            .env("OUT_DIR", root.join("out"))
            .env("ARGV_LOG", &self.argv_log);
        for (key, value) in extra_env {
            command.env(key, value);
        }
        let output = command.output().expect("runner starts");
        if !output.status.success() {
            return Err(String::from_utf8_lossy(&output.stderr).into_owned());
        }
        let log = fs::read_to_string(&self.argv_log).unwrap_or_default();
        Ok(log
            .lines()
            .filter(|line| !line.starts_with("report "))
            .map(str::to_string)
            .collect())
    }
}

#[test]
fn the_output_neuron_arm_pins_the_focus_to_an_output() {
    let campaign = Campaign::new();
    let runs = campaign
        .run_arm("output-neuron", &[("OUTPUT_NEURON_SECONDS", "1")])
        .expect("output-neuron arm completes");
    assert_eq!(runs.len(), 1, "the output slice is a single run: {runs:?}");
    let argv = &runs[0];
    assert!(
        argv.contains("--focus-neuron output-0"),
        "the slice must pin an output neuron, got: {argv}"
    );
    assert!(
        !argv.contains("--focus-policy"),
        "a policy would override the whole point of the slice: {argv}"
    );
}

#[test]
fn the_output_neuron_arm_honours_an_overridden_neuron() {
    let campaign = Campaign::new();
    let runs = campaign
        .run_arm(
            "output-neuron",
            &[
                ("OUTPUT_NEURON", "output-3"),
                ("OUTPUT_NEURON_SECONDS", "1"),
            ],
        )
        .expect("output-neuron arm completes");
    assert!(
        runs[0].contains("--focus-neuron output-3"),
        "OUTPUT_NEURON must select the pinned neuron, got: {}",
        runs[0]
    );
}

#[test]
fn the_backprop_cap_arm_varies_the_cap_and_nothing_else() {
    let campaign = Campaign::new();
    let runs = campaign
        .run_arm(
            "backprop-cap",
            &[("BACKPROP_CAPS", "10 0.001"), ("CAP_SECONDS", "1")],
        )
        .expect("backprop-cap arm completes");
    assert_eq!(runs.len(), 2, "one run per cap: {runs:?}");
    assert!(runs[0].contains("--backprop-max-bias-adjustment-scale 10"));
    assert!(runs[1].contains("--backprop-max-bias-adjustment-scale 0.001"));
    for argv in &runs {
        assert!(
            argv.contains("--seed 51"),
            "both caps share a seed so only the cap moves: {argv}"
        );
        assert!(
            !argv.contains("--backprop-learning-rate"),
            "the rate is the knob #75 already showed is inert here: {argv}"
        );
    }
}

/// Issue #108: the paired batch-economics benchmark varies only whether the
/// generator's quotas scale — same seed, same budget, same `--candidates`.
#[test]
fn the_candidate_quotas_arm_varies_only_the_quota_scaling() {
    let campaign = Campaign::new();
    let runs = campaign
        .run_arm(
            "candidate-quotas",
            &[("QUOTA_SECONDS", "1"), ("QUOTA_CANDIDATES", "100")],
        )
        .expect("candidate-quotas arm completes");
    assert_eq!(runs.len(), 2, "the A/B is a pair of runs: {runs:?}");
    assert!(
        !runs[0].contains("--scale-candidate-quotas"),
        "the control must run at the fixed ceiling: {}",
        runs[0]
    );
    assert!(
        runs[1].contains("--scale-candidate-quotas"),
        "the treatment must scale the quotas: {}",
        runs[1]
    );
    for argv in &runs {
        assert!(
            argv.contains("--candidates 100") && argv.contains("--seed 61"),
            "both sides share the budget and the seed: {argv}"
        );
    }
}

#[test]
fn an_unknown_arm_fails_loudly() {
    let campaign = Campaign::new();
    let err = campaign
        .run_arm("no-such-arm", &[])
        .expect_err("an unknown arm must abort the campaign");
    assert!(
        err.contains("unknown arm"),
        "the runner should name the fault: {err}"
    );
}

#[test]
fn the_binary_accepts_the_flags_the_campaign_arms_pass() {
    let help = Command::new(env!("CARGO_BIN_EXE_neat_ai_lamarck"))
        .arg("--help")
        .output()
        .expect("binary runs");
    let help = String::from_utf8_lossy(&help.stdout);
    for flag in [
        "--focus-neuron",
        "--backprop-learning-rate",
        "--backprop-max-bias-adjustment-scale",
        "--scale-candidate-quotas",
    ] {
        assert!(help.contains(flag), "the binary has no `{flag}`");
    }
}

#[test]
fn a_non_positive_bias_cap_aborts_before_the_run() {
    let dir = tempdir().expect("temp dir");
    let output = Command::new(env!("CARGO_BIN_EXE_neat_ai_lamarck"))
        .arg(dir.path().join("creature.json"))
        .arg(dir.path())
        .arg("--backprop-max-bias-adjustment-scale")
        .arg("0")
        .output()
        .expect("binary runs");
    assert!(
        !output.status.success(),
        "a zero cap must abort rather than silently keep the default"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--backprop-max-bias-adjustment-scale"),
        "the failure must name the flag: {stderr}"
    );
}
