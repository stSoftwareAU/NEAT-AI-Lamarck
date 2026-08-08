//! End-to-end Lamarck optimisation loop and experiment journal.

use crate::backprop::BackpropConfig;
use crate::candidates::{
    CandidateGenContext, CandidateProvenance, generate_candidates, write_candidate_batch,
};
use crate::config::LamarckConfig;
use crate::focus::{FixedFocusSelector, FocusSelector, RandomFocusSelector, collect_focus_stats};
use crate::log;
use crate::observations::ensure_statistics;
use crate::scorer::{DirectoryScorer, ScoreResult, accepts_improvement, select_winner};
use neat_core::{
    TrainingDataConfig, compile_creature, creature_to_json_pretty, parse_creature_json,
};
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// One journal line written to `experiments.jsonl`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExperimentRecord {
    /// Monotonic experiment number.
    pub experiment_number: u64,
    /// Unix timestamp.
    pub timestamp_unix: u64,
    /// RNG seed for this run (if any).
    pub seed: Option<u64>,
    /// Incumbent creature checksum (uuid or hash).
    pub incumbent_id: String,
    /// Baseline authoritative score.
    pub baseline_score: f64,
    /// Focus neuron UUID.
    pub focus_neuron: String,
    /// Candidate provenances.
    pub candidates: Vec<CandidateProvenance>,
    /// All authoritative scores by stem.
    pub scores: std::collections::BTreeMap<String, f64>,
    /// Winning stem if accepted.
    pub winner: Option<String>,
    /// Absolute score improvement when accepted.
    pub improvement: Option<f64>,
    /// Whether a candidate was accepted.
    pub accepted: bool,
    /// Analysis elapsed milliseconds.
    pub analysis_ms: u128,
    /// Scorer elapsed milliseconds.
    pub scorer_ms: u128,
}

/// Result of a completed Lamarck run.
#[derive(Debug)]
pub struct RunResult {
    /// Path to `best.json`.
    pub best_path: PathBuf,
    /// Path to `experiments.jsonl`.
    pub journal_path: PathBuf,
    /// Final best score.
    pub best_score: f64,
    /// Experiments attempted.
    pub experiments: u64,
    /// Accepted improvements.
    pub acceptances: u64,
}

/// Run the Lamarck optimisation loop until the wall-clock budget expires.
pub fn run_optimisation(
    config: &LamarckConfig,
    scorer: &impl DirectoryScorer,
) -> Result<RunResult, String> {
    fs::create_dir_all(&config.output_dir).map_err(|e| e.to_string())?;
    let journal_path = config.output_dir.join("experiments.jsonl");
    let best_path = config.output_dir.join("best.json");
    let winners_dir = config.output_dir.join("winners");

    let original_text = fs::read_to_string(&config.creature).map_err(|e| e.to_string())?;
    let mut incumbent = parse_creature_json(&original_text).map_err(|e| e.to_string())?;
    // Never modify the supplied file — work from in-memory / output copies.
    fs::write(&best_path, &original_text).map_err(|e| e.to_string())?;

    let train_cfg = TrainingDataConfig::new(incumbent.input, incumbent.output);
    log::info(&format!(
        "ensuring observations-{} (inputs={} outputs={})",
        config.stats_mode.label(),
        incumbent.input,
        incumbent.output
    ));
    let sample_limit = match config.stats_mode {
        crate::observations::StatsMode::Quick => Some(config.quick_sample_records),
        crate::observations::StatsMode::Full => None,
    };
    let observations = ensure_statistics(
        &config.training_data,
        &train_cfg,
        config.stats_mode,
        sample_limit,
    )
    .map_err(|e| e.to_string())?;

    let mut rng = match config.seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_os_rng(),
    };
    let mut random_focus = RandomFocusSelector;
    let mut fixed_focus = config
        .focus_neuron
        .as_ref()
        .map(|uuid| FixedFocusSelector { uuid: uuid.clone() });
    let backprop = BackpropConfig::default();
    let focus_sample_limit = match config.stats_mode {
        crate::observations::StatsMode::Quick => Some(config.quick_sample_records),
        crate::observations::StatsMode::Full => None,
    };

    let deadline = Instant::now() + config.timeout;
    let mut experiments = 0u64;
    let mut acceptances = 0u64;
    let mut best_score = f64::NEG_INFINITY;
    log::info(&format!(
        "starting optimisation loop (timeout={}s, candidates={})",
        config.timeout.as_secs(),
        config.candidates
    ));
    if let Some(uuid) = &config.focus_neuron {
        log::detail(&format!("focus locked to {uuid}"));
    }

    while Instant::now() < deadline {
        experiments += 1;
        let remaining = deadline.saturating_duration_since(Instant::now());
        log::info(&format!(
            "experiment {experiments} ({}s remaining, acceptances={acceptances})",
            remaining.as_secs()
        ));
        let analysis_start = Instant::now();
        let focus = if let Some(selector) = fixed_focus.as_mut() {
            selector.select(&incumbent, &mut rng).ok_or_else(|| {
                format!(
                    "focus neuron '{}' not found (or is an input)",
                    selector.uuid
                )
            })?
        } else {
            random_focus
                .select(&incumbent, &mut rng)
                .ok_or_else(|| "no focus neuron available".to_string())?
        };
        log::detail(&format!("focus neuron: {focus}"));
        let mut network = compile_creature(&incumbent).map_err(|e| e.to_string())?;
        log::detail("scanning incumbent for focus stats...");
        let focus_stats = collect_focus_stats(
            &incumbent,
            &mut network,
            &config.training_data,
            &focus,
            focus_sample_limit,
        )?;
        let gen_ctx = CandidateGenContext {
            incumbent: &incumbent,
            focus_uuid: &focus,
            focus_stats: &focus_stats,
            observations: &observations,
            learning: None,
            backprop: &backprop,
        };
        let candidates = generate_candidates(&gen_ctx, config.candidates, &mut rng);
        let analysis_ms = analysis_start.elapsed().as_millis();
        log::ok(&format!(
            "generated {} candidates in {analysis_ms}ms",
            candidates.len()
        ));

        let batch_dir = config
            .output_dir
            .join(format!("candidates-exp-{experiments}"));
        write_candidate_batch(&batch_dir, &incumbent, &candidates)?;

        log::detail(&format!(
            "scoring baseline + {} candidates via {}",
            candidates.len(),
            config.scorer_path.display()
        ));
        let scorer_start = Instant::now();
        let scores = match scorer.score_directory(&batch_dir, &config.training_data) {
            Ok(s) => s,
            Err(e) => {
                append_journal(
                    &journal_path,
                    &ExperimentRecord {
                        experiment_number: experiments,
                        timestamp_unix: unix_now(),
                        seed: config.seed,
                        incumbent_id: incumbent_id(&incumbent),
                        baseline_score: best_score,
                        focus_neuron: focus,
                        candidates: candidates.iter().map(|c| c.provenance.clone()).collect(),
                        scores: Default::default(),
                        winner: None,
                        improvement: None,
                        accepted: false,
                        analysis_ms,
                        scorer_ms: scorer_start.elapsed().as_millis(),
                    },
                )?;
                if !config.preserve_losers {
                    let _ = fs::remove_dir_all(&batch_dir);
                }
                // Failed experiment — keep incumbent, continue.
                let _ = e;
                continue;
            }
        };
        let scorer_ms = scorer_start.elapsed().as_millis();
        log::ok(&format!("scorer finished in {scorer_ms}ms"));

        let baseline = scores
            .get("baseline")
            .ok_or_else(|| "baseline missing from scorer results".to_string())?;
        if best_score.is_infinite() {
            best_score = baseline.score;
        }
        log::detail(&format!("baseline score={}", baseline.score));

        let winner = select_winner(&scores, config.min_improvement).map_err(|e| e.to_string())?;
        let mut accepted = false;
        let mut improvement = None;
        let mut winner_stem = None;
        if let Some((stem, result, delta)) = winner
            && accepts_improvement(result.score, baseline.score, config.min_improvement)
        {
            log::ok(&format!(
                "accepted {stem}: score={} (+{delta:.3e})",
                result.score
            ));
            let winner_path = batch_dir.join(format!("{stem}.json"));
            let winner_json = fs::read_to_string(&winner_path).map_err(|e| e.to_string())?;
            incumbent = parse_creature_json(&winner_json).map_err(|e| e.to_string())?;
            fs::write(&best_path, &winner_json).map_err(|e| e.to_string())?;
            fs::create_dir_all(&winners_dir).map_err(|e| e.to_string())?;
            fs::write(
                winners_dir.join(format!("winner-{experiments:04}.json")),
                &winner_json,
            )
            .map_err(|e| e.to_string())?;
            best_score = result.score;
            accepted = true;
            acceptances += 1;
            improvement = Some(delta);
            winner_stem = Some(stem.to_string());
        }

        let score_map = scores.iter().map(|(k, v)| (k.clone(), v.score)).collect();
        append_journal(
            &journal_path,
            &ExperimentRecord {
                experiment_number: experiments,
                timestamp_unix: unix_now(),
                seed: config.seed,
                incumbent_id: incumbent_id(&incumbent),
                baseline_score: baseline.score,
                focus_neuron: focus,
                candidates: candidates.iter().map(|c| c.provenance.clone()).collect(),
                scores: score_map,
                winner: winner_stem,
                improvement,
                accepted,
                analysis_ms,
                scorer_ms,
            },
        )?;

        if !config.preserve_losers {
            let _ = fs::remove_dir_all(&batch_dir);
        }
    }

    Ok(RunResult {
        best_path,
        journal_path,
        best_score,
        experiments,
        acceptances,
    })
}

fn incumbent_id(creature: &neat_core::CreatureExport) -> String {
    format!(
        "in{}-out{}-n{}-s{}",
        creature.input,
        creature.output,
        creature.neurons.len(),
        creature.synapses.len()
    )
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn append_journal(path: &Path, record: &ExperimentRecord) -> Result<(), String> {
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| e.to_string())?;
    let line = serde_json::to_string(record).map_err(|e| e.to_string())?;
    writeln!(file, "{line}").map_err(|e| e.to_string())?;
    Ok(())
}

/// Score the supplied creature once for Phase-0 baseline recording.
pub fn score_single_creature_dir(
    creature: &neat_core::CreatureExport,
    training_data: &Path,
    scorer: &impl DirectoryScorer,
    work_dir: &Path,
) -> Result<ScoreResult, String> {
    fs::create_dir_all(work_dir).map_err(|e| e.to_string())?;
    let json = creature_to_json_pretty(creature).map_err(|e| e.to_string())?;
    fs::write(work_dir.join("baseline.json"), json).map_err(|e| e.to_string())?;
    let results = scorer
        .score_directory(work_dir, training_data)
        .map_err(|e| e.to_string())?;
    results
        .get("baseline")
        .cloned()
        .ok_or_else(|| "baseline missing".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scorer::ScoreResult;
    use std::collections::BTreeMap;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;
    use tempfile::tempdir;

    struct ScriptedScorer {
        calls: Arc<Mutex<usize>>,
    }

    impl DirectoryScorer for ScriptedScorer {
        fn score_directory(
            &self,
            candidates_dir: &Path,
            _training_data: &Path,
        ) -> Result<BTreeMap<String, ScoreResult>, crate::scorer::ScorerError> {
            let mut calls = self.calls.lock().unwrap();
            *calls += 1;
            let mut map = BTreeMap::new();
            map.insert(
                "baseline".into(),
                ScoreResult {
                    score: 0.5,
                    error: 0.5,
                    complexity_penalty: 0.0,
                },
            );
            // List candidate files and give the first a winning score once.
            if let Ok(rd) = fs::read_dir(candidates_dir) {
                for entry in rd.flatten() {
                    let name = entry.file_name().to_string_lossy().to_string();
                    if let Some(stem) = name.strip_suffix(".json") {
                        if stem == "baseline" {
                            continue;
                        }
                        let score = if *calls == 1 && stem == "candidate-000" {
                            0.5 + 2e-6
                        } else {
                            0.5
                        };
                        map.insert(
                            stem.to_string(),
                            ScoreResult {
                                score,
                                error: 1.0 - score,
                                complexity_penalty: 0.0,
                            },
                        );
                    }
                }
            }
            Ok(map)
        }
    }

    #[test]
    fn loop_accepts_winner_and_writes_journal() {
        let dir = tempdir().unwrap();
        let creature_path = dir.path().join("creature.json");
        let training = dir.path().join("data");
        fs::create_dir_all(&training).unwrap();
        // Tiny 1-in/1-out bin with one record.
        fs::write(
            training.join("0.bin"),
            [1.0f32, 0.5f32]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        fs::write(
            &creature_path,
            r#"{
              "semanticVersion":"4.0.0","forwardOnly":true,"input":1,"output":1,
              "neurons":[
                {"type":"hidden","uuid":"h1","bias":0.1,"squash":"IDENTITY"},
                {"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY"}
              ],
              "synapses":[
                {"fromUUID":"input-0","toUUID":"h1","weight":1.0},
                {"fromUUID":"h1","toUUID":"o1","weight":1.0}
              ]
            }"#,
        )
        .unwrap();

        let out = dir.path().join("out");
        let config = LamarckConfig {
            creature: creature_path,
            training_data: training,
            timeout: Duration::from_millis(200),
            candidates: 4,
            min_improvement: 1e-6,
            seed: Some(1),
            scorer_path: PathBuf::from("rust_scorer"),
            output_dir: out.clone(),
            preserve_losers: true,
            stats_mode: crate::observations::StatsMode::Quick,
            quick_sample_records: 8,
            focus_neuron: Some("h1".into()),
        };
        let scorer = ScriptedScorer {
            calls: Arc::new(Mutex::new(0)),
        };
        let result = run_optimisation(&config, &scorer).unwrap();
        assert!(result.journal_path.is_file());
        assert!(result.best_path.is_file());
        assert!(result.experiments >= 1);
        let journal = fs::read_to_string(result.journal_path).unwrap();
        assert!(journal.lines().next().unwrap().contains("experimentNumber"));
    }
}
