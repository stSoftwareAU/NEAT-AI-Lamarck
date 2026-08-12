//! Paired benchmark for the cross-experiment analysis memo (issue #106).
//!
//! Runs the real optimisation loop twice over identical inputs — same creature,
//! same sample, same seed, same wall-clock budget — with the memo off and on,
//! and reports experiments completed and score improvement per wall-clock hour.
//!
//! The scorer is in-process (local MSE over the same corpus) so the benchmark
//! needs no `rust_scorer` binary and both arms pay exactly the same scoring
//! cost; only the analysis phase differs.
//!
//! Usage (release build — debug timings are meaningless):
//!
//! ```text
//! cargo run --release --example analysis_memo_bench -- [SECONDS] [RECORDS] [INPUTS] [HIDDEN] [MIN_IMPROVEMENT]
//! ```
//!
//! `MIN_IMPROVEMENT` selects the regime. The default `1e-6` accepts often, which
//! is *not* the production shape; pass a large value (e.g. `1`) to model the
//! accept-free stretch `docs/followup-economics.md` measured (0 accepts in 118
//! experiments) — the regime the memo is aimed at.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;
use std::time::{Duration, Instant};

use neat_ai_lamarck::focus::FocusPolicy;
use neat_ai_lamarck::observations::StatsMode;
use neat_ai_lamarck::scorer::{DirectoryScorer, ScoreResult, ScoreSample, ScorerError};
use neat_ai_lamarck::{LamarckConfig, compute_local_mse, report_from_journal, run_optimisation};
use neat_core::{compile_creature, parse_creature_json};

/// Scores every creature in a directory by local MSE over the training corpus.
struct LocalMseScorer;

impl DirectoryScorer for LocalMseScorer {
    fn score_directory_sampled(
        &self,
        candidates_dir: &Path,
        training_data: &Path,
        _sample: ScoreSample,
    ) -> Result<BTreeMap<String, ScoreResult>, ScorerError> {
        let mut scores = BTreeMap::new();
        let entries =
            std::fs::read_dir(candidates_dir).map_err(|e| ScorerError::Process(e.to_string()))?;
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            let Some(stem) = name.strip_suffix(".json") else {
                continue;
            };
            let text = std::fs::read_to_string(entry.path())
                .map_err(|e| ScorerError::Json(e.to_string()))?;
            let creature =
                parse_creature_json(&text).map_err(|e| ScorerError::Json(e.to_string()))?;
            let mut network =
                compile_creature(&creature).map_err(|e| ScorerError::Invalid(e.to_string()))?;
            let (mse, _) = compute_local_mse(&creature, &mut network, training_data)
                .map_err(ScorerError::Invalid)?;
            scores.insert(
                stem.to_string(),
                ScoreResult {
                    score: 1.0 - mse,
                    error: mse,
                    complexity_penalty: 0.0,
                },
            );
        }
        Ok(scores)
    }
}

/// Synthetic creature: `inputs` inputs, `hidden` TANH hiddens, one output.
fn creature_json(inputs: usize, hidden: usize) -> String {
    let mut neurons = String::new();
    let mut synapses = String::new();
    for h in 0..hidden {
        if h > 0 {
            neurons.push(',');
        }
        let bias = (h as f64 % 7.0) * 0.01 - 0.03;
        neurons.push_str(&format!(
            r#"{{"type":"hidden","uuid":"h{h}","bias":{bias},"squash":"TANH"}}"#
        ));
        for k in 0..4 {
            let i = (h * 4 + k) % inputs;
            let weight = 0.05 + ((h + k) as f64 % 11.0) * 0.01;
            synapses.push_str(&format!(
                r#"{{"fromUUID":"input-{i}","toUUID":"h{h}","weight":{weight}}},"#
            ));
        }
        synapses.push_str(&format!(
            r#"{{"fromUUID":"h{h}","toUUID":"o1","weight":{}}},"#,
            0.02 + (h as f64 % 5.0) * 0.01
        ));
    }
    let synapses = synapses.trim_end_matches(',');
    format!(
        r#"{{"semanticVersion":"4.0.0","forwardOnly":true,"input":{inputs},"output":1,
           "neurons":[{neurons},{{"type":"output","uuid":"o1","bias":0.01,"squash":"IDENTITY"}}],
           "synapses":[{synapses}]}}"#
    )
}

/// Deterministic xorshift sample: `inputs` inputs + one target per record.
fn write_sample(dir: &Path, records: usize, inputs: usize) {
    let mut file = std::io::BufWriter::new(std::fs::File::create(dir.join("0.bin")).unwrap());
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    for _ in 0..records {
        for _ in 0..=inputs {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let v = ((state >> 11) as f64 / (1u64 << 53) as f64) as f32 * 2.0 - 1.0;
            file.write_all(&v.to_le_bytes()).unwrap();
        }
    }
    file.flush().unwrap();
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let arg = |i: usize, default: &str| -> String {
        args.get(i).cloned().unwrap_or_else(|| default.to_string())
    };
    let seconds: u64 = arg(0, "60").parse().expect("SECONDS must be a number");
    let records: usize = arg(1, "20000").parse().expect("RECORDS must be a number");
    let inputs: usize = arg(2, "128").parse().expect("INPUTS must be a number");
    let hidden: usize = arg(3, "24").parse().expect("HIDDEN must be a number");
    let min_improvement: f64 = arg(4, "1e-6")
        .parse()
        .expect("MIN_IMPROVEMENT must be a number");

    let dir = tempfile::tempdir().unwrap();
    let training = dir.path().join("data");
    std::fs::create_dir_all(&training).unwrap();
    write_sample(&training, records, inputs);
    let creature_path = dir.path().join("creature.json");
    std::fs::write(&creature_path, creature_json(inputs, hidden)).unwrap();

    println!(
        "records={records} inputs={inputs} hidden={hidden} budget={seconds}s \
         min_improvement={min_improvement:e} seed=7 (same both arms)"
    );

    for (label, entries) in [("memo off", 0usize), ("memo on", 16usize)] {
        let out = dir.path().join(format!("out-{entries}"));
        let config = LamarckConfig {
            creature: creature_path.clone(),
            training_data: training.clone(),
            timeout: Duration::from_secs(seconds),
            candidates: 8,
            min_improvement,
            seed: Some(7),
            output_dir: out,
            stats_mode: StatsMode::Quick,
            quick_sample_records: records as u64,
            focus_policy: FocusPolicy::Weighted,
            phase0_parity: false,
            screen_sample_rate: None,
            analysis_memo_entries: entries,
            ..LamarckConfig::default()
        };

        let start = Instant::now();
        let result = run_optimisation(&config, &LocalMseScorer).expect("run completes");
        let wall = start.elapsed().as_secs_f64();
        let report = report_from_journal(&result.journal_path).expect("journal reports");
        let improvement = report.total_score_improvement.unwrap_or(0.0);

        println!(
            "{label:>8}: {exp} experiments  {acc} accept(s)  analysis {analysis}ms  \
             memo {hits}h/{misses}m saved {saved}ms  Δscore/hour {per_hour:.3e}  ({wall:.1}s wall)",
            exp = result.experiments,
            acc = result.acceptances,
            analysis = report.total_analysis_ms,
            hits = report.analysis_memo.hits,
            misses = report.analysis_memo.misses,
            saved = report.analysis_memo.ms_saved,
            per_hour = improvement * 3600.0 / wall.max(f64::EPSILON),
        );
    }
}
