//! Paired benchmark for multi-focus experiments (issue #109).
//!
//! Runs the real optimisation loop once per `--focus-count` arm over identical
//! inputs — same creature, same sample, same seed, same wall-clock budget —
//! and reports what the issue asks for: candidates per analysis-minute,
//! promote-scores per scorer-minute, accepts, and improvement per wall-clock
//! hour.
//!
//! The scorer is in-process (local MSE over the same corpus) so the benchmark
//! needs no `rust_scorer` binary and every arm pays the same per-creature
//! scoring cost; only the analysis fan-out differs.
//!
//! Usage (release build — debug timings are meaningless):
//!
//! ```text
//! cargo run --release --example focus_fanout_bench -- [SECONDS] [RECORDS] [INPUTS] [HIDDEN] [CANDIDATES] [K,...]
//! ```
//!
//! The default `MIN_IMPROVEMENT` regime is deliberately accept-free (`1`), the
//! shape `docs/followup-economics.md` measured, so the arms are compared on
//! throughput rather than on a lucky accept.

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

/// One arm's measured economics.
#[derive(Debug, Clone, Copy)]
struct ArmResult {
    candidates_per_analysis_minute: f64,
    promote_per_scorer_minute: f64,
    experiments: u64,
    generated: u64,
    accepts: u64,
    improvement_per_hour: f64,
}

fn per_minute(count: u64, ms: u128) -> f64 {
    if ms == 0 {
        return 0.0;
    }
    count as f64 * 60_000.0 / ms as f64
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
    let candidates: usize = arg(4, "12").parse().expect("CANDIDATES must be a number");
    let arms: Vec<usize> = arg(5, "1,3")
        .split(',')
        .map(|k| k.trim().parse().expect("K must be a number"))
        .collect();
    let min_improvement: f64 = arg(6, "1")
        .parse()
        .expect("MIN_IMPROVEMENT must be a number");
    let repeats: usize = arg(7, "3").parse().expect("REPEATS must be a number");
    assert!(repeats > 0, "REPEATS must be at least 1");

    let dir = tempfile::tempdir().unwrap();
    let training = dir.path().join("data");
    std::fs::create_dir_all(&training).unwrap();
    write_sample(&training, records, inputs);
    let creature_path = dir.path().join("creature.json");
    std::fs::write(&creature_path, creature_json(inputs, hidden)).unwrap();

    println!(
        "records={records} inputs={inputs} hidden={hidden} candidates={candidates}/focus \
         budget={seconds}s min_improvement={min_improvement:e} seed=7 (same every arm)"
    );
    // Two budget shapes per focus count, because they answer different
    // questions: a fixed total budget holds the batch size still and shows what
    // the fan-out costs, and a fixed per-focus budget shows the bigger, more
    // diverse batch the shared analysis buys (issue #108 + #109).
    let plans: Vec<(usize, usize)> = arms
        .iter()
        .flat_map(|k| {
            let mut budgets = vec![(*k, candidates)];
            if *k > 1 {
                budgets.push((*k, candidates * k));
            }
            budgets
        })
        .collect();

    let mut best: BTreeMap<(usize, usize), ArmResult> = BTreeMap::new();
    // Interleave the repeats so machine drift hits every arm, and keep the best
    // rate per arm — the least-noise estimator on a contended box.
    for repeat in 0..repeats {
        for (k, budget) in &plans {
            let out = dir.path().join(format!("out-k{k}-c{budget}-r{repeat}"));
            let config = LamarckConfig {
                creature: creature_path.clone(),
                training_data: training.clone(),
                timeout: Duration::from_secs(seconds),
                candidates: *budget,
                focus_count: *k,
                min_improvement,
                seed: Some(7),
                output_dir: out,
                stats_mode: StatsMode::Quick,
                quick_sample_records: records as u64,
                focus_policy: FocusPolicy::Weighted,
                phase0_parity: false,
                screen_sample_rate: None,
                ..LamarckConfig::default()
            };

            let start = Instant::now();
            let result = run_optimisation(&config, &LocalMseScorer).expect("run completes");
            let wall = start.elapsed().as_secs_f64();
            let report = report_from_journal(&result.journal_path).expect("journal reports");
            let generated =
                (report.candidate_batch.mean_generated * report.experiments as f64) as u64;
            let row = ArmResult {
                candidates_per_analysis_minute: per_minute(generated, report.total_analysis_ms),
                promote_per_scorer_minute: per_minute(
                    report.candidates_scored,
                    report.total_scorer_ms,
                ),
                experiments: result.experiments,
                generated,
                accepts: result.acceptances,
                improvement_per_hour: report.total_score_improvement.unwrap_or(0.0) * 3600.0
                    / wall.max(f64::EPSILON),
            };
            eprintln!(
                "  k={k} candidates={budget} repeat={repeat}: {:.1} candidates/analysis-min, \
                 {:.1} promote/scorer-min",
                row.candidates_per_analysis_minute, row.promote_per_scorer_minute
            );
            let entry = best.entry((*k, *budget)).or_insert(row);
            if row.candidates_per_analysis_minute > entry.candidates_per_analysis_minute {
                *entry = row;
            }
        }
    }

    println!("\nbest of {repeats} interleaved repeats:");
    println!(
        "\n| --focus-count | --candidates | experiments | candidates | candidates/analysis-min | \
         promote scores/scorer-min | accepts | Δscore/hour |"
    );
    println!("|---|---|---|---|---|---|---|---|");
    for ((k, budget), arm) in &best {
        println!(
            "| {k} | {budget} | {} | {} | {:.1} | {:.1} | {} | {:.3e} |",
            arm.experiments,
            arm.generated,
            arm.candidates_per_analysis_minute,
            arm.promote_per_scorer_minute,
            arm.accepts,
            arm.improvement_per_hour
        );
    }
}
