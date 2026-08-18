//! Paired benchmark for the remembered full-corpus baseline (issue #113).
//!
//! Runs the real optimisation loop twice over identical inputs — same creature,
//! same corpus, same seed, same wall-clock budget — with
//! `--baseline-reverify-interval` off and on, and reports promote-phase scorer
//! milliseconds per experiment and experiments completed.
//!
//! The scorer is in-process (local MSE over the same corpus) so the benchmark
//! needs no `rust_scorer` binary, and it is **tiered like production**: a screen
//! call scores over a 5% slice of the corpus, a promote call over all of it. The
//! per-creature cost is therefore real work in both tiers, and the only
//! difference between the arms is whether the promote call carries the
//! incumbent.
//!
//! Usage (release build — debug timings are meaningless):
//!
//! ```text
//! cargo run --release --example promote_baseline_bench -- [SECONDS] [RECORDS] [INPUTS] [HIDDEN] [MIN_IMPROVEMENT]
//! ```
//!
//! `MIN_IMPROVEMENT` defaults to `1`, which models the accept-free stretch
//! `docs/followup-economics.md` measured (0 accepts in 118 experiments) — the
//! regime the saving is aimed at, because a promote call that rejects is the
//! one that never needed the baseline re-scored.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use neat_ai_lamarck::focus::FocusPolicy;
use neat_ai_lamarck::observations::StatsMode;
use neat_ai_lamarck::scorer::{DirectoryScorer, ScoreResult, ScoreSample, ScorerError};
use neat_ai_lamarck::{
    LamarckConfig, compute_local_mse, load_creature, report_from_journal, run_optimisation,
};
use neat_core::compile_creature;

/// Scores a directory by local MSE, over the sample corpus for a screen call
/// and the full corpus for a promote call.
struct TieredMseScorer {
    /// 5% slice of the corpus, standing in for `--sample-rate`.
    sample_corpus: PathBuf,
}

impl DirectoryScorer for TieredMseScorer {
    fn score_directory_sampled(
        &self,
        candidates_dir: &Path,
        training_data: &Path,
        sample: ScoreSample,
    ) -> Result<BTreeMap<String, ScoreResult>, ScorerError> {
        let corpus = if sample.is_subsample() {
            self.sample_corpus.as_path()
        } else {
            training_data
        };
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
            let creature = load_creature(&text).map_err(ScorerError::Json)?;
            let mut network =
                compile_creature(&creature).map_err(|e| ScorerError::Invalid(e.to_string()))?;
            let (mse, _) =
                compute_local_mse(&creature, &mut network, corpus).map_err(ScorerError::Invalid)?;
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
    let min_improvement: f64 = arg(4, "1")
        .parse()
        .expect("MIN_IMPROVEMENT must be a number");
    // Arms run sequentially, so box load drifting between them shows up as a
    // wall-clock difference the arms did not cause. Alternating sweeps let a
    // reader take medians instead of trusting one ordering.
    let repeats: usize = arg(5, "1").parse().expect("REPEATS must be a number");

    let dir = tempfile::tempdir().unwrap();
    let training = dir.path().join("data");
    let sample_corpus = dir.path().join("data-5pc");
    std::fs::create_dir_all(&training).unwrap();
    std::fs::create_dir_all(&sample_corpus).unwrap();
    write_sample(&training, records, inputs);
    write_sample(&sample_corpus, records.div_ceil(20), inputs);
    let creature_path = dir.path().join("creature.json");
    std::fs::write(&creature_path, creature_json(inputs, hidden)).unwrap();

    println!(
        "records={records} (screen tier {}) inputs={inputs} hidden={hidden} budget={seconds}s \
         min_improvement={min_improvement:e} seed=7 (same both arms)",
        records.div_ceil(20)
    );

    let scorer = TieredMseScorer { sample_corpus };
    for sweep in 1..=repeats {
        for (label, interval) in [("baseline paired", 0u64), ("baseline remembered", 25u64)] {
            run_arm(
                sweep,
                label,
                interval,
                &creature_path,
                &training,
                dir.path(),
                &scorer,
                seconds,
                records,
                min_improvement,
            );
        }
    }
}

/// Run one arm and print its promote-phase economics.
#[allow(clippy::too_many_arguments)]
fn run_arm(
    sweep: usize,
    label: &str,
    interval: u64,
    creature_path: &Path,
    training: &Path,
    work: &Path,
    scorer: &TieredMseScorer,
    seconds: u64,
    records: usize,
    min_improvement: f64,
) {
    let out = work.join(format!("out-{sweep}-{interval}"));
    let config = LamarckConfig {
        creature: creature_path.to_path_buf(),
        training_data: training.to_path_buf(),
        timeout: Duration::from_secs(seconds),
        candidates: 8,
        min_improvement,
        seed: Some(7),
        output_dir: out,
        stats_mode: StatsMode::Quick,
        quick_sample_records: records as u64,
        focus_policy: FocusPolicy::Weighted,
        phase0_parity: false,
        screen_sample_rate: Some(0.05),
        screen_promote_threshold: 0.0,
        baseline_reverify_interval: interval,
        ..LamarckConfig::default()
    };

    let start = Instant::now();
    let result = run_optimisation(&config, scorer).expect("run completes");
    let wall = start.elapsed().as_secs_f64();
    let report = report_from_journal(&result.journal_path).expect("journal reports");
    let promote = report.scorer_call_cost.by_phase.get("promote");
    let (calls, promote_ms, creatures) = promote.map_or((0, 0.0, 0.0), |fit| {
        (
            fit.calls,
            fit.mean_ms * fit.calls as f64,
            fit.mean_creatures * fit.calls as f64,
        )
    });
    let reuse = &report.baseline_reuse;
    let per_call = promote_ms / calls.max(1) as f64;

    println!(
        "sweep {sweep} {label:>20}: {exp} experiments  promote {calls} call(s)  \
         {per_creature:.2} creature-score(s)/call  {per_call:.1}ms/call  \
         {per_creature_ms:.2}ms/creature-score  remembered {remembered}/{fresh}  \
         net saved {saved}  ({wall:.1}s wall)",
        exp = result.experiments,
        per_creature = creatures / calls.max(1) as f64,
        per_creature_ms = promote_ms / creatures.max(1.0),
        remembered = reuse.remembered_promote_calls,
        fresh = reuse.fresh_promote_calls,
        saved = reuse.net_creature_scores_saved,
    );
}
