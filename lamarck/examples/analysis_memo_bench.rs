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

use std::time::{Duration, Instant};

use neat_ai_lamarck::focus::FocusPolicy;
use neat_ai_lamarck::observations::StatsMode;
use neat_ai_lamarck::{LamarckConfig, report_from_journal, run_optimisation};

mod support;
use support::{LocalMseScorer, creature_json, write_sample};

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
