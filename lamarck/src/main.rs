//! NEAT-AI-Lamarck CLI entry point.

use clap::{Parser, Subcommand};
use neat_ai_lamarck::{
    DEFAULT_CANDIDATE_COUNT, DEFAULT_MIN_IMPROVEMENT, DEFAULT_TIMEOUT_SECONDS, ExternalScorer,
    LamarckConfig, report_from_journal, run_optimisation,
};
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

#[derive(Debug, Parser)]
#[command(name = "neat_ai_lamarck")]
#[command(about = "Experimental optimiser for already-fit NEAT-AI creatures")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Current fittest creature JSON.
    #[arg(global = true)]
    creature: Option<PathBuf>,

    /// NEAT-AI training-data directory.
    #[arg(global = true)]
    training_data: Option<PathBuf>,

    /// Wall-clock budget in seconds.
    #[arg(long, default_value_t = DEFAULT_TIMEOUT_SECONDS)]
    timeout_seconds: u64,

    /// Candidate creatures generated per experiment.
    #[arg(long, default_value_t = DEFAULT_CANDIDATE_COUNT)]
    candidates: usize,

    /// Minimum absolute score improvement (strict `>`).
    #[arg(long, default_value_t = DEFAULT_MIN_IMPROVEMENT)]
    min_improvement: f64,

    /// Optional deterministic random seed.
    #[arg(long)]
    seed: Option<u64>,

    /// Path to the `rust_scorer` binary.
    #[arg(long, default_value = "rust_scorer")]
    scorer: PathBuf,

    /// Output directory for best.json / experiments.jsonl.
    #[arg(long, default_value = ".")]
    output_dir: PathBuf,

    /// Preserve rejected candidate directories.
    #[arg(long, default_value_t = false)]
    preserve_losers: bool,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Summarise strategy economics from an experiments.jsonl journal.
    Report {
        /// Path to experiments.jsonl.
        journal: PathBuf,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    if let Some(Commands::Report { journal }) = cli.command {
        match report_from_journal(&journal) {
            Ok(report) => match serde_json::to_string_pretty(&report) {
                Ok(text) => {
                    println!("{text}");
                    return ExitCode::SUCCESS;
                }
                Err(e) => {
                    eprintln!("failed to serialise report: {e}");
                    return ExitCode::FAILURE;
                }
            },
            Err(e) => {
                eprintln!("report failed: {e}");
                return ExitCode::FAILURE;
            }
        }
    }

    let (Some(creature), Some(training_data)) = (cli.creature, cli.training_data) else {
        eprintln!("creature and training_data positional arguments are required for optimisation");
        eprintln!("see --help");
        return ExitCode::FAILURE;
    };

    let config = LamarckConfig {
        creature,
        training_data,
        timeout: Duration::from_secs(cli.timeout_seconds),
        candidates: cli.candidates,
        min_improvement: cli.min_improvement,
        seed: cli.seed,
        scorer_path: cli.scorer.clone(),
        output_dir: cli.output_dir,
        preserve_losers: cli.preserve_losers,
    };

    let scorer = ExternalScorer { binary: cli.scorer };

    match run_optimisation(&config, &scorer) {
        Ok(result) => {
            println!(
                "Lamarck finished: experiments={} acceptances={} best_score={} best={} journal={}",
                result.experiments,
                result.acceptances,
                result.best_score,
                result.best_path.display(),
                result.journal_path.display()
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("Lamarck failed: {e}");
            ExitCode::FAILURE
        }
    }
}
