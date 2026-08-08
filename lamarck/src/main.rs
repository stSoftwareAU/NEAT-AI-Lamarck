use clap::Parser;
use neat_ai_lamarck::{DEFAULT_CANDIDATE_COUNT, DEFAULT_TIMEOUT_SECONDS};
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(name = "neat_ai_lamarck")]
#[command(about = "Experimental optimiser for already-fit NEAT-AI creatures")]
struct Cli {
    /// Current fittest creature JSON.
    creature: PathBuf,

    /// NEAT-AI training-data directory.
    training_data: PathBuf,

    /// Wall-clock budget in seconds.
    #[arg(long, default_value_t = DEFAULT_TIMEOUT_SECONDS)]
    timeout_seconds: u64,

    /// Candidate creatures generated per experiment.
    #[arg(long, default_value_t = DEFAULT_CANDIDATE_COUNT)]
    candidates: usize,

    /// Optional deterministic random seed.
    #[arg(long)]
    seed: Option<u64>,
}

fn main() {
    let cli = Cli::parse();
    println!(
        "Lamarck scaffold: creature={} training_data={} timeout={}s candidates={} seed={:?}",
        cli.creature.display(),
        cli.training_data.display(),
        cli.timeout_seconds,
        cli.candidates,
        cli.seed
    );
}
