//! NEAT-AI-Lamarck CLI entry point.

use clap::{Parser, Subcommand};
use neat_ai_lamarck::focus::FocusPolicy;
use neat_ai_lamarck::observations::{DEFAULT_QUICK_SAMPLE_RECORDS, StatsMode};
use neat_ai_lamarck::{
    CancelToken, DEFAULT_ANALYSIS_MEMO_ENTRIES, DEFAULT_ANALYSIS_THREADS,
    DEFAULT_CACHE_MAX_RESIDENT_BYTES, DEFAULT_CACHE_STAND_DOWN_MARGIN_MS,
    DEFAULT_CACHE_STAND_DOWN_WINDOW, DEFAULT_CANDIDATE_COUNT, DEFAULT_FAILED_CACHE_MAX_AGE_SECONDS,
    DEFAULT_FAILED_CACHE_MAX_ENTRIES, DEFAULT_FAILED_CACHE_TOLERANCE_ABS,
    DEFAULT_FAILED_CACHE_TOLERANCE_REL, DEFAULT_FOCUS_COUNT, DEFAULT_MIN_IMPROVEMENT,
    DEFAULT_SCREEN_PROMOTE_SIGMA_K, DEFAULT_SCREEN_PROMOTE_THRESHOLD, DEFAULT_SCREEN_SAMPLE_RATE,
    DEFAULT_TIMEOUT_SECONDS, ExternalScorer, LamarckConfig, PromoteGateMode, print_run_summary,
    report_from_journal, run_optimisation_cancellable,
};
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

#[derive(Debug, Parser)]
#[command(name = "neat_ai_lamarck")]
#[command(about = "Experimental optimiser for already-fit NEAT-AI creatures")]
#[command(version)]
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

    /// Stop after this many experiments (in addition to the wall-clock budget).
    #[arg(long)]
    max_experiments: Option<u64>,

    /// Candidate creatures generated per experiment.
    #[arg(long, default_value_t = DEFAULT_CANDIDATE_COUNT)]
    candidates: usize,

    /// Scale the generator's per-phase quotas with `--candidates` (issue #108).
    ///
    /// On by default; accepted as a no-op for older scripts.
    #[arg(long, default_value_t = true)]
    scale_candidate_quotas: bool,

    /// Use the legacy fixed per-phase quotas instead of scaling with
    /// `--candidates`. Caps a batch at ~33 distinct candidates whatever the
    /// budget says; kept only for A/B benchmarking against pre-#108 runs.
    #[arg(
        long,
        default_value_t = false,
        conflicts_with = "scale_candidate_quotas"
    )]
    fixed_candidate_quotas: bool,

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

    /// Use sampled `observations-quick.statistics` instead of a full-corpus cache.
    ///
    /// Analysis (focus/learning) uses the sample; the authoritative scorer still
    /// evaluates the full training corpus.
    #[arg(long, default_value_t = false)]
    quick: bool,

    /// Max records for `--quick` observations / focus / learning sampling.
    #[arg(long, default_value_t = DEFAULT_QUICK_SAMPLE_RECORDS)]
    quick_sample_records: u64,

    /// Lock focus to this neuron UUID (overrides `--focus-policy`).
    #[arg(long)]
    focus_neuron: Option<String>,

    /// Focus selection policy: weighted (default) | high-error | random | unsaturated.
    ///
    /// Weighted draws ∝ error-influence mass (output residual L1, depth-decayed
    /// hidden blame). Prefer this over high-error, which always sticks on one
    /// neuron. Zero-signal neurons are never selected.
    #[arg(long, default_value = "weighted")]
    focus_policy: String,

    /// Focus neurons proposed against per experiment (issue #109). Must be >= 1.
    ///
    /// The creature-wide learning and output-residual passes run once per
    /// experiment whatever this is, so `K > 1` amortises them over `K` focuses
    /// and splits `--candidates` between them. `--focus-neuron` pins the focus
    /// and caps this at 1.
    #[arg(long, default_value_t = DEFAULT_FOCUS_COUNT)]
    focus_count: usize,

    /// Compute expensive input×input correlations in observations.
    #[arg(long, default_value_t = false)]
    compute_correlations: bool,

    /// Skip Phase-0 baseline scorer gate.
    #[arg(long, default_value_t = false)]
    skip_phase0: bool,

    /// Only propose synapse/neuron growth candidates (skip weight/bias strategies).
    #[arg(long, default_value_t = false)]
    structural_only: bool,

    /// Screen candidates on a scorer subsample before full-corpus promote scoring
    /// (issue #24). Values in `(0, 1)` enable screening; `1` (or `>=1`) disables
    /// and scores the full batch on the full corpus only.
    #[arg(long, default_value_t = DEFAULT_SCREEN_SAMPLE_RATE)]
    screen_sample_rate: f64,

    /// Minimum sample-score Δ to promote a candidate to full-corpus scoring.
    ///
    /// Stays in force under `--screen-promote-gate noise-aware` as that gate's
    /// absolute floor, so the noise-aware gate is never the weaker of the two.
    #[arg(long, default_value_t = DEFAULT_SCREEN_PROMOTE_THRESHOLD)]
    screen_promote_threshold: f64,

    /// Promote gate: absolute (default) | noise-aware (issue #111).
    ///
    /// `absolute` is the pre-#111 run: promote on a bare
    /// `--screen-promote-threshold`. `noise-aware` prices the batch's own
    /// screen-Δ spread first and promotes on
    /// `Δ > max(k · σ̂, --screen-promote-threshold)`. Opt-in until a paired
    /// benchmark on accepts per wall-clock hour justifies moving the default.
    #[arg(long, default_value = "absolute")]
    screen_promote_gate: String,

    /// σ̂ multiplier `k` for `--screen-promote-gate noise-aware`. Must be > 0.
    ///
    /// Ignored under the absolute gate. `docs/screen-calibration.md` measured
    /// the screen's noise floor and recommended 3σ.
    #[arg(long, default_value_t = DEFAULT_SCREEN_PROMOTE_SIGMA_K)]
    screen_promote_sigma_k: f64,

    /// Promote calls served from the remembered full-corpus baseline before one
    /// scores it fresh again (issue #113). `0` (default) disables reuse.
    ///
    /// Between accepts the incumbent's full-corpus score is a constant the run
    /// already knows, so a promote call that reuses it drops ≈20% of its
    /// creature-scores. That also drops the pairing that makes a promote call
    /// self-verifying, so every Nth call re-scores the incumbent and checks it
    /// against `--baseline-drift-epsilon`, and any accept is re-decided against
    /// a freshly scored pair before the incumbent is swapped.
    #[arg(long, default_value_t = 0)]
    baseline_reverify_interval: u64,

    /// Absolute baseline-score drift that aborts the run (issue #113).
    ///
    /// Omit to auto-tune from corpus size and Phase-0 error (directory-scorer
    /// `f32` association model). Pass an absolute value only for expert / A/B
    /// overrides — hosts should not ship a competing default. Accept safety is
    /// the paired re-score, not this epsilon.
    #[arg(long)]
    baseline_drift_epsilon: Option<f64>,

    /// Local JSON store for structural graft memory (phase-G replay).
    ///
    /// When set, Lamarck replays missing applicable grafts onto the opening
    /// fittest before the experiment loop, and records new structural accepts.
    /// Keep this outside any wiped work directory (e.g. GRQ `.lamarck-sampler`).
    #[arg(long)]
    grafts_path: Option<PathBuf>,

    /// Wall-clock seconds for phase-G graft replay (default: 10% of --timeout-seconds).
    #[arg(long)]
    graft_replay_budget_seconds: Option<u64>,

    /// Backprop learning rate used when proposing `backprop` candidates
    /// (default: 0.01, the NEAT-AI port value). Must be > 0.
    #[arg(long)]
    backprop_learning_rate: Option<f64>,

    /// Focus-dependent entries the cross-experiment analysis memo may hold
    /// (issue #106).
    ///
    /// While the incumbent is unchanged, the focus stats / incoming-source /
    /// residual-ranking scan is a pure function of `(incumbent, focus, sample)`,
    /// so a repeated focus is served from the memo and skips a whole training
    /// scan. `0` disables memoisation. Every entry is dropped whenever the
    /// incumbent changes.
    #[arg(long, default_value_t = DEFAULT_ANALYSIS_MEMO_ENTRIES)]
    analysis_memo_entries: usize,

    /// Worker threads folding record chunks in the two per-experiment analysis
    /// scans (issue #107). Must be >= 1.
    ///
    /// The sample is cut into fixed-size record chunks and the per-chunk
    /// partials are merged in chunk order, so the analysis is bit-identical at
    /// every thread count — only the wall clock moves. Deliberately not "all
    /// cores": the scorer owns the box whenever it runs.
    #[arg(long, default_value_t = DEFAULT_ANALYSIS_THREADS)]
    analysis_threads: usize,

    /// Cap on one `backprop` bias step, overriding
    /// `BackpropConfig::maximum_bias_adjustment_scale` (default: 10). Must be > 0.
    ///
    /// On a focus carrying a saturating blame mass the proposal hits this cap
    /// at every learning rate, so this — not `--backprop-learning-rate` — is
    /// the knob that resizes the step (issue #96).
    #[arg(long)]
    backprop_max_bias_adjustment_scale: Option<f64>,

    /// Skip candidates a previous experiment or run already scored as failures
    /// (issue #69), backfilling the batch so the scorer still runs at full
    /// width. Off by default until the feature proves it saves more scorer time
    /// than it costs.
    #[arg(long, default_value_t = false)]
    failed_cache: bool,

    /// Size cap on the failed-candidate cache. `0` disables the cache.
    #[arg(long, default_value_t = DEFAULT_FAILED_CACHE_MAX_ENTRIES)]
    failed_cache_max_entries: usize,

    /// Drop failed-candidate entries older than this many seconds. `0` keeps
    /// entries until the size cap evicts them.
    #[arg(long, default_value_t = DEFAULT_FAILED_CACHE_MAX_AGE_SECONDS)]
    failed_cache_max_age_seconds: u64,

    /// Absolute bound for treating two candidate values as the same proposal.
    #[arg(long, default_value_t = DEFAULT_FAILED_CACHE_TOLERANCE_ABS)]
    failed_cache_tolerance_abs: f64,

    /// Relative bound for treating two candidate values as the same proposal.
    #[arg(long, default_value_t = DEFAULT_FAILED_CACHE_TOLERANCE_REL)]
    failed_cache_tolerance_rel: f64,

    /// Milliseconds the failed-candidate cache's cumulative overhead must
    /// exceed its cumulative estimated savings by before an experiment counts
    /// as net-negative (issue #92).
    #[arg(long, default_value_t = DEFAULT_CACHE_STAND_DOWN_MARGIN_MS)]
    failed_cache_stand_down_margin_ms: f64,

    /// Consecutive net-negative experiments before the cache stands down for
    /// the rest of the run. `0` disables the guardrail.
    #[arg(long, default_value_t = DEFAULT_CACHE_STAND_DOWN_WINDOW)]
    failed_cache_stand_down_window: usize,

    /// Resident-footprint ceiling in bytes, enforced by evicting oldest-first.
    /// `0` disables the ceiling; the entry cap still bounds the cache.
    #[arg(long, default_value_t = DEFAULT_CACHE_MAX_RESIDENT_BYTES)]
    failed_cache_max_bytes: usize,
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

    let focus_policy = FocusPolicy::parse(&cli.focus_policy).unwrap_or_else(|| {
        eprintln!(
            "unknown --focus-policy '{}'; expected weighted|high-error|random|unsaturated",
            cli.focus_policy
        );
        std::process::exit(2);
    });

    let screen_promote_gate =
        PromoteGateMode::parse(&cli.screen_promote_gate).unwrap_or_else(|| {
            eprintln!(
                "unknown --screen-promote-gate '{}'; expected absolute|noise-aware",
                cli.screen_promote_gate
            );
            std::process::exit(2);
        });

    let config = LamarckConfig {
        creature,
        training_data,
        timeout: Duration::from_secs(cli.timeout_seconds),
        max_experiments: cli.max_experiments,
        candidates: cli.candidates,
        scale_candidate_quotas: cli.scale_candidate_quotas && !cli.fixed_candidate_quotas,
        min_improvement: cli.min_improvement,
        seed: cli.seed,
        scorer_path: cli.scorer.clone(),
        output_dir: cli.output_dir,
        preserve_losers: cli.preserve_losers,
        stats_mode: if cli.quick {
            StatsMode::Quick
        } else {
            StatsMode::Full
        },
        quick_sample_records: cli.quick_sample_records,
        focus_neuron: cli.focus_neuron,
        focus_policy,
        focus_count: cli.focus_count,
        compute_correlations: cli.compute_correlations,
        max_consecutive_scorer_failures: neat_ai_lamarck::DEFAULT_MAX_CONSECUTIVE_SCORER_FAILURES,
        phase0_parity: !cli.skip_phase0,
        structural_only: cli.structural_only,
        screen_sample_rate: if cli.screen_sample_rate > 0.0 && cli.screen_sample_rate < 1.0 {
            Some(cli.screen_sample_rate)
        } else {
            None
        },
        screen_promote_threshold: cli.screen_promote_threshold,
        screen_promote_gate,
        screen_promote_sigma_k: cli.screen_promote_sigma_k,
        baseline_reverify_interval: cli.baseline_reverify_interval,
        baseline_drift_epsilon: cli.baseline_drift_epsilon,
        grafts_path: cli.grafts_path,
        graft_replay_budget: cli.graft_replay_budget_seconds.map(Duration::from_secs),
        backprop_learning_rate: cli.backprop_learning_rate,
        analysis_memo_entries: cli.analysis_memo_entries,
        backprop_max_bias_adjustment_scale: cli.backprop_max_bias_adjustment_scale,
        analysis_threads: cli.analysis_threads,
        failed_cache: cli.failed_cache,
        failed_cache_max_entries: cli.failed_cache_max_entries,
        failed_cache_max_age_seconds: cli.failed_cache_max_age_seconds,
        failed_cache_tolerance_abs: cli.failed_cache_tolerance_abs,
        failed_cache_tolerance_rel: cli.failed_cache_tolerance_rel,
        failed_cache_stand_down_margin_ms: cli.failed_cache_stand_down_margin_ms,
        failed_cache_stand_down_window: cli.failed_cache_stand_down_window,
        failed_cache_max_bytes: cli.failed_cache_max_bytes,
    };

    // Fail before spawning the scorer rather than deep inside the run.
    if let Err(e) = config.backprop_config() {
        eprintln!("{e}");
        return ExitCode::FAILURE;
    }
    if let Err(e) = config.analysis_threads() {
        eprintln!("{e}");
        return ExitCode::FAILURE;
    }
    if let Err(e) = config.focus_count() {
        eprintln!("{e}");
        return ExitCode::FAILURE;
    }
    if let Err(e) = config.promote_gate() {
        eprintln!("{e}");
        return ExitCode::FAILURE;
    }
    // Validate an explicit override early; auto-tune is resolved once the
    // corpus size is known inside the optimisation loop.
    if let Err(e) = config.baseline_reuse_policy(2, 1.0) {
        eprintln!("{e}");
        return ExitCode::FAILURE;
    }

    let scorer = ExternalScorer { binary: cli.scorer };

    // SIGINT/SIGTERM must stop the run gracefully (issue #72): the loop polls
    // the token, so the final best.json stamp and run summary still happen. A
    // handler that cannot be installed is fatal — cancellation would otherwise
    // look wired up while silently killing the run mid-experiment.
    let cancel = CancelToken::new();
    #[cfg(unix)]
    if let Err(e) = neat_ai_lamarck::cancel::install_cancel_signals(&cancel) {
        eprintln!("failed to install cancellation signal handlers: {e}");
        return ExitCode::FAILURE;
    }

    match run_optimisation_cancellable(&config, &scorer, &cancel) {
        Ok(result) => {
            print_run_summary(&result);
            if result.experiments > 0 && result.scorer_successes == 0 {
                ExitCode::FAILURE
            } else {
                ExitCode::SUCCESS
            }
        }
        Err(e) => {
            eprintln!("Lamarck failed: {e}");
            ExitCode::FAILURE
        }
    }
}
