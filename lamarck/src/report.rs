//! Benchmark / strategy economics reporting from `experiments.jsonl`.

use crate::candidates::CandidateStrategy;
use crate::log;
use crate::run::{ExperimentRecord, RunResult};
use serde::Serialize;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Aggregated strategy hit-rate row.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StrategyStats {
    /// Strategy name.
    pub strategy: String,
    /// Times this strategy appeared on an accepted winner provenance.
    pub wins: u64,
    /// Times this strategy appeared among candidates in accepted experiments.
    pub appearances_in_accepted: u64,
}

/// Summary report for a Lamarck run journal.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct JournalReport {
    /// Experiments attempted.
    pub experiments: u64,
    /// Accepted improvements.
    pub acceptances: u64,
    /// Scorer batch failures recorded in the journal.
    pub scorer_failures: u64,
    /// First experiment baseline score (opening incumbent).
    pub opening_baseline_score: Option<f64>,
    /// Time to first acceptance (ms of wall timestamps unavailable — use scorer+analysis sums).
    pub time_to_first_acceptance_ms: Option<u128>,
    /// Total analysis milliseconds.
    pub total_analysis_ms: u128,
    /// Total scorer milliseconds.
    pub total_scorer_ms: u128,
    /// Candidates scored (sum of scored stems excluding failures).
    pub candidates_scored: u64,
    /// Candidates scored per minute of scorer wall time.
    pub candidates_per_scorer_minute: f64,
    /// Analysis share of (analysis+scorer) time.
    pub analysis_time_fraction: f64,
    /// Absolute score improvement from first baseline to last accepted.
    pub total_score_improvement: Option<f64>,
    /// Focus neurons seen (uuid → experiment count).
    pub focus_counts: BTreeMap<String, u64>,
    /// Per-strategy win counts.
    pub strategies: Vec<StrategyStats>,
}

/// Consume `experiments.jsonl` and produce an economics report.
pub fn report_from_journal(path: &Path) -> Result<JournalReport, String> {
    let file = File::open(path).map_err(|e| e.to_string())?;
    let reader = BufReader::new(file);
    let mut experiments = 0u64;
    let mut acceptances = 0u64;
    let mut scorer_failures = 0u64;
    let mut total_analysis_ms = 0u128;
    let mut total_scorer_ms = 0u128;
    let mut candidates_scored = 0u64;
    let mut time_to_first = None;
    let mut elapsed = 0u128;
    let mut first_baseline = None;
    let mut last_best = None;
    let mut wins: BTreeMap<String, u64> = BTreeMap::new();
    let mut focus_counts: BTreeMap<String, u64> = BTreeMap::new();

    for line in reader.lines() {
        let line = line.map_err(|e| e.to_string())?;
        if line.trim().is_empty() {
            continue;
        }
        let record: ExperimentRecord = serde_json::from_str(&line).map_err(|e| e.to_string())?;
        experiments += 1;
        total_analysis_ms += record.analysis_ms;
        total_scorer_ms += record.scorer_ms;
        elapsed += record.analysis_ms + record.scorer_ms;
        *focus_counts.entry(record.focus_neuron.clone()).or_default() += 1;
        if record.scorer_error.is_some() {
            scorer_failures += 1;
        } else {
            candidates_scored += record.scores.len().saturating_sub(1) as u64;
        }
        if first_baseline.is_none() {
            first_baseline = Some(record.baseline_score);
        }
        if record.accepted {
            acceptances += 1;
            if time_to_first.is_none() {
                time_to_first = Some(elapsed);
            }
            if let Some(delta) = record.improvement {
                let base = last_best.unwrap_or(record.baseline_score);
                last_best = Some(base + delta);
            }
            if let Some(winner) = &record.winner
                && let Some(idx) = winner
                    .strip_prefix("candidate-")
                    .and_then(|s| s.parse::<usize>().ok())
                && let Some(prov) = record.candidates.get(idx)
            {
                let key = strategy_name(prov.strategy);
                *wins.entry(key).or_default() += 1;
            }
        }
    }

    let strategies = wins
        .into_iter()
        .map(|(strategy, wins)| StrategyStats {
            strategy,
            wins,
            appearances_in_accepted: wins,
        })
        .collect();

    let total_score_improvement = match (first_baseline, last_best) {
        (Some(first), Some(last)) => Some(last - first),
        _ => None,
    };
    let total_ms = (total_analysis_ms + total_scorer_ms) as f64;
    let analysis_time_fraction = if total_ms > 0.0 {
        total_analysis_ms as f64 / total_ms
    } else {
        0.0
    };
    let candidates_per_scorer_minute = if total_scorer_ms > 0 {
        candidates_scored as f64 / (total_scorer_ms as f64 / 60_000.0)
    } else {
        0.0
    };

    Ok(JournalReport {
        experiments,
        acceptances,
        scorer_failures,
        opening_baseline_score: first_baseline,
        time_to_first_acceptance_ms: time_to_first,
        total_analysis_ms,
        total_scorer_ms,
        candidates_scored,
        candidates_per_scorer_minute,
        analysis_time_fraction,
        total_score_improvement,
        focus_counts,
        strategies,
    })
}

fn strategy_name(strategy: CandidateStrategy) -> String {
    match strategy {
        CandidateStrategy::Backprop => "backprop".into(),
        CandidateStrategy::MeanErrorBias => "mean_error_bias".into(),
        CandidateStrategy::StatsWeight => "stats_weight".into(),
        CandidateStrategy::StatsBias => "stats_bias".into(),
        CandidateStrategy::StructuralAdd => "structural_add".into(),
        CandidateStrategy::StructuralAddNeuron => "structural_add_neuron".into(),
        CandidateStrategy::StructuralWeaken => "structural_weaken".into(),
        CandidateStrategy::Random => "random".into(),
    }
}

fn format_ms(ms: u128) -> String {
    if ms >= 60_000 {
        let secs = ms as f64 / 1000.0;
        format!("{secs:.1}s")
    } else if ms >= 1000 {
        format!("{:.1}s", ms as f64 / 1000.0)
    } else {
        format!("{ms}ms")
    }
}

/// Print a coloured human summary of a completed optimisation run to stderr.
pub fn print_run_summary(result: &RunResult) {
    log::info("run summary");
    log::detail(&format!(
        "experiments:  {}  (accepted {}  scorer_ok {}  scorer_fail {})",
        result.experiments, result.acceptances, result.scorer_successes, result.scorer_failures
    ));

    if let Ok(report) = report_from_journal(&result.journal_path) {
        let open = result
            .opening_baseline_score
            .or(report.opening_baseline_score);
        if let Some(open) = open {
            let delta = result.best_score - open;
            if result.acceptances == 0 {
                log::detail(&format!(
                    "best score:    {}  (no improvement over opening {})",
                    result.best_score, open
                ));
            } else {
                log::detail(&format!(
                    "best score:    {}  (opening {}  Δ {delta:+.6e})",
                    result.best_score, open
                ));
            }
        } else {
            log::detail(&format!("best score:    {}", result.best_score));
        }

        let total_ms = report.total_analysis_ms + report.total_scorer_ms;
        log::detail(&format!(
            "time:          analysis {}  + scorer {}  = {}  (analysis {:.0}%)",
            format_ms(report.total_analysis_ms),
            format_ms(report.total_scorer_ms),
            format_ms(total_ms),
            report.analysis_time_fraction * 100.0
        ));
        log::detail(&format!(
            "throughput:    {:.1} candidates/scorer-minute ({} scored)",
            report.candidates_per_scorer_minute, report.candidates_scored
        ));
        if let Some(ms) = report.time_to_first_acceptance_ms {
            log::detail(&format!("first accept:  {}", format_ms(ms)));
        }
        if !report.focus_counts.is_empty() {
            let focuses = report
                .focus_counts
                .iter()
                .map(|(k, v)| format!("{k}×{v}"))
                .collect::<Vec<_>>()
                .join(", ");
            log::detail(&format!("focus hits:    {focuses}"));
        }
        if !report.strategies.is_empty() {
            let wins = report
                .strategies
                .iter()
                .map(|s| format!("{}×{}", s.strategy, s.wins))
                .collect::<Vec<_>>()
                .join(", ");
            log::detail(&format!("winning strats: {wins}"));
        }
    } else {
        log::detail(&format!("best score:    {}", result.best_score));
    }

    log::detail(&format!("best.json:     {}", result.best_path.display()));
    log::detail(&format!("journal:       {}", result.journal_path.display()));
    if result.scorer_failures > 0 {
        log::warn(&format!(
            "scorer failures during run: {}",
            result.scorer_failures
        ));
    }
    if result.acceptances > 0 {
        log::ok(&format!(
            "finished with {} acceptance(s)",
            result.acceptances
        ));
    } else {
        log::warn("finished with no acceptances");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidates::CandidateProvenance;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn report_counts_acceptances() {
        let mut file = NamedTempFile::new().unwrap();
        let record = ExperimentRecord {
            experiment_number: 1,
            timestamp_unix: 1,
            seed: Some(1),
            incumbent_id: "x".into(),
            baseline_score: 0.4,
            focus_neuron: "h1".into(),
            candidates: vec![CandidateProvenance {
                strategy: CandidateStrategy::Random,
                focus_neuron: "h1".into(),
                mutation: "x".into(),
                old_value: Some(0.0),
                new_value: Some(0.1),
            }],
            scores: Default::default(),
            screen_scores: None,
            winner: Some("candidate-000".into()),
            improvement: Some(2e-6),
            accepted: true,
            analysis_ms: 10,
            scorer_ms: 20,
            scorer_error: None,
        };
        writeln!(file, "{}", serde_json::to_string(&record).unwrap()).unwrap();
        let report = report_from_journal(file.path()).unwrap();
        assert_eq!(report.experiments, 1);
        assert_eq!(report.acceptances, 1);
        assert_eq!(report.strategies[0].strategy, "random");
    }
}
