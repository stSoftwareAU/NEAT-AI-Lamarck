//! Benchmark / strategy economics reporting from `experiments.jsonl`.

use crate::candidates::CandidateStrategy;
use crate::run::ExperimentRecord;
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
    /// Time to first acceptance (ms of wall timestamps unavailable — use scorer+analysis sums).
    pub time_to_first_acceptance_ms: Option<u128>,
    /// Total analysis milliseconds.
    pub total_analysis_ms: u128,
    /// Total scorer milliseconds.
    pub total_scorer_ms: u128,
    /// Absolute score improvement from first baseline to last accepted.
    pub total_score_improvement: Option<f64>,
    /// Per-strategy win counts.
    pub strategies: Vec<StrategyStats>,
}

/// Consume `experiments.jsonl` and produce an economics report.
pub fn report_from_journal(path: &Path) -> Result<JournalReport, String> {
    let file = File::open(path).map_err(|e| e.to_string())?;
    let reader = BufReader::new(file);
    let mut experiments = 0u64;
    let mut acceptances = 0u64;
    let mut total_analysis_ms = 0u128;
    let mut total_scorer_ms = 0u128;
    let mut time_to_first = None;
    let mut elapsed = 0u128;
    let mut first_baseline = None;
    let mut last_best = None;
    let mut wins: BTreeMap<String, u64> = BTreeMap::new();

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

    Ok(JournalReport {
        experiments,
        acceptances,
        time_to_first_acceptance_ms: time_to_first,
        total_analysis_ms,
        total_scorer_ms,
        total_score_improvement,
        strategies,
    })
}

fn strategy_name(strategy: CandidateStrategy) -> String {
    match strategy {
        CandidateStrategy::Backprop => "backprop".into(),
        CandidateStrategy::StatsWeight => "stats_weight".into(),
        CandidateStrategy::StatsBias => "stats_bias".into(),
        CandidateStrategy::Random => "random".into(),
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
            winner: Some("candidate-000".into()),
            improvement: Some(2e-6),
            accepted: true,
            analysis_ms: 10,
            scorer_ms: 20,
        };
        writeln!(file, "{}", serde_json::to_string(&record).unwrap()).unwrap();
        let report = report_from_journal(file.path()).unwrap();
        assert_eq!(report.experiments, 1);
        assert_eq!(report.acceptances, 1);
        assert_eq!(report.strategies[0].strategy, "random");
    }
}
