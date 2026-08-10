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
    /// Times this strategy appeared among candidates across all experiments.
    pub appearances_total: u64,
    /// Times this strategy appeared among candidates in accepted experiments.
    pub appearances_in_accepted: u64,
    /// `wins / appearances_total` (0 when no appearances).
    pub acceptance_rate: f64,
}

/// Per-focus hit / failure history.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FocusHistory {
    /// Focus neuron UUID.
    pub focus_neuron: String,
    /// Experiments that selected this focus.
    pub experiments: u64,
    /// Accepted improvements while focused here.
    pub accepts: u64,
    /// Cumulative accepted score Δ while focused here.
    pub cumulative_improvement: f64,
}

/// One point on the improvement-vs-fitness series.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ImprovementPoint {
    /// Experiment number.
    pub experiment_number: u64,
    /// Incumbent baseline score at experiment start.
    pub baseline_score: f64,
    /// Absolute improvement when accepted.
    pub improvement: Option<f64>,
    /// Whether a candidate was accepted.
    pub accepted: bool,
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
    /// Time to first acceptance (sum of analysis+scorer ms until first accept).
    pub time_to_first_acceptance_ms: Option<u128>,
    /// Total analysis milliseconds.
    pub total_analysis_ms: u128,
    /// Total scorer milliseconds.
    pub total_scorer_ms: u128,
    /// Wall duration from first→last `timestampUnix` when available (ms).
    pub wall_duration_ms: Option<u128>,
    /// Promote-phase candidates scored (excludes baseline stems; excludes failures).
    pub candidates_scored: u64,
    /// Screen-phase candidate stems counted when `screenScores` present.
    pub screen_candidates_scored: u64,
    /// Promote candidates scored per minute of scorer wall time.
    pub candidates_per_scorer_minute: f64,
    /// Screen + promote candidate stems per minute of scorer wall time.
    pub candidates_per_screen_minute: f64,
    /// Promote candidates per wall-clock minute (when timestamps span > 0).
    pub candidates_per_wall_minute: Option<f64>,
    /// Analysis share of (analysis+scorer) time.
    pub analysis_time_fraction: f64,
    /// Absolute score improvement from first baseline to last accepted.
    pub total_score_improvement: Option<f64>,
    /// Relative improvement `total / opening` when opening > 0.
    pub relative_score_improvement: Option<f64>,
    /// Mean experiment duration (analysis+scorer ms).
    pub mean_experiment_ms: f64,
    /// Projected batches completable in a 45-minute budget from mean duration.
    pub projected_batches_per_45_min: f64,
    /// Focus neurons seen (uuid → experiment count).
    pub focus_counts: BTreeMap<String, u64>,
    /// Focus hit/failure history.
    pub focus_history: Vec<FocusHistory>,
    /// Improvement-vs-fitness series (one point per experiment).
    pub improvement_series: Vec<ImprovementPoint>,
    /// Per-strategy win / appearance / acceptance-rate rows.
    pub strategies: Vec<StrategyStats>,
    /// Sum of `combosScored` across experiments (combo batch size for #63 tuning).
    pub combos_scored_total: u64,
    /// Sum of `combosDampened` across experiments.
    pub combos_dampened_total: u64,
    /// Acceptances whose winner was a multi-member combo.
    pub combo_acceptances: u64,
    /// Combo acceptances that also recorded a non-empty `comboDampen`.
    pub combo_acceptances_with_dampen: u64,
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
    let mut screen_candidates_scored = 0u64;
    let mut time_to_first = None;
    let mut elapsed = 0u128;
    let mut first_baseline = None;
    let mut last_best = None;
    let mut first_ts: Option<u64> = None;
    let mut last_ts: Option<u64> = None;
    let mut wins: BTreeMap<String, u64> = BTreeMap::new();
    let mut appearances_total: BTreeMap<String, u64> = BTreeMap::new();
    let mut appearances_in_accepted: BTreeMap<String, u64> = BTreeMap::new();
    let mut focus_counts: BTreeMap<String, u64> = BTreeMap::new();
    let mut focus_accepts: BTreeMap<String, u64> = BTreeMap::new();
    let mut focus_delta: BTreeMap<String, f64> = BTreeMap::new();
    let mut improvement_series = Vec::new();
    let mut combos_scored_total = 0u64;
    let mut combos_dampened_total = 0u64;
    let mut combo_acceptances = 0u64;
    let mut combo_acceptances_with_dampen = 0u64;

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
        first_ts = Some(first_ts.map_or(record.timestamp_unix, |t| t.min(record.timestamp_unix)));
        last_ts = Some(last_ts.map_or(record.timestamp_unix, |t| t.max(record.timestamp_unix)));
        *focus_counts.entry(record.focus_neuron.clone()).or_default() += 1;

        for prov in &record.candidates {
            let key = strategy_name(prov.strategy);
            *appearances_total.entry(key).or_default() += 1;
        }

        if record.scorer_error.is_some() {
            scorer_failures += 1;
        } else {
            candidates_scored += record.scores.len().saturating_sub(1) as u64;
            if let Some(screen) = &record.screen_scores {
                screen_candidates_scored += screen.len().saturating_sub(1) as u64;
            }
        }
        if first_baseline.is_none() {
            first_baseline = Some(record.baseline_score);
        }

        improvement_series.push(ImprovementPoint {
            experiment_number: record.experiment_number,
            baseline_score: record.baseline_score,
            improvement: record.improvement,
            accepted: record.accepted,
        });

        if let Some(n) = record.combos_scored {
            combos_scored_total += n as u64;
        }
        if let Some(n) = record.combos_dampened {
            combos_dampened_total += n as u64;
        }

        if record.accepted {
            acceptances += 1;
            if time_to_first.is_none() {
                time_to_first = Some(elapsed);
            }
            if record.combo_members.is_some_and(|m| m > 1) {
                combo_acceptances += 1;
                if record.combo_dampen.as_ref().is_some_and(|d| !d.is_empty()) {
                    combo_acceptances_with_dampen += 1;
                }
            }
            if let Some(delta) = record.improvement {
                let base = last_best.unwrap_or(record.baseline_score);
                last_best = Some(base + delta);
                *focus_accepts
                    .entry(record.focus_neuron.clone())
                    .or_default() += 1;
                *focus_delta.entry(record.focus_neuron.clone()).or_default() += delta;
            }
            for prov in &record.candidates {
                let key = strategy_name(prov.strategy);
                *appearances_in_accepted.entry(key).or_default() += 1;
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

    let mut strategies: Vec<StrategyStats> = appearances_total
        .iter()
        .map(|(strategy, &total)| {
            let w = *wins.get(strategy).unwrap_or(&0);
            let in_acc = *appearances_in_accepted.get(strategy).unwrap_or(&0);
            StrategyStats {
                strategy: strategy.clone(),
                wins: w,
                appearances_total: total,
                appearances_in_accepted: in_acc,
                acceptance_rate: if total > 0 {
                    w as f64 / total as f64
                } else {
                    0.0
                },
            }
        })
        .collect();
    strategies.sort_by(|a, b| {
        b.wins
            .cmp(&a.wins)
            .then_with(|| a.strategy.cmp(&b.strategy))
    });

    let focus_history: Vec<FocusHistory> = focus_counts
        .iter()
        .map(|(uuid, &exps)| FocusHistory {
            focus_neuron: uuid.clone(),
            experiments: exps,
            accepts: *focus_accepts.get(uuid).unwrap_or(&0),
            cumulative_improvement: *focus_delta.get(uuid).unwrap_or(&0.0),
        })
        .collect();

    let total_score_improvement = match (first_baseline, last_best) {
        (Some(first), Some(last)) => Some(last - first),
        _ => None,
    };
    let relative_score_improvement = match (first_baseline, total_score_improvement) {
        (Some(open), Some(delta)) if open.abs() > f64::EPSILON => Some(delta / open),
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
    let screen_total = screen_candidates_scored + candidates_scored;
    let candidates_per_screen_minute = if total_scorer_ms > 0 {
        screen_total as f64 / (total_scorer_ms as f64 / 60_000.0)
    } else {
        0.0
    };
    let wall_duration_ms = match (first_ts, last_ts) {
        (Some(a), Some(b)) if b > a => Some(((b - a) as u128).saturating_mul(1000)),
        _ => None,
    };
    let candidates_per_wall_minute = wall_duration_ms.and_then(|wall| {
        if wall > 0 {
            Some(candidates_scored as f64 / (wall as f64 / 60_000.0))
        } else {
            None
        }
    });
    let mean_experiment_ms = if experiments > 0 {
        total_ms / experiments as f64
    } else {
        0.0
    };
    let projected_batches_per_45_min = if mean_experiment_ms > 0.0 {
        (45.0 * 60_000.0) / mean_experiment_ms
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
        wall_duration_ms,
        candidates_scored,
        screen_candidates_scored,
        candidates_per_scorer_minute,
        candidates_per_screen_minute,
        candidates_per_wall_minute,
        analysis_time_fraction,
        total_score_improvement,
        relative_score_improvement,
        mean_experiment_ms,
        projected_batches_per_45_min,
        focus_counts,
        focus_history,
        improvement_series,
        strategies,
        combos_scored_total,
        combos_dampened_total,
        combo_acceptances,
        combo_acceptances_with_dampen,
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
            "throughput:    {:.1} promote/scorer-min ({} scored); ~{:.1} batches/45min",
            report.candidates_per_scorer_minute,
            report.candidates_scored,
            report.projected_batches_per_45_min
        ));
        if let Some(ms) = report.time_to_first_acceptance_ms {
            log::detail(&format!("first accept:  {}", format_ms(ms)));
        }
        if report.combos_scored_total > 0 || report.combo_acceptances > 0 {
            log::detail(&format!(
                "combos:        scored {}  dampened {}  accepts {}  (with dampen {})  exponent={}",
                report.combos_scored_total,
                report.combos_dampened_total,
                report.combo_acceptances,
                report.combo_acceptances_with_dampen,
                crate::combos::STACK_DAMPEN_EXPONENT
            ));
        }
        if !report.focus_history.is_empty() {
            let focuses = report
                .focus_history
                .iter()
                .map(|h| {
                    format!(
                        "{}×{} (accepts={} Δ={:+.3e})",
                        h.focus_neuron, h.experiments, h.accepts, h.cumulative_improvement
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");
            log::detail(&format!("focus history: {focuses}"));
        }
        if !report.strategies.is_empty() {
            let wins = report
                .strategies
                .iter()
                .filter(|s| s.wins > 0 || s.appearances_total > 0)
                .map(|s| {
                    format!(
                        "{}×{}/{} ({:.0}%)",
                        s.strategy,
                        s.wins,
                        s.appearances_total,
                        s.acceptance_rate * 100.0
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");
            log::detail(&format!("strategies:    {wins}"));
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
    use std::collections::BTreeMap;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn prov(strategy: CandidateStrategy) -> CandidateProvenance {
        CandidateProvenance {
            strategy,
            focus_neuron: "h1".into(),
            mutation: "x".into(),
            old_value: Some(0.0),
            new_value: Some(0.1),
        }
    }

    #[test]
    fn report_counts_acceptances_and_strategy_rates() {
        let mut file = NamedTempFile::new().unwrap();
        let accepted = ExperimentRecord {
            experiment_number: 1,
            timestamp_unix: 1000,
            seed: Some(1),
            incumbent_id: "x".into(),
            baseline_score: 0.4,
            focus_neuron: "h1".into(),
            candidates: vec![
                prov(CandidateStrategy::Random),
                prov(CandidateStrategy::Backprop),
            ],
            scores: {
                let mut m = BTreeMap::new();
                m.insert("baseline".into(), 0.4);
                m.insert("candidate-000".into(), 0.400002);
                m
            },
            screen_scores: Some({
                let mut m = BTreeMap::new();
                m.insert("baseline".into(), 0.4);
                m.insert("candidate-000".into(), 0.41);
                m.insert("candidate-001".into(), 0.39);
                m
            }),
            winner: Some("candidate-000".into()),
            improvement: Some(2e-6),
            accepted: true,
            analysis_ms: 10,
            scorer_ms: 20,
            scorer_error: None,
            combo_members: None,
            combos_scored: None,
            combos_dampened: None,
            combo_dampen: None,
        };
        let rejected = ExperimentRecord {
            experiment_number: 2,
            timestamp_unix: 1060,
            seed: Some(1),
            incumbent_id: "x".into(),
            baseline_score: 0.400002,
            focus_neuron: "h1".into(),
            candidates: vec![prov(CandidateStrategy::Random)],
            scores: {
                let mut m = BTreeMap::new();
                m.insert("baseline".into(), 0.400002);
                m.insert("candidate-000".into(), 0.400001);
                m
            },
            screen_scores: None,
            winner: None,
            improvement: None,
            accepted: false,
            analysis_ms: 5,
            scorer_ms: 15,
            scorer_error: None,
            combo_members: None,
            combos_scored: None,
            combos_dampened: None,
            combo_dampen: None,
        };
        writeln!(file, "{}", serde_json::to_string(&accepted).unwrap()).unwrap();
        writeln!(file, "{}", serde_json::to_string(&rejected).unwrap()).unwrap();
        let report = report_from_journal(file.path()).unwrap();
        assert_eq!(report.experiments, 2);
        assert_eq!(report.acceptances, 1);
        assert_eq!(report.screen_candidates_scored, 2);
        assert_eq!(report.candidates_scored, 2); // 1 + 1 promote candidates
        assert!(report.projected_batches_per_45_min > 0.0);
        assert_eq!(report.improvement_series.len(), 2);
        assert_eq!(report.focus_history.len(), 1);
        assert_eq!(report.focus_history[0].accepts, 1);
        let random = report
            .strategies
            .iter()
            .find(|s| s.strategy == "random")
            .unwrap();
        assert_eq!(random.appearances_total, 2);
        assert_eq!(random.wins, 1);
        assert!((random.acceptance_rate - 0.5).abs() < 1e-12);
        let backprop = report
            .strategies
            .iter()
            .find(|s| s.strategy == "backprop")
            .unwrap();
        assert_eq!(backprop.wins, 0);
        assert_eq!(backprop.appearances_total, 1);
        assert_eq!(backprop.appearances_in_accepted, 1);
    }
}
