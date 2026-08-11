//! Multi-focus experiments (issue #109).
//!
//! Three things have to hold before proposing against several focus neurons is
//! worth anything:
//!
//! 1. `--focus-count 1` proposes the pre-#109 stream. The fixture in
//!    `fixtures/focus/k1-candidate-stream.txt` was captured from the commit
//!    before this change, so any leak of multi-focus plumbing into the default
//!    — a stray rng draw, a reshaped journal — fails here. The comparison is
//!    exact on structure and six significant figures on values, because the
//!    analysis reduction is not bit-identical across architectures.
//! 2. At `K > 1` the creature-wide passes still run **once** per experiment.
//!    An implementation that loops the whole analysis per focus would be
//!    *slower*, not faster, and would otherwise look correct.
//! 3. Each candidate names the focus it was proposed for, so a winner is
//!    attributable to one focus rather than to the whole set.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use neat_ai_lamarck::candidates::CandidateProvenance;
use neat_ai_lamarck::focus::FocusPolicy;
use neat_ai_lamarck::observations::StatsMode;
use neat_ai_lamarck::scorer::{DirectoryScorer, ScoreResult, ScoreSample, ScorerError};
use neat_ai_lamarck::{ExperimentRecord, JournalLine, LamarckConfig, RunResult, run_optimisation};

/// Two hidden neurons and an output, so `--focus-count 3` has a full set.
const CREATURE: &str = r#"{
  "semanticVersion":"4.0.0","forwardOnly":true,"input":2,"output":1,
  "neurons":[
    {"type":"hidden","uuid":"h1","bias":0.1,"squash":"TANH"},
    {"type":"hidden","uuid":"h2","bias":-0.2,"squash":"TANH"},
    {"type":"output","uuid":"o1","bias":0.0,"squash":"IDENTITY"}
  ],
  "synapses":[
    {"fromUUID":"input-0","toUUID":"h1","weight":1.0},
    {"fromUUID":"input-1","toUUID":"h2","weight":0.5},
    {"fromUUID":"h1","toUUID":"o1","weight":1.0},
    {"fromUUID":"h2","toUUID":"o1","weight":0.25}
  ]
}"#;

const SAMPLE_RECORDS: usize = 64;

/// Scores everything flat, so the incumbent survives the whole run.
struct FlatScorer;

impl DirectoryScorer for FlatScorer {
    fn score_directory_sampled(
        &self,
        candidates_dir: &Path,
        _training_data: &Path,
        _sample: ScoreSample,
    ) -> Result<BTreeMap<String, ScoreResult>, ScorerError> {
        Ok(flat_scores(candidates_dir, |_| 0.64))
    }
}

/// Improves exactly one stem, so the win belongs to exactly one focus.
struct WinningStemScorer {
    stem: String,
}

impl DirectoryScorer for WinningStemScorer {
    fn score_directory_sampled(
        &self,
        candidates_dir: &Path,
        _training_data: &Path,
        _sample: ScoreSample,
    ) -> Result<BTreeMap<String, ScoreResult>, ScorerError> {
        Ok(flat_scores(candidates_dir, |stem| {
            if stem == self.stem { 0.64 + 2e-6 } else { 0.64 }
        }))
    }
}

fn flat_scores(dir: &Path, score_of: impl Fn(&str) -> f64) -> BTreeMap<String, ScoreResult> {
    let mut map = BTreeMap::new();
    for entry in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        let Some(stem) = name.strip_suffix(".json") else {
            continue;
        };
        let score = if stem == "baseline" {
            0.64
        } else {
            score_of(stem)
        };
        map.insert(
            stem.to_string(),
            ScoreResult {
                score,
                error: 1.0 - score,
                complexity_penalty: 0.0,
            },
        );
    }
    map
}

/// Deterministic xorshift sample: two inputs and a target per record.
fn write_sample(dir: &Path) {
    let mut bytes = Vec::new();
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    for _ in 0..SAMPLE_RECORDS {
        for _ in 0..3 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let v = ((state >> 11) as f64 / (1u64 << 53) as f64) as f32 * 2.0 - 1.0;
            bytes.extend_from_slice(&v.to_le_bytes());
        }
    }
    std::fs::write(dir.join("0.bin"), &bytes).unwrap();
}

/// The configuration the pre-#109 golden fixture was captured under.
fn config(dir: &Path, out: &str) -> LamarckConfig {
    let creature = dir.join("creature.json");
    let training = dir.join("data");
    std::fs::create_dir_all(&training).unwrap();
    write_sample(&training);
    std::fs::write(&creature, CREATURE).unwrap();
    LamarckConfig {
        creature,
        training_data: training,
        timeout: Duration::from_secs(60),
        max_experiments: Some(3),
        candidates: 6,
        min_improvement: 1e-6,
        seed: Some(42),
        scorer_path: PathBuf::from("rust_scorer"),
        output_dir: dir.join(out),
        stats_mode: StatsMode::Quick,
        quick_sample_records: SAMPLE_RECORDS as u64,
        focus_policy: FocusPolicy::Weighted,
        phase0_parity: false,
        screen_sample_rate: None,
        screen_promote_threshold: 0.0,
        ..LamarckConfig::default()
    }
}

fn experiments(result: &RunResult) -> Vec<ExperimentRecord> {
    std::fs::read_to_string(&result.journal_path)
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(
            |l| match JournalLine::parse(l).expect("journal line parses") {
                JournalLine::Experiment(record) => Some(*record),
                _ => None,
            },
        )
        .collect()
}

/// Replace every decimal number in `text` with a placeholder.
///
/// The mutation text is compared for its *shape* — strategy, target, squash,
/// grown-neuron UUID and ordering — because the analysis reduction is not
/// bit-identical across architectures: the same proposal reads
/// `w=-0.04081369646976907` on aarch64 and `w=-0.04081369645778815` on x86_64.
/// The proposal's actual value is compared separately, to six significant
/// figures, which is four orders of magnitude tighter than that drift.
///
/// Only runs containing a `.` are redacted, so hexadecimal UUID segments and
/// integer counts survive verbatim.
fn redact_decimals(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < chars.len() {
        let negative = chars[i] == '-' && chars.get(i + 1).is_some_and(char::is_ascii_digit);
        if !negative && !chars[i].is_ascii_digit() {
            out.push(chars[i]);
            i += 1;
            continue;
        }
        let start = i;
        if negative {
            i += 1;
        }
        while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
            i += 1;
        }
        let run: String = chars[start..i].iter().collect();
        if !run.contains('.') {
            out.push_str(&run);
            continue;
        }
        // A decimal may carry an exponent; a UUID segment never does.
        if i < chars.len() && (chars[i] == 'e' || chars[i] == 'E') {
            let mut j = i + 1;
            if j < chars.len() && (chars[j] == '+' || chars[j] == '-') {
                j += 1;
            }
            if j < chars.len() && chars[j].is_ascii_digit() {
                while j < chars.len() && chars[j].is_ascii_digit() {
                    j += 1;
                }
                i = j;
            }
        }
        out.push_str("<num>");
    }
    out
}

/// Six significant figures — tight enough to catch a changed proposal, loose
/// enough to survive a different FMA/vectorisation on another architecture.
fn approx(value: Option<f64>) -> String {
    value.map_or_else(|| "null".to_string(), |v| format!("{v:.6e}"))
}

/// One comparable line per experiment: focus, then each candidate's strategy,
/// focus, mutation shape and proposal values.
fn normalise(focus: &str, candidates: &[CandidateProvenance]) -> String {
    let body: Vec<String> = candidates
        .iter()
        .map(|p| {
            format!(
                "{}|{}|{}|{}|{}",
                serde_json::to_value(p.strategy).unwrap(),
                p.focus_neuron,
                redact_decimals(&p.mutation),
                approx(p.old_value),
                approx(p.new_value)
            )
        })
        .collect();
    format!("\"{focus}\"|{}\n", body.join(" ;; "))
}

fn candidate_stream(result: &RunResult) -> String {
    experiments(result)
        .iter()
        .map(|record| normalise(&record.focus_neuron, &record.candidates))
        .collect()
}

/// The pre-#109 stream, captured from commit 993f853 and normalised the same
/// way, so the two sides are compared on identical terms.
fn golden_stream() -> String {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/focus/k1-candidate-stream.txt");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
    text.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let (focus, candidates) = line
                .split_once('|')
                .unwrap_or_else(|| panic!("malformed fixture line: {line}"));
            let candidates: Vec<CandidateProvenance> = serde_json::from_str(candidates)
                .unwrap_or_else(|e| panic!("fixture candidates parse: {e}"));
            normalise(focus.trim_matches('"'), &candidates)
        })
        .collect()
}

/// The default must be indistinguishable from the run before multi-focus
/// existed — same focus choices, same candidates, same order.
#[test]
fn focus_count_one_reproduces_the_pre_change_candidate_stream() {
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = config(dir.path(), "out");
    assert_eq!(
        cfg.focus_count, 1,
        "one focus per experiment is the default"
    );
    cfg.focus_count = 1;
    let result = run_optimisation(&cfg, &FlatScorer).unwrap();

    let golden = golden_stream();
    assert!(
        golden.lines().count() >= 3,
        "the fixture must carry the whole pre-change stream"
    );
    assert_eq!(candidate_stream(&result), golden);
}

/// The normaliser must hide float drift without hiding a different proposal.
#[test]
fn the_normaliser_redacts_drift_but_not_uuids_or_changed_values() {
    // Same proposal, different last digits: identical after redaction.
    assert_eq!(
        redact_decimals("add input-0 -> o1 w=-0.04081369646976907 (scale=0.050)"),
        redact_decimals("add input-0 -> o1 w=-0.04081369645778815 (scale=0.050)")
    );
    // A different target, squash or grown-neuron UUID still differs.
    assert_ne!(
        redact_decimals("add input-0 -> o1 w=-0.04"),
        redact_decimals("add input-1 -> o1 w=-0.04")
    );
    assert_eq!(
        redact_decimals("split-neuron input-0 -> 016c3466-b81e-4083-a254-36b55143c127 -> h1"),
        "split-neuron input-0 -> 016c3466-b81e-4083-a254-36b55143c127 -> h1",
        "hexadecimal UUID segments carry no decimal point and must survive"
    );
    assert_eq!(redact_decimals("(count=64)"), "(count=64)");
    assert_eq!(redact_decimals("mean=1.2e-7 x"), "mean=<num> x");

    // Six significant figures: drift collapses, a real change does not.
    assert_eq!(
        approx(Some(0.10061501836753808)),
        approx(Some(0.10061501759267771))
    );
    assert_ne!(approx(Some(0.1006150)), approx(Some(0.1006151)));
    assert_eq!(approx(None), "null");
}

/// A single-focus journal keeps its pre-#109 shape: no focus set at all.
#[test]
fn focus_count_one_journals_no_focus_set() {
    let dir = tempfile::tempdir().unwrap();
    let result = run_optimisation(&config(dir.path(), "out"), &FlatScorer).unwrap();

    for record in experiments(&result) {
        assert_eq!(record.focus_neurons, None, "a single focus needs no set");
        for prov in &record.candidates {
            assert_eq!(prov.focus_neuron, record.focus_neuron);
        }
    }
    let encoded = std::fs::read_to_string(&result.journal_path).unwrap();
    assert!(
        !encoded.contains("\"focusNeurons\""),
        "a K=1 journal must not grow a focus-set field"
    );
    assert!(
        encoded.contains("\"focusCount\":1"),
        "the runHeader records the focus count in force"
    );
}

/// The creature-wide learning + output-residual pass is what #109 amortises, so
/// it must run once per experiment however many focuses the batch serves.
#[test]
fn three_focuses_share_one_learning_pass_per_experiment() {
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = config(dir.path(), "out");
    cfg.focus_count = 3;
    // The memo would remove scan two on a repeated focus; count the raw scans.
    cfg.analysis_memo_entries = 0;

    neat_ai_lamarck::reset_training_scan_count();
    let result = run_optimisation(&cfg, &FlatScorer).unwrap();
    let records = experiments(&result);
    assert!(!records.is_empty(), "the run completed experiments");

    let focus_scans: u64 = records
        .iter()
        .map(|r| r.focus_neurons.as_ref().map_or(1, Vec::len) as u64)
        .sum();
    assert_eq!(
        neat_ai_lamarck::training_scans_opened(),
        result.experiments + focus_scans,
        "one shared pre-focus scan per experiment, plus one focus scan per focus"
    );
    for record in &records {
        let set = record
            .focus_neurons
            .as_ref()
            .expect("a multi-focus experiment journals its set");
        assert_eq!(set.len(), 3, "three focuses were asked for and available");
        let mut sorted = set.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 3, "the focuses are distinct: {set:?}");
        assert_eq!(&record.focus_neuron, &set[0], "focusNeuron is the primary");
    }
}

/// The batch must actually be spread across the focus set, and every candidate
/// must name the focus it was proposed for — that is what makes a winner
/// attributable to one focus.
#[test]
fn candidates_are_split_across_the_focus_set_and_tagged_with_their_focus() {
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = config(dir.path(), "out");
    cfg.focus_count = 3;
    let result = run_optimisation(&cfg, &FlatScorer).unwrap();

    for record in experiments(&result) {
        let set = record.focus_neurons.clone().expect("focus set journalled");
        for prov in &record.candidates {
            assert!(
                set.contains(&prov.focus_neuron),
                "candidate focus {} is outside the experiment's focus set {set:?}",
                prov.focus_neuron
            );
        }
        let distinct: std::collections::BTreeSet<&str> = record
            .candidates
            .iter()
            .map(|p| p.focus_neuron.as_str())
            .collect();
        assert!(
            distinct.len() > 1,
            "a K=3 batch aimed every candidate at one focus: {distinct:?}"
        );
    }
}

/// An accepted winner belongs to the focus that proposed it, and the report
/// credits only that focus — the other focuses of the set stay at zero accepts.
#[test]
fn only_the_winning_focus_is_credited_with_the_accept() {
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = config(dir.path(), "out");
    cfg.focus_count = 3;
    cfg.max_experiments = Some(1);

    // Learn which candidate belongs to which focus, then make a candidate
    // belonging to a non-primary focus the winner.
    let probe = run_optimisation(&cfg, &FlatScorer).unwrap();
    let record = experiments(&probe)
        .into_iter()
        .next()
        .expect("one experiment");
    let primary = record.focus_neuron.clone();
    let (winner_index, winner_focus) = record
        .candidates
        .iter()
        .enumerate()
        .map(|(i, p)| (i, p.focus_neuron.clone()))
        .find(|(_, focus)| *focus != primary)
        .expect("a K=3 batch proposes for a focus other than the primary");

    let mut accept_cfg = config(dir.path(), "accept");
    accept_cfg.focus_count = 3;
    accept_cfg.max_experiments = Some(1);
    let result = run_optimisation(
        &accept_cfg,
        &WinningStemScorer {
            stem: format!("candidate-{winner_index:03}"),
        },
    )
    .unwrap();
    assert_eq!(result.acceptances, 1, "the scripted winner was accepted");

    let report = neat_ai_lamarck::report_from_journal(&result.journal_path).unwrap();
    let accepting: Vec<&str> = report
        .focus_history
        .iter()
        .filter(|h| h.accepts > 0)
        .map(|h| h.focus_neuron.as_str())
        .collect();
    assert_eq!(
        accepting,
        vec![winner_focus.as_str()],
        "only the winner's focus is credited with the accept"
    );
    assert_eq!(
        report.focus_history.len(),
        3,
        "every focus the experiment served appears in the history"
    );
    for entry in &report.focus_history {
        if entry.focus_neuron != winner_focus {
            assert_eq!(
                entry.cumulative_improvement, 0.0,
                "a sterile focus earns no improvement"
            );
        }
    }
}

/// A zero focus count is a configuration fault, not a silent clamp to one.
#[test]
fn a_zero_focus_count_aborts_the_run() {
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = config(dir.path(), "out");
    cfg.focus_count = 0;
    let err = run_optimisation(&cfg, &FlatScorer).unwrap_err();
    assert!(
        err.contains("--focus-count"),
        "the error names the flag at fault: {err}"
    );
}

/// A pinned focus is one neuron by definition, so `--focus-count` cannot widen it.
#[test]
fn a_pinned_focus_stays_a_single_focus() {
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = config(dir.path(), "out");
    cfg.focus_count = 3;
    cfg.focus_neuron = Some("o1".into());
    let result = run_optimisation(&cfg, &FlatScorer).unwrap();

    for record in experiments(&result) {
        assert_eq!(record.focus_neuron, "o1");
        assert_eq!(record.focus_neurons, None, "a pinned run serves one focus");
    }
}
