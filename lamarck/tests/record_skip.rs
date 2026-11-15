//! Tests for the opt-in record-skip experiment (issue #267).

use std::io::Write;

use neat_ai_lamarck::record_skip::experiment::{
    BenchRecord, CorpusSource, GridPoint, Manifest, Row, git_commit, load_corpus, neat_core_tag,
    neat_core_tag_from, refine, run_manifest, synthetic_corpus, synthetic_parent, write_outputs,
};
use neat_ai_lamarck::record_skip::schedule::{
    DifficultyScope, DifficultyTracker, ScheduleConfig, ScheduleError, SkipPolicy, StratumEdges,
};
use neat_ai_lamarck::record_skip::stats::{Verdict, verdict};
use neat_core::parse_creature_json;
use rand::SeedableRng;
use rand::rngs::StdRng;

fn scope(n: usize) -> DifficultyScope {
    DifficultyScope {
        parent_fingerprint: 7,
        corpus_key: "k".to_string(),
        record_count: n,
    }
}

fn config(alpha: f64, revisit: f64, rare: f64) -> ScheduleConfig {
    ScheduleConfig {
        ewma_alpha: alpha,
        warmup_epochs: 1,
        revisit_share: revisit,
        rare_stratum_share: rare,
    }
}

/// Tracker over `strata` with record `i` observed at error `errors[i]`.
fn observed(strata: Vec<u16>, errors: &[f64], cfg: ScheduleConfig) -> DifficultyTracker {
    let mut t = DifficultyTracker::new(scope(strata.len()), strata, cfg).unwrap();
    for (i, e) in errors.iter().enumerate() {
        t.observe(i, *e);
    }
    t
}

fn ramp(n: usize) -> Vec<f64> {
    (0..n).map(|i| i as f64).collect()
}

#[test]
fn ewma_blends_observations() {
    let mut t = DifficultyTracker::new(scope(1), vec![0], config(0.25, 0.0, 0.0)).unwrap();
    assert_eq!(t.ewma(0), None);
    t.observe(0, 4.0);
    assert_eq!(t.ewma(0), Some(4.0));
    t.observe(0, 2.0);
    assert_eq!(t.ewma(0), Some(3.5));
    assert_eq!(t.last_seen(0), Some(2.0));
}

#[test]
fn full_and_warmup_keep_everything() {
    let t = observed(vec![0; 10], &ramp(10), config(0.5, 0.0, 0.0));
    let mut rng = StdRng::seed_from_u64(1);
    let full = t.select(5, &SkipPolicy::Full, &mut rng).unwrap();
    assert_eq!(full.processed, 10);
    for policy in [
        SkipPolicy::EasySkip { reduction: 0.5 },
        SkipPolicy::RandomSkip { reduction: 0.5 },
    ] {
        let warm = t.select(0, &policy, &mut rng).unwrap();
        assert_eq!(warm.processed, 10, "warm-up keeps all for {policy:?}");
        assert!(warm.keep.iter().all(|&k| k));
        // Control: the same tracker skips once past warm-up.
        assert_eq!(t.select(1, &policy, &mut rng).unwrap().processed, 5);
    }
    // No scores yet: keep all even past warm-up.
    let fresh = DifficultyTracker::new(scope(10), vec![0; 10], config(0.5, 0.0, 0.0)).unwrap();
    let sel = fresh
        .select(3, &SkipPolicy::EasySkip { reduction: 0.5 }, &mut rng)
        .unwrap();
    assert_eq!(sel.processed, 10);
}

#[test]
fn easy_skip_keeps_the_hardest() {
    let t = observed(vec![0; 10], &ramp(10), config(0.5, 0.0, 0.0));
    let mut rng = StdRng::seed_from_u64(1);
    let sel = t
        .select(1, &SkipPolicy::EasySkip { reduction: 0.5 }, &mut rng)
        .unwrap();
    let kept: Vec<usize> = (0..10).filter(|&i| sel.keep[i]).collect();
    assert_eq!(kept, vec![5, 6, 7, 8, 9]);
    assert_eq!(sel.processed, 5);
    assert_eq!(sel.revisits, 0);
    assert!(sel.schedule_ops > 0);
}

#[test]
fn rare_stratum_is_protected() {
    let mut strata = vec![0u16; 20];
    strata[0] = 1; // 5% of records: below the 10% rare share
    let mut errors = ramp(20);
    errors[0] = 0.0; // the easiest record of all
    let t = observed(strata, &errors, config(0.5, 0.0, 0.10));
    assert_eq!(t.protected_count(), 1);
    let mut rng = StdRng::seed_from_u64(1);
    let sel = t
        .select(1, &SkipPolicy::EasySkip { reduction: 0.5 }, &mut rng)
        .unwrap();
    assert!(sel.keep[0], "protected record kept despite lowest error");
    assert_eq!(sel.protected, 1);
    assert_eq!(sel.processed, 10);
    // Control: with protection off the same record is skipped.
    let off = observed(
        {
            let mut s = vec![0u16; 20];
            s[0] = 1;
            s
        },
        &errors,
        config(0.5, 0.0, 0.0),
    );
    assert_eq!(off.protected_count(), 0);
    assert!(
        !off.select(1, &SkipPolicy::EasySkip { reduction: 0.5 }, &mut rng)
            .unwrap()
            .keep[0]
    );
}

#[test]
fn revisits_come_from_the_easy_pool() {
    let t = observed(vec![0; 20], &ramp(20), config(0.5, 0.4, 0.0));
    let mut rng = StdRng::seed_from_u64(9);
    let sel = t
        .select(1, &SkipPolicy::EasySkip { reduction: 0.5 }, &mut rng)
        .unwrap();
    // P = 10, S = 10, R = round(0.4 * 10) = 4, hard = 6 (indices 14..=19).
    assert_eq!(sel.processed, 10);
    assert_eq!(sel.revisits, 4);
    for i in 14..20 {
        assert!(sel.keep[i], "hard record {i} kept");
    }
    let revisits: Vec<usize> = (0..14).filter(|&i| sel.keep[i]).collect();
    assert_eq!(
        revisits.len(),
        4,
        "every revisit lies in the easy pool 0..14"
    );
}

#[test]
fn random_skip_matches_easy_count_and_varies_with_seed() {
    let mut strata = vec![0u16; 40];
    strata[0] = 1;
    let t = observed(strata, &ramp(40), config(0.5, 0.25, 0.10));
    let easy = t
        .select(
            1,
            &SkipPolicy::EasySkip { reduction: 0.5 },
            &mut StdRng::seed_from_u64(1),
        )
        .unwrap();
    let a = t
        .select(
            1,
            &SkipPolicy::RandomSkip { reduction: 0.5 },
            &mut StdRng::seed_from_u64(1),
        )
        .unwrap();
    let b = t
        .select(
            1,
            &SkipPolicy::RandomSkip { reduction: 0.5 },
            &mut StdRng::seed_from_u64(2),
        )
        .unwrap();
    assert_eq!(easy.processed, 20);
    assert_eq!(a.processed, easy.processed);
    assert_eq!(b.processed, easy.processed);
    assert_eq!(a.revisits, 0);
    assert!(a.keep[0] && b.keep[0], "protected record kept");
    assert_ne!(a.keep, b.keep, "different seeds, different masks");
}

#[test]
fn ensure_scope_refuses_other_parent_or_corpus() {
    let t = DifficultyTracker::new(scope(3), vec![0; 3], config(0.5, 0.0, 0.0)).unwrap();
    assert!(t.ensure_scope(&scope(3)).is_ok());
    let mut other_parent = scope(3);
    other_parent.parent_fingerprint = 8;
    assert!(matches!(
        t.ensure_scope(&other_parent),
        Err(ScheduleError::ScopeMismatch { .. })
    ));
    let mut other_corpus = scope(3);
    other_corpus.corpus_key = "other".to_string();
    assert!(matches!(
        t.ensure_scope(&other_corpus),
        Err(ScheduleError::ScopeMismatch { .. })
    ));
}

#[test]
fn config_and_reduction_refusals() {
    assert!(config(0.5, 0.25, 0.1).validate().is_ok());
    let cases = [
        (config(0.0, 0.25, 0.1), "ewma_alpha"),
        (config(1.5, 0.25, 0.1), "ewma_alpha"),
        (config(0.5, 1.0, 0.1), "revisit_share"),
        (
            ScheduleConfig {
                warmup_epochs: 0,
                ..config(0.5, 0.25, 0.1)
            },
            "warmup_epochs",
        ),
    ];
    for (cfg, field) in cases {
        let msg = cfg.validate().unwrap_err().to_string();
        assert!(msg.contains(field), "{msg} should name {field}");
    }
    assert_eq!(
        SkipPolicy::EasySkip { reduction: 0.0 }.validate(),
        Err(ScheduleError::InvalidReduction(0.0))
    );
    assert_eq!(
        SkipPolicy::RandomSkip { reduction: 1.0 }.validate(),
        Err(ScheduleError::InvalidReduction(1.0))
    );
    assert!(SkipPolicy::EasySkip { reduction: 0.5 }.validate().is_ok());
}

#[test]
fn stratum_edges_bin_first_targets() {
    let bin = |values: &[f32]| {
        let edges = StratumEdges::from_values(values, 4);
        values
            .iter()
            .map(|&v| edges.assign(v))
            .collect::<Vec<u16>>()
    };
    assert_eq!(bin(&[0.0, 1.0, 2.0, 3.0, 4.0]), vec![0, 1, 2, 3, 3]);
    assert_eq!(bin(&[2.5, 2.5, 2.5]), vec![0, 0, 0]);
}

fn smoke_setup(
    manifest: &Manifest,
) -> (
    neat_core::CreatureExport,
    neat_ai_lamarck::record_skip::experiment::Corpus,
    DifficultyScope,
) {
    let parent = synthetic_parent(manifest.parent_seed);
    let corpus = synthetic_corpus(1, manifest.train_records, manifest.validation_records);
    let scope = DifficultyScope {
        parent_fingerprint: neat_ai_lamarck::memo::creature_fingerprint(&parent),
        corpus_key: corpus.key.clone(),
        record_count: corpus.train.len(),
    };
    (parent, corpus, scope)
}

fn sched(manifest: &Manifest, rare: f64) -> ScheduleConfig {
    ScheduleConfig {
        ewma_alpha: 0.5,
        warmup_epochs: manifest.warmup_epochs,
        revisit_share: 0.0,
        rare_stratum_share: rare,
    }
}

#[test]
fn refine_full_lowers_train_mse_and_keeps_parent() {
    let m = Manifest::smoke();
    let (parent, corpus, scope) = smoke_setup(&m);
    let before = serde_json::to_string(&parent).unwrap();
    let out = refine(
        &parent,
        &corpus,
        &scope,
        &SkipPolicy::Full,
        &sched(&m, 0.1),
        &m,
        0.2,
        1,
    )
    .unwrap();
    assert!(
        out.final_train_mse < out.initial_train_mse,
        "{} !< {}",
        out.final_train_mse,
        out.initial_train_mse
    );
    assert_eq!(out.record_passes, 4 * 200);
    assert_eq!(serde_json::to_string(&parent).unwrap(), before);
}

#[test]
fn easy_skip_record_passes_follow_the_rule() {
    let m = Manifest::smoke();
    let (parent, corpus, scope) = smoke_setup(&m);
    let cfg = sched(&m, 0.0);
    let full = refine(
        &parent,
        &corpus,
        &scope,
        &SkipPolicy::Full,
        &cfg,
        &m,
        0.2,
        1,
    )
    .unwrap();
    let easy = refine(
        &parent,
        &corpus,
        &scope,
        &SkipPolicy::EasySkip { reduction: 0.5 },
        &cfg,
        &m,
        0.2,
        1,
    )
    .unwrap();
    // One full warm-up epoch of 200, then P = 200 - round(0.5 * 200) = 100 for 3 epochs.
    assert_eq!(easy.record_passes, 200 + 3 * 100);
    assert_eq!(full.record_passes, 800);
    assert!(easy.record_passes < full.record_passes);
    assert_eq!(easy.arm, "easy-50");
    // Control: a mismatched scope is refused.
    let mut wrong = scope.clone();
    wrong.corpus_key = "other".to_string();
    let err = refine(
        &parent,
        &corpus,
        &wrong,
        &SkipPolicy::Full,
        &cfg,
        &m,
        0.2,
        1,
    )
    .unwrap_err();
    assert!(err.contains("scope mismatch"), "{err}");
}

#[test]
fn manifest_refusals() {
    assert!(Manifest::predeclared().validate().is_ok());
    assert!(Manifest::smoke().validate().is_ok());

    let mut m = Manifest::smoke();
    m.version = 1;
    assert!(m.validate().unwrap_err().contains("version"));

    let mut m = Manifest::smoke();
    m.learning_rate_grid.clear();
    assert!(m.validate().unwrap_err().contains("learningRateGrid"));

    for bad in [0.0, -0.1, f64::NAN, f64::INFINITY] {
        let mut m = Manifest::smoke();
        m.learning_rate_grid = vec![0.1, bad];
        assert!(m.validate().unwrap_err().contains("learningRateGrid"));
    }

    let mut m = Manifest::smoke();
    m.max_record_passes = 0;
    assert!(m.validate().unwrap_err().contains("maxRecordPasses"));

    let mut m = Manifest::smoke();
    m.validation_seeds = vec![1, 5];
    assert!(m.validate().unwrap_err().contains("overlap"));

    let mut m = Manifest::smoke();
    m.validation_seeds.clear();
    assert!(m.validate().unwrap_err().contains("validationSeeds"));

    let mut m = Manifest::smoke();
    m.reductions = vec![0.5, 1.0];
    assert!(m.validate().unwrap_err().contains("reductions"));

    let mut m = Manifest::smoke();
    m.epochs = 0;
    assert!(m.validate().unwrap_err().contains("epochs"));

    let mut m = Manifest::smoke();
    m.warmup_epochs = 0;
    assert!(m.validate().unwrap_err().contains("warmupEpochs"));

    for bad in [-0.1, 1.0, f64::NAN] {
        let mut m = Manifest::smoke();
        m.rare_stratum_share = bad;
        assert!(m.validate().unwrap_err().contains("rareStratumShare"));
    }
    let mut m = Manifest::smoke();
    m.rare_stratum_share = 0.0;
    assert!(m.validate().is_ok());

    for bad in [0.0, -0.5, 1.5, f64::NAN] {
        let mut m = Manifest::smoke();
        m.calibration_grid[0] = GridPoint {
            ewma_alpha: bad,
            revisit_share: 0.1,
        };
        assert!(m.validate().unwrap_err().contains("ewmaAlpha"));
    }
    for bad in [-0.1, 1.0, f64::NAN] {
        let mut m = Manifest::smoke();
        m.calibration_grid[0] = GridPoint {
            ewma_alpha: 0.5,
            revisit_share: bad,
        };
        assert!(m.validate().unwrap_err().contains("revisitShare"));
    }
    // Allowed controls: the inclusive upper bound of alpha and a zero revisit share.
    let mut m = Manifest::smoke();
    m.calibration_grid[0] = GridPoint {
        ewma_alpha: 1.0,
        revisit_share: 0.0,
    };
    assert!(m.validate().is_ok());
}

fn tiny() -> Manifest {
    Manifest {
        train_records: 100,
        validation_records: 50,
        reductions: vec![0.5],
        calibration_seeds: vec![1],
        validation_seeds: vec![1001, 1002],
        ..Manifest::smoke()
    }
}

#[test]
fn smoke_run_is_complete_and_deterministic() {
    let m = tiny();
    let a = run_manifest(&m, &CorpusSource::Synthetic).unwrap();
    let b = run_manifest(&m, &CorpusSource::Synthetic).unwrap();
    // 1 seed * 1 reduction * 2 grid * 2 arms + 2 seeds * 3 arms.
    // Plus 4 preparation runs (1 seed * 4 rates).
    assert_eq!(a.coverage.planned, 4 + 6 + 4);
    assert_eq!(a.coverage.completed, 14);
    assert_eq!(a.coverage.failed, 0);
    assert_eq!(a.rows.len(), a.coverage.planned);
    let calib_passes: u64 = a
        .rows
        .iter()
        .filter_map(|r| match r {
            Row::Ok(o) if o.phase == "calibration" => Some(o.record_passes),
            _ => None,
        })
        .sum();
    assert_eq!(a.calibration_cost.record_passes, calib_passes);
    assert!(calib_passes > 0);
    assert_eq!(a.summary_json().unwrap(), b.summary_json().unwrap());
    assert!(!a.summary_json().unwrap().contains("\"rows\""));

    let dir = tempfile::tempdir().unwrap();
    write_outputs(dir.path(), &a).unwrap();
    let runs = std::fs::read_to_string(dir.path().join("runs.jsonl")).unwrap();
    assert_eq!(runs.lines().count(), 14);
    assert!(dir.path().join("manifest.json").exists());
    assert!(dir.path().join("summary.json").exists());
}

fn write_files(dir: &std::path::Path, records: usize) -> neat_core::CreatureExport {
    let json = r#"{"semanticVersion":"4.0.0","forwardOnly":true,"input":2,"output":1,
      "neurons":[{"type":"output","uuid":"o0","bias":0.0,"squash":"IDENTITY"}],
      "synapses":[{"fromUUID":"input-0","toUUID":"o0","weight":0.1},
                  {"fromUUID":"input-1","toUUID":"o0","weight":0.1}]}"#;
    let mut f = std::fs::File::create(dir.join("0.bin")).unwrap();
    for i in 0..records {
        let x0 = ((i * 37) % 100) as f32 / 50.0 - 1.0;
        let x1 = ((i * 53) % 100) as f32 / 50.0 - 1.0;
        let y = 0.3 * x0 - 0.2 * x1;
        for v in [x0, x1, y] {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
    }
    parse_creature_json(json).unwrap()
}

#[test]
fn files_mode_runs_and_load_corpus_checks_counts() {
    let dir = tempfile::tempdir().unwrap();
    let creature = write_files(dir.path(), 150);

    let corpus = load_corpus(dir.path(), &creature, 50, 1000).unwrap();
    assert_eq!(corpus.train.len(), 100);
    assert_eq!(corpus.validation.len(), 50);
    let _: &BenchRecord = &corpus.train[0];

    let err = load_corpus(dir.path(), &creature, 150, 1000).unwrap_err();
    assert!(
        err.contains("loaded 150") && err.contains("validation_records 150"),
        "{err}"
    );

    let report = run_manifest(
        &tiny(),
        &CorpusSource::Files {
            corpus,
            parent: creature,
        },
    )
    .unwrap();
    assert_eq!(report.coverage.completed, report.coverage.planned);
    assert_eq!(report.coverage.failed, 0);
    assert!(report.source.starts_with("files: corpus corpus:"));
}

#[test]
fn verdict_rules() {
    assert_eq!(verdict((-0.2, -0.1), &[-0.01]), Verdict::Better);
    assert_eq!(verdict((-0.2, -0.1), &[0.01]), Verdict::Inconclusive);
    // No rare-stratum data means the guard cannot pass, so no `better`.
    assert_eq!(verdict((-0.2, -0.1), &[]), Verdict::Inconclusive);
    assert_eq!(verdict((0.1, 0.2), &[-0.01]), Verdict::Worse);
    assert_eq!(verdict((-0.1, 0.1), &[-0.01]), Verdict::Inconclusive);
}

#[test]
fn tracker_refuses_strata_that_do_not_match_record_count() {
    let err = DifficultyTracker::new(scope(3), vec![0; 2], config(0.5, 0.0, 0.0)).unwrap_err();
    assert_eq!(
        err,
        ScheduleError::StrataLength {
            expected: 3,
            found: 2
        }
    );
    assert!(err.to_string().contains("strata length 2"));
    // Allowed control: matching lengths are accepted.
    assert!(DifficultyTracker::new(scope(3), vec![0; 3], config(0.5, 0.0, 0.0)).is_ok());
}

#[test]
fn reported_neat_core_tag_follows_the_cargo_toml_pin() {
    // Read the pin from the manifest at test time, so a moved pin cannot break this.
    let path = format!("{}/Cargo.toml", env!("CARGO_MANIFEST_DIR"));
    let toml = std::fs::read_to_string(path).unwrap();
    let tag = neat_core_tag_from(&toml);
    assert!(tag.starts_with('v'), "{tag}");
    assert_eq!(neat_core_tag(), tag);

    let report = run_manifest(&tiny(), &CorpusSource::Synthetic).unwrap();
    assert_eq!(report.versions.neat_core_tag, tag);

    // The parser follows whatever tag the manifest carries.
    let moved =
        "[dependencies]\nother = { tag = \"v9\" }\nneat-core = { git = \"u\", tag = \"v9.9.9\" }\n";
    assert_eq!(neat_core_tag_from(moved), "v9.9.9");
    assert_eq!(neat_core_tag_from("[dependencies]\n"), "unknown");
}

/// The committed version-1 manifest, before the learning-rate grid and compute
/// limit were added.
const VERSION_1_MANIFEST: &str = r#"{
  "version": 1,
  "epochs": 30,
  "trainRecords": 2000,
  "validationRecords": 1000,
  "parentSeed": 267,
  "calibrationSeeds": [
    1,
    2,
    3,
    4,
    5,
    6
  ],
  "validationSeeds": [
    1001,
    1002,
    1003,
    1004,
    1005,
    1006,
    1007,
    1008,
    1009,
    1010,
    1011,
    1012
  ],
  "reductions": [
    0.25,
    0.5,
    0.75
  ],
  "strataBins": 8,
  "rareStratumShare": 0.1,
  "warmupEpochs": 1,
  "calibrationGrid": [
    {
      "ewmaAlpha": 0.3,
      "revisitShare": 0.1
    },
    {
      "ewmaAlpha": 0.3,
      "revisitShare": 0.25
    },
    {
      "ewmaAlpha": 0.7,
      "revisitShare": 0.1
    },
    {
      "ewmaAlpha": 0.7,
      "revisitShare": 0.25
    }
  ],
  "learningRate": 0.2,
  "bootstrapResamples": 2000,
  "bootstrapSeed": 267,
  "forgettingTolerance": 0.0001
}"#;

#[test]
fn version_1_manifest_file_fails_loudly() {
    let err = serde_json::from_str::<Manifest>(VERSION_1_MANIFEST).unwrap_err();
    let text = err.to_string();
    assert!(
        text.contains("learningRateGrid") || text.contains("maxRecordPasses"),
        "{text}"
    );
    // Control: the current predeclared manifest round-trips.
    let current = serde_json::to_string(&Manifest::predeclared()).unwrap();
    assert_eq!(
        serde_json::from_str::<Manifest>(&current).unwrap(),
        Manifest::predeclared()
    );
}

#[test]
fn compute_limit_refuses_a_manifest_over_budget() {
    let mut m = tiny();
    // runs: 1 seed * 4 rates + 1 seed * 1 reduction * 2 grid * 2 arms + 2 seeds * 3 arms.
    let bound = ((4 + 4 + 6) * 4 * 100) as u64;
    m.max_record_passes = bound - 1;
    assert!(m.validate().is_ok());
    let err = run_manifest(&m, &CorpusSource::Synthetic).unwrap_err();
    assert!(err.contains("maxRecordPasses"), "{err}");
    assert!(err.contains(&bound.to_string()), "{err}");

    m.max_record_passes = bound;
    let report = run_manifest(&m, &CorpusSource::Synthetic).unwrap();
    assert_eq!(report.compute.planned_bound_record_passes, bound);
    assert_eq!(report.compute.limit_record_passes, bound);
    assert!(report.compute.actual_record_passes <= bound);
    assert_eq!(
        report.compute.actual_record_passes,
        report.preparation_cost.record_passes
            + report.calibration_cost.record_passes
            + report.validation_cost.record_passes
    );
    assert!(report.compute.actual_record_passes > 0);
}

#[test]
fn files_mode_compute_bound_uses_the_loaded_training_count() {
    let dir = tempfile::tempdir().unwrap();
    // 170 records less 50 validation: 120 training, not tiny()'s 100.
    let creature = write_files(dir.path(), 170);
    let corpus = load_corpus(dir.path(), &creature, 50, 1000).unwrap();
    assert_eq!(corpus.train.len(), 120);
    let source = CorpusSource::Files {
        corpus,
        parent: creature,
    };

    let mut m = tiny();
    assert_eq!(m.train_records, 100);
    // 4 rates + 4 calibration + 6 validation runs.
    let bound = 14 * u64::from(m.epochs) * 120;
    m.max_record_passes = bound - 1;
    let err = run_manifest(&m, &source).unwrap_err();
    assert!(err.contains("maxRecordPasses"), "{err}");
    assert!(err.contains(&bound.to_string()), "{err}");

    m.max_record_passes = bound;
    let report = run_manifest(&m, &source).unwrap();
    assert_eq!(report.compute.planned_bound_record_passes, bound);
    assert!(report.compute.actual_record_passes <= bound);
}

#[test]
fn learning_rate_sweep_is_counted_and_chooses_by_training_mse() {
    let mut m = tiny();
    m.learning_rate_grid = vec![1e-6, 0.2];
    let report = run_manifest(&m, &CorpusSource::Synthetic).unwrap();
    assert_eq!(report.chosen_learning_rate, 0.2);
    assert_eq!(
        report.preparation_cost.runs,
        m.calibration_seeds.len() * m.learning_rate_grid.len()
    );
    let rates: Vec<f64> = report
        .learning_rate_sweep
        .iter()
        .map(|e| e.learning_rate)
        .collect();
    assert_eq!(rates, [1e-6, 0.2]);
    let tiny_rate = report.learning_rate_sweep[0].mean_train_mse.unwrap();
    let normal = report.learning_rate_sweep[1].mean_train_mse.unwrap();
    assert!(normal < tiny_rate, "{normal} !< {tiny_rate}");
    for e in &report.learning_rate_sweep {
        assert_eq!(e.n_runs, m.calibration_seeds.len());
    }
    // The sweep reads training MSE: with one calibration seed each entry is
    // that seed's preparation row, and validation MSE would differ from it.
    let prep: Vec<_> = report
        .rows
        .iter()
        .filter_map(|r| match r {
            Row::Ok(o) if o.phase == "preparation" => Some(o),
            _ => None,
        })
        .collect();
    assert_eq!(prep.len(), report.learning_rate_sweep.len());
    for (e, o) in report.learning_rate_sweep.iter().zip(&prep) {
        assert_eq!(e.mean_train_mse, Some(o.final_train_mse));
        assert_ne!(e.mean_train_mse, Some(o.final_validation_mse));
    }
    // 2 prep + 4 calibration + 6 validation.
    assert_eq!(report.coverage.planned, 2 + 4 + 6);
    assert_eq!(report.coverage.completed, report.coverage.planned);
    // Control: reversing the grid order still picks the normal rate.
    m.learning_rate_grid.reverse();
    let again = run_manifest(&m, &CorpusSource::Synthetic).unwrap();
    assert_eq!(again.chosen_learning_rate, 0.2);
}

#[test]
fn arms_summarise_validation_runs() {
    let m = tiny();
    let report = run_manifest(&m, &CorpusSource::Synthetic).unwrap();
    let names: Vec<&str> = report.arms.iter().map(|a| a.arm.as_str()).collect();
    assert_eq!(names, ["full", "random-50", "easy-50"]);
    for a in &report.arms {
        assert_eq!(a.n_runs, m.validation_seeds.len());
        assert_eq!(a.strata_validation_mse.len(), m.strata_bins);
    }
    let rows: Vec<f64> = report
        .rows
        .iter()
        .filter_map(|r| match r {
            Row::Ok(o) if o.phase == "validation" && o.arm == "easy-50" => Some(o.forgotten_share),
            _ => None,
        })
        .collect();
    assert_eq!(rows.len(), 2);
    let expected = rows.iter().sum::<f64>() / rows.len() as f64;
    let easy = &report.arms[2];
    assert!((easy.forgotten_share.unwrap() - expected).abs() < 1e-12);
    assert!(easy.record_passes.unwrap() < report.arms[0].record_passes.unwrap());
    let json = report.summary_json().unwrap();
    assert!(json.contains("\"arms\""));
}

#[test]
fn git_commit_names_the_commit_or_the_cause() {
    let git = |dir: &std::path::Path, args: &[&str]| {
        let out = std::process::Command::new("git")
            .arg("-C")
            .arg(dir)
            .args(["-c", "user.name=t", "-c", "user.email=t@t"])
            .args(["-c", "commit.gpgsign=false"])
            .args(args)
            .output()
            .unwrap();
        assert!(out.status.success(), "{args:?}: {out:?}");
    };
    let dir = tempfile::tempdir().unwrap();
    git(dir.path(), &["init", "-q"]);
    let file = dir.path().join("f.txt");
    std::fs::write(&file, "one").unwrap();
    git(dir.path(), &["add", "f.txt"]);
    git(dir.path(), &["commit", "-q", "-m", "x"]);
    let clean = git_commit(dir.path());
    assert_eq!(clean.len(), 40, "{clean}");
    assert!(clean.chars().all(|c| c.is_ascii_hexdigit()), "{clean}");

    std::fs::write(&file, "two").unwrap();
    let dirty = git_commit(dir.path());
    assert_eq!(dirty, format!("{clean}-dirty"));

    let empty = tempfile::tempdir().unwrap();
    let none = git_commit(empty.path());
    assert!(none.starts_with("unavailable: "), "{none}");
    assert!(none.contains("not a git repository"), "{none}");
}
