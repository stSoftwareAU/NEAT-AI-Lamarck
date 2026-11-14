//! Tests for the opt-in record-skip experiment (issue #267).

use std::io::Write;

use neat_ai_lamarck::record_skip::experiment::{
    BenchRecord, CorpusSource, Manifest, NEAT_CORE_TAG, Row, load_corpus, refine, run_manifest,
    synthetic_corpus, synthetic_parent, write_outputs,
};
use neat_ai_lamarck::record_skip::schedule::{
    DifficultyScope, DifficultyTracker, ScheduleConfig, ScheduleError, SkipPolicy, output_strata,
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
fn output_strata_bins() {
    assert_eq!(
        output_strata(&[0.0, 1.0, 2.0, 3.0, 4.0], 4),
        vec![0, 1, 2, 3, 3]
    );
    assert_eq!(output_strata(&[2.5, 2.5, 2.5], 4), vec![0, 0, 0]);
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
    let full = refine(&parent, &corpus, &scope, &SkipPolicy::Full, &cfg, &m, 1).unwrap();
    let easy = refine(
        &parent,
        &corpus,
        &scope,
        &SkipPolicy::EasySkip { reduction: 0.5 },
        &cfg,
        &m,
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
    let err = refine(&parent, &corpus, &wrong, &SkipPolicy::Full, &cfg, &m, 1).unwrap_err();
    assert!(err.contains("scope mismatch"), "{err}");
}

#[test]
fn manifest_refusals() {
    assert!(Manifest::predeclared().validate().is_ok());
    assert!(Manifest::smoke().validate().is_ok());

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
    assert_eq!(a.coverage.planned, 4 + 6);
    assert_eq!(a.coverage.completed, 10);
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
    assert_eq!(runs.lines().count(), 10);
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
    let d = [0.0];
    assert_eq!(verdict(&d, (-0.2, -0.1), &[-0.01]), Verdict::Better);
    assert_eq!(verdict(&d, (-0.2, -0.1), &[0.01]), Verdict::Inconclusive);
    assert_eq!(verdict(&d, (0.1, 0.2), &[-0.01]), Verdict::Worse);
    assert_eq!(verdict(&d, (-0.1, 0.1), &[-0.01]), Verdict::Inconclusive);
}

#[test]
fn neat_core_tag_matches_cargo_toml() {
    let path = format!("{}/Cargo.toml", env!("CARGO_MANIFEST_DIR"));
    let toml = std::fs::read_to_string(path).unwrap();
    assert!(
        toml.contains(&format!("tag = \"{NEAT_CORE_TAG}\"")),
        "NEAT_CORE_TAG {NEAT_CORE_TAG} out of step with Cargo.toml"
    );
}
