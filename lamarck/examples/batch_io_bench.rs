//! Paired batch-I/O benchmark for issue #114.
//!
//! Measures the two things the issue changes, on the same creature, in one
//! process:
//!
//! 1. **Serialisation format** — writing a whole candidate batch pretty-printed
//!    (the pre-#114 run) against compact (what `write_candidate_batch` does
//!    now), reporting bytes written and the wall clock of the write **and** of
//!    the scorer-side parse those bytes feed.
//! 2. **Promote directory** — presenting the promoted subset at a second path
//!    by copying every file (the pre-#114 run) against hard-linking it.
//!
//! The creature is synthesised to the production shape
//! (`README.md`: ~2511 inputs, ~1590 hidden neurons, ~21k synapses) so the
//! numbers are the ones a GRQ run would see; pass a creature path to measure a
//! real one instead.
//!
//! Usage (release build — debug timings are meaningless):
//!
//! ```text
//! cargo run --release --example batch_io_bench -- [CANDIDATES] [REPEATS] [CREATURE_JSON]
//! ```

use std::fs;
use std::path::Path;
use std::time::{Duration, Instant};

use neat_ai_lamarck::candidates::{Candidate, CandidateProvenance, CandidateStrategy};
use neat_core::{
    CreatureExport, NeuronExport, SynapseExport, creature_to_json, creature_to_json_pretty,
    parse_creature_json,
};

/// Production shape from the README.
const PROD_INPUTS: usize = 2511;
const PROD_HIDDEN: usize = 1590;
const PROD_SYNAPSES: usize = 21_889;

/// Candidates a production batch reaches on the fixed opening quotas.
const DEFAULT_CANDIDATES: usize = 29;

/// Interleaved repeats per arm.
const DEFAULT_REPEATS: usize = 5;

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let candidate_count: usize = args
        .next()
        .map(|a| a.parse().expect("CANDIDATES must be a positive integer"))
        .unwrap_or(DEFAULT_CANDIDATES);
    let repeats: usize = args
        .next()
        .map(|a| a.parse().expect("REPEATS must be a positive integer"))
        .unwrap_or(DEFAULT_REPEATS)
        .max(1);
    let incumbent = match args.next() {
        Some(path) => {
            let text = fs::read_to_string(&path).map_err(|e| format!("read {path}: {e}"))?;
            parse_creature_json(&text).map_err(|e| e.to_string())?
        }
        None => production_shaped_creature(),
    };
    let candidates = perturbed_candidates(&incumbent, candidate_count);

    println!("Batch I/O benchmark (issue #114)");
    println!(
        "creature: {} inputs, {} neurons, {} synapses; batch: baseline + {} candidates\n",
        incumbent.input,
        incumbent.neurons.len(),
        incumbent.synapses.len(),
        candidates.len()
    );

    let dir = tempfile::tempdir().map_err(|e| e.to_string())?;

    let mut pretty = measure_batch(&dir.path().join("pretty"), &incumbent, &candidates, false)?;
    let mut compact = measure_batch(&dir.path().join("compact"), &incumbent, &candidates, true)?;
    for _ in 1..repeats {
        pretty.keep_faster(measure_batch(
            &dir.path().join("pretty"),
            &incumbent,
            &candidates,
            false,
        )?);
        compact.keep_faster(measure_batch(
            &dir.path().join("compact"),
            &incumbent,
            &candidates,
            true,
        )?);
    }

    println!("Serialisation — whole batch written, then parsed back (best of {repeats})");
    println!("  arm      bytes written   write ms   read ms   parse ms");
    println!(
        "  pretty   {:>13}   {:>8.1}   {:>7.1}   {:>8.1}",
        pretty.bytes,
        ms(pretty.write),
        ms(pretty.read),
        ms(pretty.parse)
    );
    println!(
        "  compact  {:>13}   {:>8.1}   {:>7.1}   {:>8.1}",
        compact.bytes,
        ms(compact.write),
        ms(compact.read),
        ms(compact.parse)
    );
    println!(
        "  delta    {:>12.1}%   {:>7.1}%   {:>6.1}%   {:>7.1}%\n",
        percent_change(pretty.bytes as f64, compact.bytes as f64),
        percent_change(pretty.write.as_secs_f64(), compact.write.as_secs_f64()),
        percent_change(pretty.read.as_secs_f64(), compact.read.as_secs_f64()),
        percent_change(pretty.parse.as_secs_f64(), compact.parse.as_secs_f64()),
    );

    // Promote: the subset a screen typically admits (docs/baseline-reuse.md
    // measured ≈3.4 candidates per non-empty screen), plus the baseline.
    let promote_stems = promote_subset(&compact.stems, 4);
    let mut copied = measure_promote(
        &dir.path().join("compact"),
        &dir.path().join("promote-copy"),
        &promote_stems,
        false,
    )?;
    let mut linked = measure_promote(
        &dir.path().join("compact"),
        &dir.path().join("promote-link"),
        &promote_stems,
        true,
    )?;
    for _ in 1..repeats {
        copied.keep_faster(measure_promote(
            &dir.path().join("compact"),
            &dir.path().join("promote-copy"),
            &promote_stems,
            false,
        )?);
        linked.keep_faster(measure_promote(
            &dir.path().join("compact"),
            &dir.path().join("promote-link"),
            &promote_stems,
            true,
        )?);
    }

    println!(
        "Promote directory — {} file(s) presented again (best of {repeats})",
        promote_stems.len()
    );
    println!("  arm      bytes written   ms");
    println!(
        "  copy     {:>13}   {:>6.1}",
        copied.bytes,
        ms(copied.elapsed)
    );
    println!(
        "  link     {:>13}   {:>6.1}",
        linked.bytes,
        ms(linked.elapsed)
    );
    println!(
        "  delta    {:>12.1}%   {:>5.1}%",
        percent_change(copied.bytes as f64, linked.bytes as f64),
        percent_change(copied.elapsed.as_secs_f64(), linked.elapsed.as_secs_f64()),
    );

    Ok(())
}

struct BatchMeasurement {
    bytes: u64,
    write: Duration,
    read: Duration,
    parse: Duration,
    stems: Vec<String>,
}

impl BatchMeasurement {
    /// Keep the fastest observation of each phase across repeats.
    fn keep_faster(&mut self, other: Self) {
        self.write = self.write.min(other.write);
        self.read = self.read.min(other.read);
        self.parse = self.parse.min(other.parse);
        assert_eq!(
            self.bytes, other.bytes,
            "the same batch must write the same bytes"
        );
    }
}

/// Write a whole batch in one format, then parse every file back.
///
/// The parse side is the scorer's half of the cost: `rust_scorer` reads exactly
/// these bytes on the screen call, and the promoted subset again on the promote
/// call.
fn measure_batch(
    dir: &Path,
    incumbent: &CreatureExport,
    candidates: &[Candidate],
    compact: bool,
) -> Result<BatchMeasurement, String> {
    if dir.exists() {
        fs::remove_dir_all(dir).map_err(|e| e.to_string())?;
    }
    fs::create_dir_all(dir).map_err(|e| e.to_string())?;

    let serialise = |creature: &CreatureExport| -> Result<String, String> {
        if compact {
            creature_to_json(creature).map_err(|e| e.to_string())
        } else {
            creature_to_json_pretty(creature).map_err(|e| e.to_string())
        }
    };

    let mut stems = vec!["baseline".to_string()];
    for i in 0..candidates.len() {
        stems.push(format!("candidate-{i:03}"));
    }

    let start = Instant::now();
    fs::write(dir.join("baseline.json"), serialise(incumbent)?).map_err(|e| e.to_string())?;
    for (i, candidate) in candidates.iter().enumerate() {
        fs::write(
            dir.join(format!("candidate-{i:03}.json")),
            serialise(&candidate.creature)?,
        )
        .map_err(|e| e.to_string())?;
    }
    let write = start.elapsed();

    // Read and parse are timed apart: reading is the host's filesystem cache,
    // parsing is the work the scorer's CPU does on the bytes this arm chose.
    let start = Instant::now();
    let mut texts = Vec::with_capacity(stems.len());
    for stem in &stems {
        texts
            .push(fs::read_to_string(dir.join(format!("{stem}.json"))).map_err(|e| e.to_string())?);
    }
    let read = start.elapsed();

    let start = Instant::now();
    let mut parsed = 0usize;
    for text in &texts {
        let creature = parse_creature_json(text).map_err(|e| e.to_string())?;
        parsed += creature.neurons.len();
    }
    let parse = start.elapsed();
    assert!(parsed > 0, "the parse arm must actually parse creatures");

    Ok(BatchMeasurement {
        bytes: directory_bytes(dir)?,
        write,
        read,
        parse,
        stems,
    })
}

struct PromoteMeasurement {
    bytes: u64,
    elapsed: Duration,
}

impl PromoteMeasurement {
    /// Keep the fastest observation across repeats.
    fn keep_faster(&mut self, other: Self) {
        self.elapsed = self.elapsed.min(other.elapsed);
    }
}

/// Present `stems` from `source` at a second path, by copy or by hard link.
fn measure_promote(
    source: &Path,
    dir: &Path,
    stems: &[String],
    link: bool,
) -> Result<PromoteMeasurement, String> {
    if dir.exists() {
        fs::remove_dir_all(dir).map_err(|e| e.to_string())?;
    }
    fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let start = Instant::now();
    for stem in stems {
        let name = format!("{stem}.json");
        let (src, dst) = (source.join(&name), dir.join(&name));
        if link {
            fs::hard_link(&src, &dst).map_err(|e| e.to_string())?;
        } else {
            fs::copy(&src, &dst).map_err(|e| e.to_string())?;
        }
    }
    let elapsed = start.elapsed();
    // A hard link adds a directory entry, not bytes: count only new blocks.
    let bytes = if link { 0 } else { directory_bytes(dir)? };
    Ok(PromoteMeasurement { bytes, elapsed })
}

fn directory_bytes(dir: &Path) -> Result<u64, String> {
    let mut total = 0u64;
    for entry in fs::read_dir(dir).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        total += entry.metadata().map_err(|e| e.to_string())?.len();
    }
    Ok(total)
}

fn promote_subset(stems: &[String], count: usize) -> Vec<String> {
    stems.iter().take(count.min(stems.len())).cloned().collect()
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

fn percent_change(before: f64, after: f64) -> f64 {
    if before == 0.0 {
        return 0.0;
    }
    (after - before) / before * 100.0
}

/// A creature of the production shape, with the value spread real weights have.
fn production_shaped_creature() -> CreatureExport {
    let mut neurons = Vec::with_capacity(PROD_HIDDEN + 1);
    for i in 0..PROD_HIDDEN {
        neurons.push(NeuronExport {
            neuron_type: "hidden".into(),
            uuid: format!("neuron-{i:07}"),
            bias: pseudo_weight(i as u64),
            squash: Some(if i % 3 == 0 { "TANH" } else { "IDENTITY" }.into()),
            tags: None,
            extra: Default::default(),
        });
    }
    neurons.push(NeuronExport {
        neuron_type: "output".into(),
        uuid: "output-0".into(),
        bias: 0.0,
        squash: Some("IDENTITY".into()),
        tags: None,
        extra: Default::default(),
    });

    let mut synapses = Vec::with_capacity(PROD_SYNAPSES);
    for i in 0..PROD_SYNAPSES {
        // Inputs feed the hidden layer; the last slice feeds the output, so
        // every hidden neuron is reachable and the output has real fan-in.
        let to = i % PROD_HIDDEN;
        let from = if i % 7 == 0 && to > 0 {
            format!("neuron-{:07}", to - 1)
        } else {
            format!("input-{}", i % PROD_INPUTS)
        };
        synapses.push(SynapseExport {
            from_uuid: from,
            to_uuid: format!("neuron-{to:07}"),
            weight: pseudo_weight(i as u64 + 1),
            synapse_type: None,
            tags: None,
            extra: Default::default(),
        });
    }
    for i in 0..PROD_HIDDEN {
        synapses.push(SynapseExport {
            from_uuid: format!("neuron-{i:07}"),
            to_uuid: "output-0".into(),
            weight: pseudo_weight(i as u64 + 99),
            synapse_type: None,
            tags: None,
            extra: Default::default(),
        });
    }

    CreatureExport {
        input: PROD_INPUTS,
        output: 1,
        neurons,
        synapses,
        semantic_version: Some("4.0.0".into()),
        forward_only: true,
        uuid: None,
        tags: None,
        memetic: None,
        extra: Default::default(),
    }
}

/// Full-precision weights: a rounded value would serialise short and flatter
/// both arms equally, but real creature weights are long.
fn pseudo_weight(seed: u64) -> f64 {
    let x = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    let unit = (x >> 11) as f64 / (1u64 << 53) as f64;
    (unit - 0.5) * 2.0
}

/// One bias-nudge candidate per slot, as a real batch produces.
fn perturbed_candidates(incumbent: &CreatureExport, count: usize) -> Vec<Candidate> {
    (0..count)
        .map(|i| {
            let mut creature = incumbent.clone();
            let pos = i % creature.neurons.len();
            creature.neurons[pos].bias += pseudo_weight(i as u64 + 7) * 0.01;
            Candidate {
                creature,
                provenance: CandidateProvenance {
                    strategy: CandidateStrategy::Random,
                    focus_neuron: incumbent.neurons[pos].uuid.clone(),
                    mutation: format!("bench bias nudge {i}"),
                    old_value: None,
                    new_value: None,
                },
            }
        })
        .collect()
}
