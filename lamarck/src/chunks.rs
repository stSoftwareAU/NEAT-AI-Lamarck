//! Deterministic record chunking for the parallel analysis scans (issue #107).
//!
//! The analysis phase is a set of read-only reductions over the training
//! sample: every record contributes independently to a set of accumulators.
//! Those reductions parallelise, but float addition does not commute, so a
//! merge order that follows *completion* order would move the low-order bits
//! from run to run and break `--seed` replay.
//!
//! This module pins the order instead of the schedule:
//!
//! * the sample is cut into fixed [`ANALYSIS_CHUNK_RECORDS`]-record chunks, so
//!   the partition depends only on the sample — never on the thread count, the
//!   host or the core count;
//! * [`map_chunks`] hands chunks to workers in whatever order they become free,
//!   but returns the per-chunk results **indexed by chunk**, so the caller
//!   always merges ascending chunk index.
//!
//! One thread and eight threads therefore fold exactly the same partials in
//! exactly the same order, and produce bit-identical accumulators.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use neat_core::{TrainingDataConfig, TrainingRecord, find_bin_files};

/// Records folded into one analysis chunk.
///
/// Fixed on purpose: it is the unit that fixes the float summation order of
/// every parallel reduction. Changing it changes the low-order bits of the
/// analysis, so it is a deliberate, reviewable edit — not a runtime knob.
pub const ANALYSIS_CHUNK_RECORDS: u64 = 2048;

/// Default worker threads for one analysis scan.
///
/// Deliberately **not** `num_cpus`: the scorer and the analysis alternate today,
/// but a future overlap would turn "all cores" into contention that slows the
/// scorer down. Four leaves headroom on the 10-core measured host, and the
/// parallel reduction is bounded by training-data read bandwidth well before
/// the core count is.
pub const DEFAULT_ANALYSIS_THREADS: usize = 4;

/// One `.bin` file and where its records sit in the sample's global order.
#[derive(Debug, Clone)]
struct FileSpan {
    path: PathBuf,
    /// Global index of this file's first record.
    start: u64,
    /// Records in this file.
    records: u64,
}

/// A contiguous half-open range of sample records folded as one unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecordChunk {
    /// Position in the sample's chunk sequence — the merge key.
    pub index: usize,
    /// Global index of the first record in the chunk.
    pub start: u64,
    /// Records in the chunk.
    pub len: u64,
}

/// The sample one scan will read: its `.bin` files and their record layout.
///
/// Built once on the calling thread; every worker reads through it.
#[derive(Debug, Clone)]
pub struct SamplePlan {
    config: TrainingDataConfig,
    files: Vec<FileSpan>,
    total: u64,
}

impl SamplePlan {
    /// Measure the sample without reading any record data.
    ///
    /// `max_records` caps the sample exactly as the streaming scans do: the
    /// first `n` records in file order. A file whose size is not a whole number
    /// of records is an error rather than a silently truncated sample.
    pub fn new(
        training_data: &Path,
        config: TrainingDataConfig,
        max_records: Option<u64>,
    ) -> Result<Self, String> {
        let bin_files = find_bin_files(training_data).map_err(|e| e.to_string())?;
        let record_bytes = config.bytes_per_record() as u64;
        if record_bytes == 0 {
            return Err("training-data config has a zero-byte record".to_string());
        }

        let mut files = Vec::with_capacity(bin_files.len());
        let mut start = 0u64;
        for path in bin_files {
            let size = std::fs::metadata(&path)
                .map_err(|e| format!("failed to stat {}: {e}", path.display()))?
                .len();
            if size % record_bytes != 0 {
                return Err(format!(
                    "{} has size {size} which is not a multiple of the {record_bytes}-byte record",
                    path.display()
                ));
            }
            let records = size / record_bytes;
            if records == 0 {
                continue;
            }
            files.push(FileSpan {
                path,
                start,
                records,
            });
            start += records;
        }

        let total = match max_records {
            Some(limit) => start.min(limit),
            None => start,
        };
        Ok(Self {
            config,
            files,
            total,
        })
    }

    /// Records this scan will fold in, after the `max_records` cap.
    pub fn total_records(&self) -> u64 {
        self.total
    }

    /// Cut the sample into chunks of at most `chunk_records` records.
    ///
    /// Pass [`u64::MAX`] for a single chunk (the serial fold). The partition is
    /// a pure function of the sample size and `chunk_records`, which is what
    /// makes the merge thread-count independent.
    pub fn chunks(&self, chunk_records: u64) -> Vec<RecordChunk> {
        let stride = chunk_records.max(1);
        let mut chunks = Vec::new();
        let mut start = 0u64;
        while start < self.total {
            let len = stride.min(self.total - start);
            chunks.push(RecordChunk {
                index: chunks.len(),
                start,
                len,
            });
            start += len;
        }
        chunks
    }

    /// Open a reader positioned at the first record of `chunk`.
    pub fn reader(&self, chunk: &RecordChunk) -> ChunkReader<'_> {
        ChunkReader {
            plan: self,
            next: chunk.start,
            end: chunk.start.saturating_add(chunk.len).min(self.total),
            open: None,
            buffer: Vec::new(),
            cursor: 0,
            filled: 0,
        }
    }
}

/// Target size of one read from disk, rounded down to whole records.
///
/// neat-core's positional reader ([`neat_core::SeekingRecordReader`]) seeks and
/// reads a single record per call, so a chunk built on it pays a syscall per
/// record. Reading a batch and decoding out of it restores the streaming read
/// pattern while still starting at an arbitrary record; 64 KiB keeps the batch
/// inside cache at production record width (~6 records of 2 511 inputs).
const READ_BATCH_BYTES: usize = 1 << 16;

/// The `.bin` file a [`ChunkReader`] is currently reading from.
struct OpenFile {
    span: usize,
    file: File,
    /// Next record index within the file the handle is positioned at.
    next_local: u64,
}

/// Streaming reader over one [`RecordChunk`], spanning files as needed.
pub struct ChunkReader<'a> {
    plan: &'a SamplePlan,
    next: u64,
    end: u64,
    open: Option<OpenFile>,
    /// Batch of record bytes read in one go.
    buffer: Vec<u8>,
    /// Byte offset of the next undecoded record in `buffer`.
    cursor: usize,
    /// Bytes of `buffer` currently holding records.
    filled: usize,
}

impl ChunkReader<'_> {
    /// Read the next record of the chunk in place, returning `false` at its end.
    ///
    /// The record's buffers are refilled, reusing their capacity, so a steady
    /// scan performs no per-record allocation.
    pub fn next_record_into(&mut self, record: &mut TrainingRecord) -> Result<bool, String> {
        if self.next >= self.end {
            return Ok(false);
        }
        let record_bytes = self.plan.config.bytes_per_record();
        if self.cursor + record_bytes > self.filled {
            self.refill(record_bytes)?;
        }
        let bytes = &self.buffer[self.cursor..self.cursor + record_bytes];
        decode_record(bytes, self.plan.config.num_inputs, record);
        self.cursor += record_bytes;
        self.next += 1;
        Ok(true)
    }

    /// Read the next batch of records for the current file into `buffer`.
    fn refill(&mut self, record_bytes: usize) -> Result<(), String> {
        let span_idx = self.span_for(self.next).ok_or_else(|| {
            format!(
                "training sample has no record at index {} (sample holds {})",
                self.next, self.plan.total
            )
        })?;
        let span = &self.plan.files[span_idx];
        let local = self.next - span.start;

        // Re-open (or seek) whenever the handle is not already sitting on the
        // record we want — the first refill of a chunk always seeks.
        let needs_seek = match &self.open {
            Some(open) => open.span != span_idx || open.next_local != local,
            None => true,
        };
        if needs_seek {
            let mut file = File::open(&span.path)
                .map_err(|e| format!("failed to open {}: {e}", span.path.display()))?;
            file.seek(SeekFrom::Start(local * record_bytes as u64))
                .map_err(|e| format!("failed to seek {}: {e}", span.path.display()))?;
            self.open = Some(OpenFile {
                span: span_idx,
                file,
                next_local: local,
            });
        }

        let batch_records = (READ_BATCH_BYTES / record_bytes.max(1)).max(1) as u64;
        let left_in_file = span.records - local;
        let left_in_chunk = self.end - self.next;
        let take = batch_records.min(left_in_file).min(left_in_chunk) as usize;
        let want = take * record_bytes;
        if self.buffer.len() < want {
            self.buffer.resize(want, 0);
        }
        let open = self
            .open
            .as_mut()
            .expect("the file handle was just opened or reused");
        open.file
            .read_exact(&mut self.buffer[..want])
            .map_err(|e| format!("failed to read {}: {e}", span.path.display()))?;
        open.next_local += take as u64;
        self.cursor = 0;
        self.filled = want;
        Ok(())
    }

    /// Index of the file span holding the global record `index`.
    fn span_for(&self, index: u64) -> Option<usize> {
        self.plan
            .files
            .iter()
            .position(|span| index >= span.start && index < span.start + span.records)
    }
}

/// Decode one packed little-endian `f32` record into `record`.
///
/// neat-core keeps its equivalent decoder private and only exposes it through
/// per-record readers, so the batched path above decodes here. The format is
/// the one `neat_core::training_data` documents: `num_inputs` little-endian
/// `f32`s followed by the targets.
fn decode_record(bytes: &[u8], num_inputs: usize, record: &mut TrainingRecord) {
    let split = (num_inputs * std::mem::size_of::<f32>()).min(bytes.len());
    let (input_bytes, output_bytes) = bytes.split_at(split);
    record.inputs.clear();
    record
        .inputs
        .extend(input_bytes.chunks_exact(4).map(f32_from_le));
    record.outputs.clear();
    record
        .outputs
        .extend(output_bytes.chunks_exact(4).map(f32_from_le));
}

/// Decode one little-endian `f32` from a 4-byte chunk.
#[inline]
fn f32_from_le(chunk: &[u8]) -> f32 {
    f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
}

/// Run `work` over every chunk on up to `threads` workers, results in chunk order.
///
/// `init` builds one worker's reusable state (a compiled network, say) — it runs
/// once per worker, not once per chunk. The returned vector is indexed by
/// [`RecordChunk::index`], so callers merge in a fixed order regardless of which
/// worker finished first.
///
/// The first failing chunk stops the remaining work and is returned; a chunk
/// that produced no result at all is an error too, never a silently short merge.
pub fn map_chunks<S, T, I, W>(
    threads: usize,
    chunks: &[RecordChunk],
    init: I,
    work: W,
) -> Result<Vec<T>, String>
where
    S: Send,
    T: Send,
    I: Fn() -> Result<S, String> + Sync,
    W: Fn(&mut S, &RecordChunk) -> Result<T, String> + Sync,
{
    if threads == 0 {
        return Err("analysis worker count must be at least 1".to_string());
    }
    let workers = threads.min(chunks.len().max(1));
    if workers <= 1 {
        let mut state = init()?;
        let mut out = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            out.push(work(&mut state, chunk)?);
        }
        return Ok(out);
    }

    let next = AtomicUsize::new(0);
    let failed = AtomicBool::new(false);
    let error: Mutex<Option<String>> = Mutex::new(None);
    let slots: Mutex<Vec<Option<T>>> = Mutex::new((0..chunks.len()).map(|_| None).collect());

    let record_error = |message: String| {
        failed.store(true, Ordering::SeqCst);
        if let Ok(mut slot) = error.lock()
            && slot.is_none()
        {
            *slot = Some(message);
        }
    };

    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| {
                let mut state = match init() {
                    Ok(state) => state,
                    Err(e) => {
                        record_error(e);
                        return;
                    }
                };
                while !failed.load(Ordering::SeqCst) {
                    let i = next.fetch_add(1, Ordering::SeqCst);
                    let Some(chunk) = chunks.get(i) else {
                        break;
                    };
                    match work(&mut state, chunk) {
                        Ok(value) => match slots.lock() {
                            Ok(mut slots) => slots[i] = Some(value),
                            Err(e) => {
                                record_error(format!("analysis chunk results poisoned: {e}"));
                                return;
                            }
                        },
                        Err(e) => {
                            record_error(e);
                            return;
                        }
                    }
                }
            });
        }
    });

    if let Some(message) = error
        .into_inner()
        .map_err(|e| format!("analysis error slot poisoned: {e}"))?
    {
        return Err(message);
    }
    let slots = slots
        .into_inner()
        .map_err(|e| format!("analysis chunk results poisoned: {e}"))?;
    let mut out = Vec::with_capacity(slots.len());
    for (i, slot) in slots.into_iter().enumerate() {
        out.push(slot.ok_or_else(|| format!("analysis chunk {i} produced no result"))?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::{TempDir, tempdir};

    /// Write `files` `.bin` files of `records` records each, 2 inputs + 1 output.
    fn write_sample(files: &[usize]) -> TempDir {
        let dir = tempdir().unwrap();
        let mut value = 0.0f32;
        for (f, records) in files.iter().enumerate() {
            let mut file = std::fs::File::create(dir.path().join(format!("{f}.bin"))).unwrap();
            for _ in 0..*records {
                for _ in 0..3 {
                    file.write_all(&value.to_le_bytes()).unwrap();
                    value += 1.0;
                }
            }
            file.flush().unwrap();
        }
        dir
    }

    fn plan(dir: &TempDir, max_records: Option<u64>) -> SamplePlan {
        SamplePlan::new(dir.path(), TrainingDataConfig::new(2, 1), max_records).unwrap()
    }

    /// Every record the chunk plan covers, in chunk order.
    fn read_all(plan: &SamplePlan, chunk_records: u64) -> Vec<Vec<f32>> {
        let mut out = Vec::new();
        for chunk in plan.chunks(chunk_records) {
            let mut reader = plan.reader(&chunk);
            let mut record = TrainingRecord {
                inputs: Vec::new(),
                outputs: Vec::new(),
            };
            while reader.next_record_into(&mut record).unwrap() {
                let mut row = record.inputs.clone();
                row.extend_from_slice(&record.outputs);
                out.push(row);
            }
        }
        out
    }

    #[test]
    fn the_plan_counts_every_record_across_files() {
        let dir = write_sample(&[3, 5]);
        assert_eq!(plan(&dir, None).total_records(), 8);
    }

    #[test]
    fn the_plan_honours_the_record_cap() {
        let dir = write_sample(&[3, 5]);
        assert_eq!(plan(&dir, Some(6)).total_records(), 6);
        // A cap above the sample cannot invent records.
        assert_eq!(plan(&dir, Some(99)).total_records(), 8);
    }

    #[test]
    fn chunks_partition_the_sample_exactly_once() {
        let dir = write_sample(&[7, 6]);
        let plan = plan(&dir, None);
        let chunks = plan.chunks(5);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks.iter().map(|c| c.len).sum::<u64>(), 13);
        let mut expected_start = 0;
        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.index, i, "chunk index is the merge key");
            assert_eq!(
                chunk.start, expected_start,
                "chunks must not overlap or gap"
            );
            expected_start += chunk.len;
        }
    }

    #[test]
    fn a_chunk_stride_beyond_the_sample_is_one_chunk() {
        let dir = write_sample(&[4]);
        let chunks = plan(&dir, None).chunks(u64::MAX);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len, 4);
    }

    #[test]
    fn an_empty_sample_has_no_chunks() {
        let dir = tempdir().unwrap();
        let plan = SamplePlan::new(dir.path(), TrainingDataConfig::new(2, 1), None).unwrap();
        assert_eq!(plan.total_records(), 0);
        assert!(plan.chunks(ANALYSIS_CHUNK_RECORDS).is_empty());
    }

    #[test]
    fn reading_the_chunks_reproduces_the_sample_in_order() {
        // Chunk boundaries deliberately land inside and across files.
        let dir = write_sample(&[7, 6]);
        let plan = plan(&dir, None);
        let serial = read_all(&plan, u64::MAX);
        assert_eq!(serial.len(), 13);
        assert_eq!(serial[0], vec![0.0, 1.0, 2.0]);
        for stride in [1, 2, 5, 7, 13, 100] {
            assert_eq!(
                read_all(&plan, stride),
                serial,
                "chunk stride {stride} must not reorder or drop records"
            );
        }
    }

    #[test]
    fn a_capped_sample_reads_only_the_leading_records() {
        let dir = write_sample(&[7, 6]);
        let capped = read_all(&plan(&dir, Some(9)), 4);
        let full = read_all(&plan(&dir, None), 4);
        assert_eq!(capped.len(), 9);
        assert_eq!(capped, full[..9].to_vec());
    }

    #[test]
    fn a_ragged_file_fails_loudly_rather_than_truncating() {
        let dir = tempdir().unwrap();
        let mut file = std::fs::File::create(dir.path().join("0.bin")).unwrap();
        // Ten bytes is not a whole number of 12-byte records.
        file.write_all(&[0u8; 10]).unwrap();
        file.flush().unwrap();
        let err = SamplePlan::new(dir.path(), TrainingDataConfig::new(2, 1), None)
            .expect_err("a ragged file must not be read as a short sample");
        assert!(err.contains("not a multiple"), "{err}");
    }

    #[test]
    fn refilling_the_read_batch_does_not_drop_or_reorder_records() {
        // Records wide enough that one read batch holds only a handful, so a
        // chunk spans several refills — the boundary a naive buffer loses
        // records at.
        let inputs = 2_048usize;
        let config = TrainingDataConfig::new(inputs, 1);
        let per_batch = READ_BATCH_BYTES / config.bytes_per_record();
        assert!((2..12).contains(&per_batch), "batch holds {per_batch}");
        let records = per_batch * 2 + 3;

        let dir = tempdir().unwrap();
        let mut file =
            std::io::BufWriter::new(std::fs::File::create(dir.path().join("0.bin")).unwrap());
        for r in 0..records {
            // First input and the target both carry the record index.
            file.write_all(&(r as f32).to_le_bytes()).unwrap();
            for _ in 1..inputs {
                file.write_all(&0.5f32.to_le_bytes()).unwrap();
            }
            file.write_all(&(-(r as f32)).to_le_bytes()).unwrap();
        }
        file.flush().unwrap();

        let plan = SamplePlan::new(dir.path(), config, None).unwrap();
        assert_eq!(plan.total_records(), records as u64);
        let mut seen = Vec::new();
        for chunk in plan.chunks(per_batch as u64 + 1) {
            let mut reader = plan.reader(&chunk);
            let mut record = TrainingRecord {
                inputs: Vec::new(),
                outputs: Vec::new(),
            };
            while reader.next_record_into(&mut record).unwrap() {
                assert_eq!(record.inputs.len(), inputs, "record width");
                assert_eq!(record.outputs.len(), 1, "target width");
                assert_eq!(record.inputs[1], 0.5, "record body must decode");
                assert_eq!(record.outputs[0], -record.inputs[0], "target must pair up");
                seen.push(record.inputs[0]);
            }
        }
        let expected: Vec<f32> = (0..records).map(|r| r as f32).collect();
        assert_eq!(seen, expected, "every record exactly once, in order");
    }

    #[test]
    fn a_record_wider_than_the_read_batch_still_reads() {
        // The batch is sized in whole records, so a record wider than the
        // target batch must still be read one at a time rather than never.
        let inputs = READ_BATCH_BYTES / std::mem::size_of::<f32>();
        let config = TrainingDataConfig::new(inputs, 1);
        assert!(config.bytes_per_record() > READ_BATCH_BYTES);

        let dir = tempdir().unwrap();
        let mut file =
            std::io::BufWriter::new(std::fs::File::create(dir.path().join("0.bin")).unwrap());
        for r in 0..3 {
            file.write_all(&(r as f32).to_le_bytes()).unwrap();
            for _ in 1..inputs {
                file.write_all(&0.25f32.to_le_bytes()).unwrap();
            }
            file.write_all(&1.5f32.to_le_bytes()).unwrap();
        }
        file.flush().unwrap();

        let plan = SamplePlan::new(dir.path(), config, None).unwrap();
        let mut reader = plan.reader(&plan.chunks(u64::MAX)[0]);
        let mut record = TrainingRecord {
            inputs: Vec::new(),
            outputs: Vec::new(),
        };
        let mut seen = Vec::new();
        while reader.next_record_into(&mut record).unwrap() {
            assert_eq!(record.outputs, vec![1.5]);
            seen.push(record.inputs[0]);
        }
        assert_eq!(seen, vec![0.0, 1.0, 2.0]);
    }

    #[test]
    fn map_chunks_returns_results_in_chunk_order_at_every_thread_count() {
        let dir = write_sample(&[64]);
        let plan = plan(&dir, None);
        let chunks = plan.chunks(4);
        assert!(chunks.len() > 1);
        let expected: Vec<u64> = chunks.iter().map(|c| c.start).collect();
        for threads in [1, 2, 8] {
            let got = map_chunks(threads, &chunks, || Ok(()), |(), chunk| Ok(chunk.start)).unwrap();
            assert_eq!(
                got, expected,
                "results must be ordered by chunk at {threads}"
            );
        }
    }

    #[test]
    fn map_chunks_surfaces_a_worker_failure() {
        let dir = write_sample(&[16]);
        let chunks = plan(&dir, None).chunks(4);
        for threads in [1, 4] {
            let err = map_chunks(
                threads,
                &chunks,
                || Ok(()),
                |(), chunk| {
                    if chunk.index == 2 {
                        Err("chunk 2 exploded".to_string())
                    } else {
                        Ok(chunk.index)
                    }
                },
            )
            .expect_err("a failing chunk must fail the scan");
            assert!(err.contains("chunk 2 exploded"), "{err}");
        }
    }

    #[test]
    fn map_chunks_surfaces_a_worker_setup_failure() {
        let dir = write_sample(&[16]);
        let chunks = plan(&dir, None).chunks(4);
        let err = map_chunks(
            4,
            &chunks,
            || Err::<(), String>("no network".to_string()),
            |(), chunk| Ok(chunk.index),
        )
        .expect_err("a worker that cannot start must fail the scan");
        assert!(err.contains("no network"), "{err}");
    }

    #[test]
    fn map_chunks_rejects_a_zero_worker_count() {
        let err = map_chunks(0, &[], || Ok(()), |(), _| Ok(0u64))
            .expect_err("zero workers must not be read as serial");
        assert!(err.contains("at least 1"), "{err}");
    }
}
