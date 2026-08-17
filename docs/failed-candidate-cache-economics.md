# Failed-candidate cache economics

Issue [#94](https://github.com/stSoftwareAU/NEAT-AI-Lamarck/issues/94) / parent [#69](https://github.com/stSoftwareAU/NEAT-AI-Lamarck/issues/69) — paired production benchmark that decides whether `--failed-cache` ships, and whether it flips to on-by-default.

## Status

**Measured. No-go on flipping `--failed-cache` to on-by-default.** The implementation stays in tree, **off by default**, as an opt-in. The primary gate (`scoreImprovementPerWallHour`) was underpowered on the largest practical pair: **zero accepts** on both live arms. Secondary evidence says the cache works (hit rate 16.4%, ≈156 s estimated redundant scoring avoided vs 0.7 s spend, footprint ≪ 25 MiB, guardrail silent), but that is not a default-on case until an exclusive-box 45-minute campaign actually records accepts.

Journals live under gitignored `.lamarck-failed-cache/` (not committed). This document is the retained record.

## Environment

Same host / creature / corpus as [`baseline-economics.md`](baseline-economics.md):

| Item | Value |
|------|--------|
| Host | Apple M4, arm64, macOS 26.6 |
| Creature | `../GRQ-cluster/network.json` (~2511 inputs) |
| Training | private copy of GRQ `trainData-binary_116` (~21 GiB, 2 262 277 records) under `.lamarck-baseline-45/train-data` |
| Scorer | `../NEAT-AI-scorer/target/release/rust_scorer` (CPU directory mode; GPU deep-scratch auto-fallback) |
| Binary | `neat_ai_lamarck` 0.1.20 release (`--failed-cache` re-wired onto the current run loop; screen/promote batches observed by the #92 ledger) |

## Method

Production knobs from the #8 baseline, except **wall budget** (see [Accept-rate sizing](#accept-rate-sizing)):

| Knob | Value |
|------|--------|
| `--timeout-seconds` | **600** (largest practical exclusive-box pair; method default is 2700) |
| `--candidates` | 100 |
| `--screen-sample-rate` | 0.05 |
| `--screen-promote-threshold` | 1e-6 |
| `--focus-policy` | weighted |
| Phase-0 parity | **on** (do not pass `--skip-phase0`) |
| `#83` backprop knobs | **default / unset** on every arm (`backpropLearningRate` and `backpropMaxBiasAdjustmentScale` both absent/`null` in the journal header) |

### Arms

| Arm | Flag | Notes |
|-----|------|--------|
| `control` | cache off | Today's behaviour |
| `treatment` | `--failed-cache` | Warm snapshot allowed across the arm |
| `cold-start` | `--failed-cache`, no snapshot | Rebuild from a long prior `experiments.jsonl` so startup cost is measured |

Repeats: one seed (`SEEDS=1`). Pairing controls the **start** of the RNG stream (#71); backfill draws make the streams diverge — that is expected. Three seeds × three arms × 2700 s is called out below as impractical on this box.

Reproduce:

```bash
cargo build --release -p neat_ai_lamarck
SEEDS=1 ARM_SECONDS=600 \
  COLD_START_JOURNAL=.lamarck-baseline-45/experiments.jsonl \
  bash scripts/run-failed-cache-economics.sh control treatment cold-start

scripts/summarise-failed-cache-economics.sh .lamarck-failed-cache
```

### Gate metric

Primary: **`scoreImprovementPerWallHour`** from `neat_ai_lamarck report` — full-corpus anchored (#84). A sampled baseline must never appear as the anchor. Opening scores below are Phase-0 / first full-corpus `scores.baseline`.

Secondary: accepts / scorer-minute, promote-scores / scorer-minute, cache hit rate, ms saved vs spent, peak memory / disk, whether the #92 stand-down guardrail fired.

## Accept-rate sizing

#94 requires the pair to be long enough to see accepts, or an explicit statement that the needed budget is impractical.

| Sample | Wall | Experiments | Accepts | Opening (full-corpus) |
|--------|------|-------------|---------|------------------------|
| #8 baseline | 45 min | 75 | **2** | ≈ 0.344965 |
| Local calibration `~/.lamarck-followup-75` | mixed | ~147 | **0** | (see `docs/followup-economics.md`) |
| This pair, current creature | 10 min × 2 live arms | 8 + 14 | **0 + 0** | 0.35175391064496747 |

On the current incumbent (opening ≈ 0.35175, fitter than the #8 creature), a 10-minute arm produced no accepts. The #8 rate was 2 accepts / 45 min; even that is too sparse for a 10-minute pair to distinguish arms. A decisive campaign at the documented production knobs is **3 seeds × 3 arms × 2700 s ≈ 6.75 h of exclusive box time**. That is not practical on this host (the scorer is shared with other work). The 600 s pair below is the largest practical run; the resulting uncertainty is: **the primary metric cannot tell the arms apart**.

## Validity

| Check | Result |
|-------|--------|
| Journalled seed | **1** / `supplied` on both live arms |
| Opening full-corpus score | control and treatment **match** at `0.35175391064496747` |
| Version stamp | both live arms `0.1.20` |
| `#83` state | `backpropLearningRate` / `backpropMaxBiasAdjustmentScale` **unset** on both live arms |
| Zero-accept arms | **both** live arms — underpowered; kept visible in the per-run table |
| Guardrail | `stoodDownAtExperiment` is `null` on every cache-on arm |
| Shared-box load | control started at load ≈4.9; treatment started at load ≈17. That biases **against** treatment on wall-clock throughput; treatment still completed more experiments |
| Cold-start journal | rebuilt from `.lamarck-baseline-45/experiments.jsonl` (#8 creature / era). Its experiment and accept counts **include that prior journal** and must not be read as this campaign's primary metrics |

A first 600 s pair (crate `0.1.19`) omitted `CacheEconomics::observe_screen` / `observe_promote` on the two-phase path, so `estimatedSavedMs` stayed 0 despite real hits. The table below is the re-run with those observations restored. Screen-batch logs from the unpriced treatment arm reconstructed ≈63 s of skipped screen time (194 hits × 327 ms mean screen-creature), which matches the priced estimator's order of magnitude once it is actually fed.

## Results

From `scripts/summarise-failed-cache-economics.sh .lamarck-failed-cache` plus the cache summary line. `Δ score` / `Δ / wall-hour` are `unavailable` when `report` has no accepts — that is the underpowered primary gate, not a missing field.

| Arm | Experiments | Accepts | Opening (full-corpus) | Δ score | Δ / wall-hour | Hit rate | Saved ms | Spent ms | Net ms | Peak entries | Stood down | Peak mem / disk |
|-----|-------------|---------|----------------------|---------|---------------|----------|----------|----------|--------|--------------|------------|-----------------|
| `control-seed1` | 8 | 0 | 0.35175391064496747 | unavailable | unavailable | — | — | — | — | — | — | — |
| `treatment-seed1` | 14 | 0 | 0.35175391064496747 | unavailable | unavailable | 0.1640 | 156445 | 708 | +155737 | 1392 | no | 712 704 B / 469 245 B |
| `cold-start-seed1` | 88 (75 prior + this arm) | 2 (prior journal) | 0.34496525496702934 (#8) | 2.12e-6 | 1.12e-8 | 0.1538 | 0 (unpriced ledger) | 629 | −629 | 1293 | no | 662 016 B / 435 667 B |

Ceiling is `--failed-cache-max-bytes` 25 600 000 B. Neither cache-on arm approached it (`ceilingBites=0`).

End-of-run treatment summary (parseable log line; `savedMs` here is the live ledger, a few hundred ms above the report's sum of per-experiment `cacheSavedMs`):

```text
● failed-cache economics: entries=1392 hitRate=0.1640 savedMs=158854.5 wallClockSavedMs=47768.9 spentMs=709.6 netMs=158144.9 peakMemoryBytes=712704 diskBytes=469245 standDown=false ceilingBites=0
```

`savedMs` is redundant scoring avoided, not wall clock removed. The batch is backfilled, so most hits become fresh candidates; `wallClockSavedMs` (≈48 s) is the part that actually shortened scorer work.

### Secondary evidence (live pair)

| Quantity | Control | Treatment |
|----------|---------|-----------|
| Experiments | 8 | 14 |
| Accepts | 0 | 0 |
| Wall (journal) | 570 s | 601 s |
| Screen candidates scored | 800 | 1392 |
| Promote candidates scored | 33 | 30 |
| Cache hit rate | — | 0.1640 (273 / 1665) |
| Backfilled | — | 103 |
| Peak entries | — | 1392 |
| Peak resident / disk | — | 712 704 B / 469 245 B |
| Guardrail | — | did not fire |
| Estimated saved / spent / net | — | 156445 ms / 708 ms / **+155737 ms** |
| Rebuild (cold-start from 75-exp #8 journal) | — | **1 ms** |

Treatment completed **more experiments** in the same wall budget (14 vs 8). Promote-scored creatures were similar (30 vs 33) — the streams have already diverged via backfill, so this is not a clean “saved promotes” comparison.

## Decision

**No-go on default-on. Keep `--failed-cache` opt-in (off by default).**

| Question | Answer |
|----------|--------|
| Does it ship? | **Yes, opt-in.** The filter, snapshot, rebuild, ledger, and stand-down guardrail are in the binary. |
| Flip `--failed-cache` to on-by-default? | **No.** The primary metric is unmeasured (zero accepts). #69 says a negative / unreadable result is not a merge of default-on. |
| Remove the feature? | **No.** Hits, footprint, rebuild cost, and the priced ledger all look healthy; removal would throw away a working opt-in on the basis of an underpowered primary gate. |
| Tolerance / size / age defaults | **Leave the provisional values** (`50000` entries, 7-day age, 25 MiB ceiling, stand-down window 20 / margin 1000 ms). Nothing in this campaign suggested they were wrong; nothing measured them tightly enough to retune. |

Follow-up: exclusive-box 45-minute repeats (seeds 1–3, control + treatment, same knobs) before reconsidering default-on — [#158](https://github.com/stSoftwareAU/NEAT-AI-Lamarck/issues/158), not a removal issue.

## Ledger pricing (implementation note)

`savedMs = skipped × mean_screen_ms_per_creature`, with promote time claimed only for skips whose cache entry had reached promote (#92). Spend is measured lookup + maintenance + rebuild. The two-phase run loop must call `observe_screen` after every screen batch and `observe_promote` after every promote batch; without those calls the estimator is identically zero and the stand-down guardrail cannot see savings.
