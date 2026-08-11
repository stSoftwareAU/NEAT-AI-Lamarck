# Failed-candidate cache economics

Issue [#94](https://github.com/stSoftwareAU/NEAT-AI-Lamarck/issues/94) / parent [#69](https://github.com/stSoftwareAU/NEAT-AI-Lamarck/issues/69) — paired production benchmark that decides whether `--failed-cache` ships (and whether it flips to on-by-default).

## Status

**Unmeasured.** The runner and report tooling are wired; the exclusive-box production runs have not been executed. Until they are, `--failed-cache` stays **off by default** and the go/no-go decision is open.

## Method

Follow [`baseline-economics.md`](baseline-economics.md) production knobs:

| Knob | Value |
|------|--------|
| `--timeout-seconds` | 2700 |
| `--candidates` | 100 |
| `--screen-sample-rate` | 0.05 |
| `--screen-promote-threshold` | 1e-6 |
| `--focus-policy` | weighted |
| Phase-0 parity | **on** (do not pass `--skip-phase0`) |
| Host / creature / corpus | same as the #8 baseline |

### Arms

| Arm | Flag | Notes |
|-----|------|--------|
| `control` | cache off | Today's behaviour |
| `treatment` | `--failed-cache` | Warm snapshot allowed across the arm |
| `cold-start` | `--failed-cache`, no snapshot | Rebuild from the control journal of the same seed so startup cost is measured |

Repeats: `SEEDS="1 2 3"` (override as needed). Pairing controls the **start** of the RNG stream (#71); backfill draws make the streams diverge — that is expected and must be stated in any write-up.

`#83` backprop gating must be in a **fixed** state across both arms of every pair (default knobs unless the campaign pins `--backprop-learning-rate` / `--backprop-max-bias-adjustment-scale`).

### Gate metric

Primary: **`scoreImprovementPerWallHour`** from `neat_ai_lamarck report` — full-corpus anchored (#84). A sampled baseline must never appear as the anchor.

Secondary: accepts / scorer-minute, promote-scores / scorer-minute, cache hit rate, ms saved vs spent, peak memory / disk, whether the #92 stand-down guardrail fired.

### Validity checks (before any comparison)

1. Journalled effective seed matches across control/treatment for each pair.
2. Zero-accept arms are underpowered — report per-run accept counts; do not hide them in a mean.
3. A treatment arm where `stoodDownAtExperiment` is set is a partially-disabled cache — flag it in the per-run table.
4. Build/version stamp mismatch between arms voids the pair.

## How to run

```bash
cargo build --release -p neat_ai_lamarck
# Private train-data copy recommended (see .run-baseline-economics.sh).
SEEDS="1 2 3" ARM_SECONDS=2700 \
  scripts/run-failed-cache-economics.sh

scripts/summarise-failed-cache-economics.sh .lamarck-failed-cache
```

## Results

_Fill in after the exclusive-box campaign. Per-run table first; then means._

| Arm | Experiments | Accepts | Δ score | Δ / wall-hour | Hit rate | Saved ms | Spent ms | Net ms | Stood down | Peak entries |
|-----|-------------|---------|---------|---------------|----------|----------|----------|--------|------------|--------------|
| _(unrun)_ | | | | | | | | | | |

## Decision

_Open until the table above is populated._

- **Improves** improvement-per-hour without breaching the memory/disk ceiling → keep; decide whether `--failed-cache` flips to on-by-default; settle tolerance / size / age defaults from the measured data.
- **No measurable gain** → negative result (not a merge failure). Open a follow-up to remove or permanently default-off the feature; keep the numbers so the question is not re-litigated from intuition.
