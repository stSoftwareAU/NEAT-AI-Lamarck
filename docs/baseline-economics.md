# Lamarck baseline economics (interim)

Issue [#8](https://github.com/stSoftwareAU/NEAT-AI-Lamarck/issues/8) — strategy value and runtime economics on a production-scale GRQ creature.

This is an **interim** medium-run baseline (≈15 minutes, 40 candidates). It is **not** a full production default run (`--timeout-seconds 2700 --candidates 100`). Defaults are unchanged.

## Environment

| Item | Value |
|------|--------|
| Host | Apple M4, 10 cores, arm64, macOS 26.6 |
| Creature | `../GRQ-cluster/network.json` (~2511 inputs, ~1590 hidden) |
| Training | `../GRQ/.trainData-binary_116` (~21 GiB, 2 262 405 records) |
| Scorer | `../NEAT-AI-scorer/target/release/rust_scorer` (CPU directory mode; GPU deep-scratch auto-fallback) |
| Binary | `neat_ai_lamarck` release built from `feature/focus-blame-and-economics` (atop Develop `ac02f69`) |
| Journal | local `.lamarck-medium-run/experiments.jsonl` (gitignored) |

## Config

| Knob | Medium run | Production default |
|------|------------|--------------------|
| `--timeout-seconds` | **900** | 2700 (45 min) |
| `--candidates` | **40** | 100 |
| `--quick` / sample | **25000** | full observations (no `--quick`) |
| `--screen-sample-rate` | 0.05 | 0.05 |
| `--screen-promote-threshold` | 1e-6 | 1e-6 |
| `--focus-policy` | weighted (then high-error) | **weighted** by residual/blame (zeros excluded) |
| Phase-0 parity | **on** (passed) | on |

Command shape:

```bash
neat_ai_lamarck \
  ../GRQ-cluster/network.json \
  ../GRQ/.trainData-binary_116 \
  --timeout-seconds 900 --candidates 40 --seed 1 \
  --scorer ../NEAT-AI-scorer/target/release/rust_scorer \
  --output-dir .lamarck-medium-run --preserve-losers \
  --quick --quick-sample-records 25000 --screen-sample-rate 0.05
```

Report:

```bash
neat_ai_lamarck report .lamarck-medium-run/experiments.jsonl
# or: scripts/report-experiments.sh .lamarck-medium-run/experiments.jsonl
```

## Phase-0 parity

Full-corpus scorer baseline agreed with Lamarck packed MSE within documented epsilon (`lamarck/src/parity.rs`):

| Quantity | Value |
|----------|--------|
| Scorer score | `0.345396116960` |
| Scorer error | `0.654226667040` |
| Complexity penalty | `0.000377216000` |
| Lamarck local MSE | `0.654226642777` (2 262 405 records) |
| Result | **pass** |

Phase-0 local MSE over the full GRQ corpus completed in tens of seconds (packed `mse_sum_batch_packed` chunks).

## Headline metrics

From `neat_ai_lamarck report` on the medium journal:

| Metric | Value |
|--------|--------|
| Experiments | 28 |
| Acceptances | 7 |
| Scorer failures | 0 |
| Opening baseline | ≈ `0.345396` |
| Cumulative accepted Δ | ≈ `+1.27e-5` |
| Relative improvement | ≈ `+3.68e-5` |
| Time to first accept | ≈ 99 s (analysis+scorer sum) |
| Analysis / (analysis+scorer) | ≈ **13%** |
| Promote candidates / scorer-minute | ≈ **4.7** (62 promote-scored) |
| Projected batches / 45 min | ≈ **84** (from mean experiment duration) |
| Wall | ≈ 900 s budget exhausted |

Analysis remains a minority of wall time; the scorer directory calls dominate.

## Strategy value

Appearances vs wins (all candidate strategies observed in this journal):

| Strategy | Appearances | Wins | Acceptance rate |
|----------|-------------|------|-----------------|
| `structural_add` | 224 | 3 | 1.3% |
| `stats_bias` | 84 | 2 | 2.4% |
| `random` | 84 | 1 | 1.2% |
| `structural_weaken` | 84 | 1 | 1.2% |
| `stats_weight` | 84 | 0 | 0% |
| `structural_add_neuron` | 84 | 0 | 0% |
| `backprop` | 0 | 0 | — |
| `mean_error_bias` | 0 | 0 | — |

### Interpretation (interim)

- **Promising:** `structural_add` and `stats_bias` produced most of the verified accepts. Keep them.
- **Still useful:** `random` and `structural_weaken` each cleared the `1e-6` bar once — do **not** handicap random.
- **No wins this run:** `stats_weight`, `structural_add_neuron` — not dominated with N this small; keep generating.
- **Missing this run:** `backprop` / `mean_error_bias` never appeared because weighted focus mostly selected **hidden** neurons (no output residual → mean-error path skipped; many focuses reported `blame_count=0` for the focus bias even while the creature-wide learning signal was large). Default focus is now `high-error` (output head); re-run with that default to measure backprop / mean-error economics.

**Do not disable any strategy from this interim sample alone.**

## Focus history

Weighted focus spread across many hidden UUIDs (one experiment each, plus one neuron seen twice). Seven distinct focuses produced an acceptance; most focuses were clean misses (failure history now visible in `focusHistory` of the report).

## Economics conclusions

1. Scorer batch time still dominates (~87% of analysis+scorer). Analysis (focus scan + backprop accumulate + candidate gen) is cheap enough to keep the full focused statistics scan.
2. With 40 candidates + 5% screen, many experiments end in **screen-empty** (no full promote). Raising candidates toward 100 (production default) remains economically sensible if screen keeps promoting a thin slice.
3. Projected ~84 experiments per 45 minutes at this medium pace is an upper bound under `--quick`; full observations + larger batches will be slower.

## Recommended next experiments

1. **Full production run:** `--timeout-seconds 2700 --candidates 100` without `--quick`, same creature/data; refresh this doc with that journal.
2. **Output-focus slice:** pin `output-0` / `high-error` for ≥10 experiments to populate `backprop` / `mean_error_bias` rates.
3. **Batch-size A/B:** 20 vs 40 vs 100 candidates under a fixed 15-minute budget; compare accepts and promote/scorer-minute (no silent default change).
4. **Blame-aware focus:** bias weighted focus toward neurons with nonzero `LearningSignal` blame once #4 surfaces are stable in production journals.

## Reproduce report fields

The report JSON now includes (issue #8 gaps closed in tooling):

- `strategies[].appearancesTotal` / `acceptanceRate` (wins no longer cloned into appearances)
- `focusHistory[]` (experiments, accepts, cumulative Δ)
- `improvementSeries[]`
- `candidatesPerScreenMinute`, optional `candidatesPerWallMinute`
- `projectedBatchesPer45Min`, `relativeScoreImprovement`
