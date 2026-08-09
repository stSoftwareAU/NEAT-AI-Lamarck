# Lamarck baseline economics

Issue [#8](https://github.com/stSoftwareAU/NEAT-AI-Lamarck/issues/8) — strategy value and runtime economics on a production-scale GRQ creature within the **45-minute** budget.

## Environment

| Item | Value |
|------|--------|
| Host | Apple M4, arm64, macOS 26.6 |
| Creature | `../GRQ-cluster/network.json` (~2511 inputs, ~1590 hidden) |
| Training | private copy of GRQ `trainData-binary_116` (~21 GiB, 2 262 277 records) under `.lamarck-baseline-45/train-data` (avoids mid-run deletion by GRQ `disk_guard` / `node.sh`) |
| Scorer | `../NEAT-AI-scorer/target/release/rust_scorer` (CPU directory mode; GPU deep-scratch auto-fallback) |
| Binary | `neat_ai_lamarck` release from Develop after #28 (improvement-weighted focus) |
| Journal | local `.lamarck-baseline-45/experiments.jsonl` (gitignored) |

## Config (production budget)

| Knob | Baseline run | Notes |
|------|--------------|--------|
| `--timeout-seconds` | **2700** | production default |
| `--candidates` | **100** | production default |
| `--quick` / sample | **25000** | worker-realistic analysis sample; acceptance still full-corpus |
| `--screen-sample-rate` | 0.05 | production default |
| `--screen-promote-threshold` | 1e-6 | production default |
| `--focus-policy` | **weighted** | residual MAE / `|blame|`; zeros excluded |
| `--seed` | 1 | |
| Phase-0 parity | **on** (passed) | |

Command shape:

```bash
# Prefer a private train-data copy (see .run-baseline-economics.sh).
neat_ai_lamarck \
  ../GRQ-cluster/network.json \
  .lamarck-baseline-45/train-data \
  --timeout-seconds 2700 --candidates 100 --seed 1 \
  --scorer ../NEAT-AI-scorer/target/release/rust_scorer \
  --output-dir .lamarck-baseline-45 --preserve-losers \
  --focus-policy weighted \
  --quick --quick-sample-records 25000 --screen-sample-rate 0.05
```

Report (reproducible tooling for #8):

```bash
neat_ai_lamarck report .lamarck-baseline-45/experiments.jsonl
# or: scripts/report-experiments.sh .lamarck-baseline-45/experiments.jsonl
```

## Phase-0 parity

| Quantity | Value |
|----------|--------|
| Scorer score | `0.344965267183` |
| Scorer error | `0.654657516817` |
| Complexity penalty | `0.000377216000` |
| Lamarck local MSE | `0.654657514416` (2 262 277 records) |
| Result | **pass** |

## Headline metrics (45 min)

From the end-of-run summary / journal:

| Metric | Value |
|--------|--------|
| Experiments | **75** |
| Acceptances | **2** |
| Scorer failures | 0 |
| Opening baseline (Phase-0) | ≈ `0.344965` |
| Best after run | ≈ `0.344967` |
| Cumulative accepted Δ | ≈ `+2.11e-6` (two clears of `1e-6`) |
| Time to first accept | ≈ 210 s |
| Analysis / (analysis+scorer) | ≈ **17%** |
| Scorer share | ≈ **83%** |
| Promote candidates / scorer-minute | ≈ **3.1** (115 promote-scored) |
| Screen-empty experiments | **49 / 75** (no full-corpus promote) |
| Projected batches / 45 min | ≈ **75** (matches observed) |
| Wall | ≈ 2700 s budget exhausted |

Analysis (learning + residual scan + focus stats + generate) stays a minority of wall time. Scorer directory calls dominate.

## Strategy value

Appearances vs wins (journal `strategies` + winner→candidate mapping):

| Strategy | Appearances | Wins | Acceptance rate |
|----------|-------------|------|-----------------|
| `random` | 225 | **2** | 0.89% |
| `backprop` | 225 | 0 | 0% |
| `stats_bias` | 225 | 0 | 0% |
| `stats_weight` | 225 | 0 | 0% |
| `structural_weaken` | 225 | 0 | 0% |
| `structural_add_neuron` | 228 | 0 | 0% |
| `structural_add` | 600 | 0 | 0% |
| `mean_error_bias` | 1 | 0 | 0% |

Accepted winners (both on high-blame focus `neuron-1343748843`):

1. Exp 3 — `random` bias nudge, Δ ≈ `+1.11e-6`
2. Exp 5 — `random` weight nudge, Δ ≈ `+1.00e-6`

### Interpretation

- **Do not handicap or disable `random`.** It produced **both** verified accepts in this production-budget sample.
- **`backprop` is generating** (225 appearances) under blame-weighted focus, but did not clear the accept bar here. Not dominated — keep generating; tune step size / capping next.
- **Structural + stats** families appear often (especially `structural_add`) but won **0** times in 75 experiments. With only two accepts total, that is **not** enough evidence to disable them; they remain exploratory.
- **`mean_error_bias` almost never fires** (1 appearance): weighted focus preferred high-`|blame|` **hidden** neurons (no output residual). Use an `output-0` / `high-error` slice when measuring that strategy specifically.
- Focus history: `neuron-1343748843` ×19 with **2 accepts**; several other high-signal hiddens ×1–28 with 0 accepts — history boost concentrated work on the productive focus after the first win.

**No strategy is disabled from this baseline.** Random is the clear value leader; others stay for exploration until a larger multi-run sample shows zero contribution.

## Economics conclusions

1. Within 45 minutes at `--candidates 100` + 5% screen, expect ~75 experiments and a thin accept rate (~2–3%).
2. Scorer time still dominates (~83%). Keeping full focused analysis + creature learning before focus select is affordable.
3. Screen empties most batches (49/75) — expensive full-corpus promote is already avoided when sample Δ ≤ `1e-6`. Raising candidates further would mainly inflate screen cost unless promote rate improves.
4. Improvement-weighted focus successfully avoids zero-signal neurons and surfaces real `backprop` candidates on blamed hiddens; accepts still came from `random` on that same focus.

## Recommended next experiments

1. **Output-focus slice:** `--focus-policy high-error` or `--focus-neuron output-0` for ≥15 experiments to measure `mean_error_bias` / output-residual economics.
2. **Backprop step A/B:** smaller learning-rate / plank on hidden focus — backprop appeared often but never accepted.
3. **Batch-size A/B under fixed 15 min:** 40 vs 100 vs 150 candidates; compare accepts and promote/scorer-minute (no silent default change).
4. **Multi-seed repeat** of this 45-minute config (seeds 2–5) before deprioritising any non-random strategy.
5. **Operational:** keep a private train-data copy (or hold `.in-use.lock`) when running beside GRQ `node.sh`, which can delete `.trainData-binary_*` mid-run.

## Interim medium run (superseded)

An earlier ~15-minute / 40-candidate interim (28 exps, 7 accepts; `structural_add` / `stats_bias` led) is superseded by this 45-minute production-budget baseline. Tooling from that work remains: expanded `report` fields and `scripts/report-experiments.sh`.

## Report fields (tooling)

`neat_ai_lamarck report` includes:

- `strategies[].appearancesTotal` / `wins` / `acceptanceRate`
- `focusHistory[]` (experiments, accepts, cumulative Δ)
- `improvementSeries[]`
- `candidatesPerScorerMinute`, `candidatesPerScreenMinute`, optional `candidatesPerWallMinute`
- `projectedBatchesPer45Min`, `relativeScoreImprovement`, `analysisTimeFraction`
