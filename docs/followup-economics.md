# Lamarck follow-up economics (issue #75)

The four experiments [`docs/baseline-economics.md`](baseline-economics.md)
recommended and issue [#8](https://github.com/stSoftwareAU/NEAT-AI-Lamarck/issues/8)
never ran. The #8 strategy table was a single 45-minute, single-seed sample
(75 experiments, 2 accepts, both `random`); this campaign adds the arms needed
before any strategy is judged.

Reproduce with:

```bash
cargo build --release
CREATURE=../GRQ-cluster/network.json \
SCORER=../NEAT-AI-scorer/target/release/rust_scorer \
scripts/run-followup-economics.sh                     # or one arm at a time
scripts/summarise-followup-economics.sh .lamarck-followup
```

```mermaid
flowchart TD
    B["#8 baseline<br/>1 seed, 45 min, 2 accepts"] --> A1["Arm 1: output-focus slice<br/>--focus-policy high-error"]
    B --> A2["Arm 2: backprop step A/B<br/>--backprop-learning-rate 0.01 vs 0.001"]
    B --> A3["Arm 3: batch-size A/B<br/>40 / 100 / 150 candidates, fixed wall"]
    B --> A4["Arm 4: multi-seed repeat<br/>seeds 2-5, production config"]
    A1 --> V["Verdict: keep or disable each strategy"]
    A2 --> V
    A3 --> V
    A4 --> V
```

## Environment

| Item | Value |
|------|--------|
| Host | Apple M4, 10 cores, 24 GiB, macOS 26.6 |
| Creature | `../GRQ-cluster/network.json` (2511 inputs, 1600 neurons) |
| Training | private copy of GRQ `.trainData-binary_116` (21 GiB, 2 262 277 records) under `.lamarck-followup/train-data`, per the #8 operational note |
| Scorer | `../NEAT-AI-scorer/target/release/rust_scorer` (CPU directory mode) |
| Binary | `neat_ai_lamarck` 0.1.9 (post-#83 backprop blame routing, post-#74 combo attribution) |
| Opening score | `0.346759634929` (Phase-0 parity passed in every arm) |
| Shared knobs | `--quick --quick-sample-records 25000 --screen-sample-rate 0.05` |

**Load caveat — read every per-minute figure against it.** Unlike the #8
baseline, this campaign shared the box with a live GRQ `Learn.ts` run: 1-minute
load averages of 12–18 on 10 cores throughout. Absolute throughput here is
therefore *lower* than #8's (~37–65 s/experiment against #8's ~36 s), and
arm-to-arm comparisons are only fair because every arm ran sequentially under
the same background load. Each arm records its own `loadBefore` / `loadAfter` in
`timing.txt`.

The creature has also moved on since #8 — GRQ evolution lifted the opening score
from `0.344965` to `0.346760`, so accept rates here are measured against a
harder incumbent.
