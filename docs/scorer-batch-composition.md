# Scorer batch composition and same-call deltas (issue #130)

`rust_scorer` reports a creature's full-corpus score as a function of **which
other creatures shared the directory call**, not of the creature alone. Lamarck
cannot repair that measurement, but it can refuse to subtract two scores that
were never measured together. This document records the artefact, the rule it
forces on Lamarck, and what still has to happen upstream.

## The artefact

Measured on the production creature during the issue #98 economics arms: same
`baseline.json` (sha256 `5685b363…`), same 21 GiB `trainData-binary_116` corpus,
same binary, CPU directory mode, `activationThreads: 10`.

| Directory contents | `baseline` score |
| --- | --- |
| `baseline` alone | `0.347854391736837` |
| `baseline` + `candidate-010` | `0.347854330680259` |
| `baseline` + `candidate-010` + `candidate-028` | `0.347854338215866` |

Every composition is bit-for-bit repeatable across repeated calls, so this is a
deterministic function of the batch size, not run-to-run noise. Alone versus the
pair the incumbent moves `6.106e-8` absolute — `1.755e-7` relative.
`candidate-010` moves `6.7e-8` the same way when a third creature joins the
call, and it does **not** move by the same amount as the incumbent.

Mechanism, in the scorer: `multi_score::workers_per_creature_split` sets
`target = max(activation_threads, n_creatures) * split`, so with 10 threads a
creature's chunk is cut into 10 record sub-ranges when it is scored alone, 5
beside one other creature, and 4/3/3 for three. Each creature's f64 partial sums
are grouped differently, and a different 8-record / 4-record / scalar SIMD path
is selected in the upstream loss kernels, purely because of how many *other*
creatures are in the call.

Why it matters here: Lamarck accepts at `--min-improvement 1e-6` and the
full-corpus deltas in [`docs/followup-economics.md`](followup-economics.md) live
at `1e-7`–`1e-6` (`+8.39e-7`, `+3.90e-7`, `-2.21e-7`). A `6.7e-8` artefact is a
significant fraction of that signal, and because it moves the incumbent and the
candidate by *different* amounts it perturbs the very Δ an accept is decided on.

## The rule: a Δ is only valid inside one call

Every Δ Lamarck decides on is formed from two scores produced by the **same**
`rust_scorer` invocation. Each phase already scores the incumbent in its own
call, so the rule costs nothing extra — it only forbids reaching across calls
for a baseline.

```mermaid
flowchart TD
    SCREEN["screen call<br/>baseline + candidates"] --> SD["screen Δ<br/>(gates promotion only)"]
    PROMOTE["promote call<br/>baseline + promoted candidates"] --> PD["single Δ<br/>= candidate − promote baseline"]
    COMBO["combo call<br/>baseline + combo creatures"] --> CD["combo Δ<br/>= combo − combo baseline"]
    VERIFY["verify call<br/>baseline + winner"] --> VD["verified Δ"]
    PD --> GATE{"Δ > --min-improvement?"}
    CD --> GATE
    VD --> GATE
    GATE -- yes --> WIN(["new incumbent"])
    GATE -- no --> KEEP(["keep incumbent"])
    CROSS["combo score − promote baseline"] -. "forbidden: different batch<br/>compositions, not comparable" .-> GATE

    classDef call fill:#cffafe,stroke:#0e7490,stroke-width:2px,color:#083344
    classDef delta fill:#ede9fe,stroke:#6d28d9,stroke-width:2px,color:#2e1065
    classDef win fill:#dcfce7,stroke:#15803d,stroke-width:2px,color:#052e16
    classDef reject fill:#fee2e2,stroke:#b91c1c,stroke-width:2px,color:#450a0a

    class SCREEN,PROMOTE,COMBO,VERIFY call
    class SD,PD,CD,VD delta
    class WIN win
    class KEEP,CROSS reject
```

What changed under issue #130, in `lamarck/src/combos.rs`:

- The combo call already wrote and scored `baseline.json`
  (`select_best_with_combinations`) and then threw that score away. It is now
  the number every combo is judged against — both for `--min-improvement` and
  for the comparison against the best improving single, which is now a Δ-versus-Δ
  comparison rather than one raw score against another call's raw score.
- A combo call that returns no `baseline` stem is a loud error, never a licence
  to fall back to the promote call's number.
- `ComboSelection::delta` is documented as a same-call Δ and
  `ComboSelection::accepts` is the accept gate, so `lamarck/src/run.rs` reads
  that Δ instead of re-subtracting the promote baseline from a combo winner
  scored elsewhere.

Unchanged: the screen gate and the single-candidate Δ were already same-call
subtractions, and `verify_accept_pair` was written for exactly this reason —
"same binary, same corpus, same call".

## What this does not fix

The artefact itself is upstream, in
[NEAT-AI-scorer](https://github.com/stSoftwareAU/NEAT-AI-scorer)
(`rust_scorer/src/multi_score.rs`), whose module documentation bounds this
re-association at "~1e-9 relative in practice, never more" — the measured
movement is about **175x** that bound, and batch size is an undocumented second
driver of it (`activation_threads` is a third: the same creature scores
differently on hosts with different worker defaults).

That fix is **written and pushed**, on the scorer branch
[`issue-lamarck-130-batch-invariant-partition`](https://github.com/stSoftwareAU/NEAT-AI-scorer/tree/issue-lamarck-130-batch-invariant-partition):
every chunk is cut into fixed 64-record blocks (`RECORDS_PER_PARTITION`), each
creature's per-block f64 sums fold back in block order, and a creature's workers
take a contiguous span of blocks — so the batch, `activation_threads` and
`NEAT_SCORER_WORKER_SPLIT` are all invisible in the scores, bit-identically. Its
`./quality.sh` passes, and directory scoring is unchanged at N=1/10 creatures and
~19–21% faster at N=50/200.

Opening the PR for that branch, and releasing the scorer afterwards, are human
decisions — the run allowlist refuses issue and PR creation against
`stSoftwareAU/NEAT-AI-scorer` from this repository's runs. Issue #143 tracks
that hand-off and what is owed once it lands (re-measuring the #98 economics
deltas, and revisiting the baseline drift epsilon that was priced against noise
the fix removes).

Until the released scorer arrives, the same-call rule above is what keeps the
artefact out of Lamarck's accept decisions — and it stays correct afterwards,
because a Δ formed inside one call is the right subtraction either way.
