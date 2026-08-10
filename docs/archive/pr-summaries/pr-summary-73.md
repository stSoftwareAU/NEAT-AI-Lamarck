# Distribution moments in `observations.statistics`, with a consumer

## Summary

`observations.statistics` collected no distribution shape and stored no
covariances, both of which the README's Phase 1 spec had asked for. The issue
asked for a decision, not a blind addition — adding fields nothing consumes is
the mistake #22 cleared out. Both halves are now decided and acted on.
Closes #73.

**Skewness and excess kurtosis — added, with a consumer.** Every input and
target column now carries population `skewness` (`m3 / m2^1.5`) and
`excessKurtosis` (`m4 / m2² − 3`). They come free from the existing streaming
pass: the Welford accumulator gained third and fourth central moments via
Pébay's online update, whose `term1` is algebraically identical to the previous
`delta * (v - new_mean)`, so `variance` is bit-for-bit unchanged. A constant or
empty column reports `0` rather than `NaN`.

The consumer is a new `stats_skew_bias` candidate strategy. A squared-error fit
centres a prediction on the target **mean**; under a skewed target the mean sits
away from the typical value, so stepping the output bias towards the target's
median is a distinct, cheap hypothesis for the scorer to accept or reject. It
fires only for an output focus whose target has `|skewness| ≥ 0.25`, steps a
quarter of the mean→median gap, damps by excess kurtosis (heavy tails make the
sampled median gap unreliable — an excess kurtosis of 3 halves the step) and is
skipped for a saturated neuron. Hidden focuses produce nothing.

**Covariances — dropped from the specification, derived on demand.** A
covariance is exactly `r · σ_a · σ_b` from fields the cache already writes, so
storing it would duplicate data with nothing extra to say. Instead
`ObservationsStatistics::input_covariance` (input×input, `None` unless
`--compute-correlations` ran) and `input_target_covariance` apply that identity
to the stored correlation and standard deviations. Nothing new lands on disk for
this half.

`ALGORITHM_VERSION` is bumped to `1.1.0` because the on-disk shape changed. The
new fields are `#[serde(default)]`, so a pre-`1.1.0` cache still parses and is
then rejected **loudly** by the version gate with a precise
`unsupported algorithm_version` reason, rather than dying as an opaque parse
error.

```mermaid
flowchart LR
    SCAN[Streaming corpus scan] --> ACC["OnlineMoment<br/>m2 · m3 · m4"]
    ACC --> STATS["ScalarStats<br/>+ skewness<br/>+ excessKurtosis"]
    STATS --> CACHE[("observations.statistics<br/>algorithmVersion 1.1.0")]
    CACHE --> GEN["stats_skew_bias candidate<br/>bias → target median"]
    GEN --> SCORER[Scorer accepts or rejects]
    CACHE -. "r · σ · σ, never stored" .-> COV["input_covariance()<br/>input_target_covariance()"]
```

## Evidence

Backend/CLI change with no web interface, so there is no screenshot. The
evidence is the test suite and the quality gate, both run locally:

```text
$ ./quality.sh < /dev/null
...
Running tests...
test result: ok. 107 passed; 0 failed; ...   # lib unit tests
test result: ok.   4 passed; 0 failed; ...   # backprop_parity
test result: ok.  16 passed; 0 failed; ...   # readme_contract
...
All quality checks passed!
```

Moment values are asserted against hand-computed population moments rather than
whatever the code happens to emit — for the column `[0, 0, 0, 4]`, `m2 = 3`,
`m3 = 6` and `m4 = 21`, so skewness is `2/√3` and excess kurtosis is `−2/3`;
for the symmetric column `[-1, 0, 0, 1]` they are `0` and `−1`. The derived
covariance is checked against the covariance computed by hand from the same raw
records (`1.0` for that pair), which is the real check that the
`r · σ_a · σ_b` identity holds end to end.

## Test Plan

Added to `lamarck/src/observations.rs`:

- `skewness_and_kurtosis_are_collected` — exact population skewness/excess
  kurtosis for a right-skewed input and a symmetric target.
- `constant_and_empty_columns_report_zero_moments` — zero-variance column
  reports `0`, not `NaN`.
- `input_target_covariance_is_derived_from_correlation_and_spread` — matches the
  hand-computed covariance; out-of-range indices return `None`.
- `input_covariance_needs_the_input_matrix` — `None` without
  `--compute-correlations`; with it, symmetric in its arguments and the diagonal
  equals the variance.
- `a_cache_without_moments_is_stale_rather_than_corrupt` — a pre-#73 cache
  parses and is rejected with an `algorithm_version` reason.

Added to `lamarck/src/candidates.rs`:

- `skew_bias_steps_the_output_towards_the_target_median` — exact `−0.125` step
  for a right-skewed target.
- `skew_bias_follows_a_left_skewed_target_upwards` — direction follows the skew.
- `heavy_tails_damp_the_skew_bias_step` — excess kurtosis of 3 halves the step.
- `a_symmetric_target_produces_no_skew_bias_candidate`,
  `a_hidden_focus_produces_no_skew_bias_candidate`,
  `a_saturated_focus_produces_no_skew_bias_candidate` — the three gates.
- `skew_bias_is_offered_in_a_generated_population` — the strategy is reachable
  from `generate_candidates`, not just callable directly.

Added to `lamarck/tests/readme_contract.rs`:

- `phase_one_documents_the_distribution_moments`,
  `phase_one_documents_covariance_as_derived`,
  `candidate_table_documents_the_skew_bias_strategy`,
  `outstanding_work_no_longer_lists_the_observation_moments` — README and code
  cannot drift apart again.
- `subsection_returns_only_the_requested_subsection`,
  `subsection_panics_on_a_missing_heading` — cover the new `###` slicing helper.

No existing test was modified beyond adding the two new `ScalarStats` fields to
literal fixtures.
