# Statistical analysis and evidence engine (Phase 10)

A small, dependency-free, deterministic statistics layer (`experionyx.stats`) plus a registry record
that makes each analysis reproducible from persisted evidence. It preserves raw observations, exposes
its assumptions, and never manufactures certainty.

## Vocabulary

Every quantity carries a status: `OBSERVED` (a recorded measurement), `DERIVED` (computed from
observations by a stated formula), `INCONCLUSIVE` (too little data for this statistic) or `UNDEFINED`
(the statistic does not exist for this input, with a reason). An invalid statistic is never turned
into a number: an empty sample, one observation, a zero variance or a zero denominator each produce
an explicit status. Non-finite input is rejected, never dropped or zeroed.

## What is computed

| Piece | Method |
|---|---|
| Summary | n, mean, median, sample variance (n-1), sd, standard error, min, max |
| Effect sizes | raw mean difference; relative change (`/ abs(mean reference)`); unpaired Cohen's d and Hedges' g (pooled sd); paired `d_z` (`mean(d)/sd(d)`). Each names its formula, numerator, denominator, assumptions and undefined condition. No composite score. |
| Confidence interval | seeded bootstrap of the mean or median of one sample, of paired per-key differences, or of `est(treatment) - est(reference)`: **percentile** and **BCa**. Records method, confidence, resamples, seed, estimator, sample counts, z0 and acceleration (BCa), warnings. |
| Test | paired **sign-flip permutation** (exact for n <= 15, else seeded Monte Carlo); unpaired **permutation test** of the mean difference (exact when the number of rearrangements <= 50,000, else seeded Monte Carlo). No distribution tables, no SciPy. |
| Proportion | Wilson score interval (assumes independent Bernoulli trials). |
| Multiple comparisons | `NONE`, `BONFERRONI` (family-wise error), `BENJAMINI_HOCHBERG` (false discovery rate; independent or positively dependent tests). The family is exactly the set of p-values you pass; hypotheses without a p-value are excluded and listed. |

BCa is implemented (Efron 1987): bias correction `z0` from the mid-rank share of resamples below the
estimate, acceleration from the jackknife (over both groups for two-sample statistics). It reports
`UNDEFINED` with the reason when `z0` is infinite (the estimate lies outside every resample), the
jackknife is constant, or the adjustment denominator is not positive; it never falls back to the
percentile interval silently.

## Pairing

`PAIRED`, `UNPAIRED`, `UNKNOWN`. Paired analysis needs keyed observations (a mapping key -> value with
identical key sets); position is never identity, and mismatched keys or plain lists are rejected.
`UNKNOWN` is analyzed as `UNPAIRED` and says so. Unpaired inputs are sorted first, so results do not
depend on their order.

## A p-value is not a claim

Test results carry a note that a p-value is about the stated null under stated assumptions, not about
practical importance or causation. The interval and the test answer different questions and can
disagree. Statistical significance is never rendered as causal language.

## Records and reproducibility

`StatisticalAnalysis` (`sta_`, schema v9, table `statistical_analyses`) is immutable and
content-addressed by kind, exact input digest, settings, sources and engine version. It stores the
settings, the result and the **sources**; raw observations already in digest-verified run artifacts
are referenced, not copied. Inline inputs (which exist nowhere else) are stored.

Source kinds: `inline`; `interaction` (trial values of an interaction analysis; the design's declared
pairing is used); `fault_trials` (per-seed baseline/faulted values of a fault experiment, paired by
seed); `artifact` (a mapping or list at a pointer inside any run artifact); `failure_mode` (runs with
the signal over runs analyzed, for a Wilson interval; the i.i.d. caveat is attached); `analyses`
(p-values of registered comparisons, for a correction).

`experionyx stats verify` re-reads the sources (artifacts are digest-verified, so silent changes are
refused), recomputes, and reports whether the input and result digests match. A claim can cite an
analysis as evidence (`EvidenceTarget.STATISTICAL_ANALYSIS`).

## Integration

- **Interactions:** the percentile bootstrap now also records `bootstrap_p`, the smallest two-sided
  level at which the percentile interval would exclude zero (with +1 smoothing). It is an *approximate
  inversion of the interval, not an exact test*. `InteractionConfig.multiplicity_correction`
  (`NONE` default, `BONFERRONI`, `BENJAMINI_HOCHBERG`) adds raw and adjusted values per effect and the
  correction record (method, family size, alpha) to the analysis; **labels are not changed**. The
  default `NONE` leaves existing spec IDs unchanged.
- **Faults, benchmarks, reliability profiles:** their raw measurements are untouched. `fault_trials`
  and `artifact` sources run the engine over them by reference.
- **Failures:** prevalence of a failure mode gets a Wilson interval from its recorded numerator and
  denominator. Qualitative modes get no test.

## CLI

```
experionyx stats compare   [--reference-values 1,2,3 --treatment-values 4,5,6 | --file F.json |
                            --interaction ian_ --measure M --reference-cell A --treatment-cell AB |
                            --fault-experiment fxp_ --measure M | --artifact RUN PATH --reference-pointer a/b ...]
                           [--pairing PAIRED|UNPAIRED|UNKNOWN] [--estimator mean|median] [--method percentile|bca]
                           [--confidence 0.95] [--resamples 2000] [--permutations 10000] [--seed 0]
experionyx stats bootstrap (same sources; one sample or a difference)
experionyx stats proportion (--successes K --trials N | --failure-mode fmd_)
experionyx stats correct   (--p NAME=P ... | --analysis sta_ ...) [--method BONFERRONI|BENJAMINI_HOCHBERG|NONE] [--alpha 0.05]
experionyx stats list|inspect|verify
```

## Limitations

- Bootstrap intervals assume exchangeable observations; they ignore dependence between them and say
  nothing about distribution shift. Small n gives unstable intervals (a warning is recorded).
- BCa in interaction analysis is not offered: the interaction statistic combines four cells and its
  jackknife is not defined here. Interactions use the percentile bootstrap only.
- `bootstrap_p` in interactions is an interval inversion, not a calibrated test.
- The permutation test tests exchangeability of labels; it is not a test of a mean alone when spreads
  differ. Effect sizes assume similar spread (stated in each effect).
- A single control run cannot be paired with treatment trials (no shared key) and gives no test; the
  engine reports this instead of inventing pairs.
- Benjamini-Hochberg assumes independent or positively dependent tests; the contrasts of one
  interaction analysis are usually correlated, so treat its adjusted values as indicative.
- Failure-mode prevalence treats the runs of a discovery as independent trials, which they are not
  (different faults, severities, seeds); it describes those runs, not a population rate.
- Analysis records are database entries (their payload is complete machine-readable JSON); they do not
  write a separate run artifact.

## Reuse by drift analysis (Phase 12)

The drift engine ([drift.md](drift.md)) reuses this core rather than adding its own inference: Wilson
intervals for class and category proportions, `effect_sizes` and `bootstrap_interval` for the mean
difference of a numeric feature, `compare` (unpaired) for the per-sample performance measure and
`adjust_pvalues` for the comparison families. Its only new procedure is a seeded permutation test of a
distribution distance (KS, Jensen-Shannon), which follows this module's conventions: exact when the
number of rearrangements is at most `EXACT_LIMIT`, otherwise `(hits + 1) / (permutations + 1)` with a
recorded seed.

## Reuse by data quality analysis (Phase 13)

The data-quality engine ([data-quality.md](data-quality.md)) reuses this core for every inference:
Wilson intervals for missing-value rates and class proportions, `compare` (unpaired) on 0/1 indicators
for missingness and validity rate differences between groups, and `adjust_pvalues` per comparison
family. Its own procedures are descriptive: Tukey/MAD outlier rules, population skewness and kurtosis,
Pearson correlation and value purity as leakage *indicators*. None of them is a test or a verdict.
