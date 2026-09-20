# Temporal and distribution shift (Phase 12)

How do the distributions of inputs, targets, predictions and metrics differ between explicitly
declared windows of an evaluated dataset? The drift engine reports **observed differences** between
two sets of samples, each with its method, sample counts, uncertainty and an evidence status. It does
**not** produce a composite drift score, rank features, say that drift was "detected", say that a
distribution change caused a failure or a drop in performance, or call a change *concept drift*.

## Ordering and windows

Time is never inferred from row position. A spec must declare an **ordering**:

| Ordering | Meaning |
|---|---|
| `{"field": "index"}` | the dataset sample index *is declared* to be the temporal order (the author's assertion; EXPERIONYX cannot check it) |
| `{"field": "feature:<name or column>"}` | a numeric dataset feature (e.g. a timestamp as epoch seconds) orders the samples |
| `"unique": true` | refuse duplicate ordering keys. Ties are otherwise legitimate: membership depends on the key's *value*, never on position |

A `TemporalWindow` is an interval over that ordering with a role (`REFERENCE` or `COMPARISON`), finite
numeric `start`/`end` and explicit `start_inclusive` (default true) / `end_inclusive` (default false).
Its identity, `twn_…`, is the ordering, role, bounds and inclusivity (a `name` is only a label; `5.0`
and `5` are the same). The same interval used as a reference and as a comparison has two identities.

Window sources:

- **Explicit**: one `reference` and one or more `comparisons`.
- **Rolling**: `rolling: {start, stop, width, step, reference}` generates `[start + k·step, start + k·step + width)`.
  `reference` is `FIXED` (the spec's `reference` window), `PREVIOUS` (the preceding window; needs
  `step >= width`) or `EXPANDING` (everything from `start` to the window's start).

Every generated candidate is used or listed under `skipped` with a reason
(`PARTIAL_WINDOW`, `NO_REFERENCE`, `EMPTY_WINDOW`, `EMPTY_REFERENCE`); nothing is dropped silently, and
any skipped window makes the analysis `PARTIAL`.

**Refused before anything runs**: missing ordering, ambiguous or malformed ordering, non-finite or empty
bounds, overlapping reference/comparison windows (the tests need disjoint groups), duplicate windows,
unsupported feature types, invalid thresholds, a covariate dimension without features. Data-dependent
refusals (unknown feature, explicit window with no samples, duplicate keys when `unique`, missing
dataset, unevaluable slice, unknown failure mode, unfinished baseline) are checked by a *preflight*
before a run is created, so a refused request executes nothing. A sample whose ordering key is missing,
non-numeric or non-finite belongs to no window and is counted in `windows.json`.

## What is compared

For each window pair and each population (`POPULATION`, plus each explicitly requested slice):

| Dimension | Subject | Source |
|---|---|---|
| covariate | each declared feature (`NUMERIC`, `CATEGORICAL`, `BOOLEAN`) | dataset columns |
| label | the true target (categorical for classification, numeric for regression) | baseline predictions |
| prediction | the predicted value; the stored confidence as a numeric distribution when present | baseline predictions |
| performance | window metric values, per-metric direction, unpaired comparison of the per-sample measure | slice/evaluation metrics + Phase 10 |

Feature types are declared, never guessed. Only those three are supported; anything else is refused. A
value that does not fit the declared type is **invalid** (a string in a numeric feature, a `bool` as a
number) and counted, never coerced. `1` and `1.0` are one category; `1` and `"1"` are not.

## Methods (what they measure, assume, and where they fail)

| Method | Measures | Undefined / unavailable when |
|---|---|---|
| `ks_statistic` | largest gap between two empirical CDFs, 0–1 | a window has no valid values |
| `wasserstein_1` | mean distance mass moves, in feature units (`_scaled` divides by the reference standard deviation) | scaled: reference std is zero |
| `location` | mean difference, effect sizes (Phase 10) and a seeded bootstrap interval (Phase 10) | fewer than `min_samples` valid values |
| `ks_permutation` | p-value of the KS statistic; null: one shared distribution (exchangeable labels). Exact when few rearrangements exist, otherwise seeded Monte Carlo `(hits+1)/(n+1)` | as above |
| proportions | per category and window with Wilson intervals (Phase 10), difference = comparison − reference; categories only in one window are listed | more than 200 categories (not a categorical feature) |
| `jensen_shannon_divergence` | symmetric divergence in bits, 0–1 | as above |
| `total_variation_distance` | half the sum of absolute proportion differences, 0–1 | as above |
| `jensen_shannon_permutation` | p-value of the JS divergence, same null | as above |
| window metrics | the slice/evaluation metric engine (per-metric `higher_is_better`, bootstrap intervals) | metric undefined for the window |
| performance inference | Phase 10 unpaired comparison of the per-sample measure (accuracy / MAE) | overlapping samples, too few samples |

Assumptions and limits that always apply:

- Samples inside a window are treated as independent draws. Real time series are autocorrelated, which
  makes p-values **optimistic**.
- Distances are non-negative and biased upward in small samples: two samples of one distribution give a
  positive distance. They are compared with the permutation test, not with zero. No bootstrap interval
  is reported for KS or Wasserstein for that reason; the interval is for the mean difference.
- Windows contain *different* samples, so performance comparisons are unpaired and descriptive.
- With fewer than `min_samples` valid values in either window the status is `INSUFFICIENT_EVIDENCE`:
  descriptive measures are shown, inference (tests, intervals, effect sizes) is withheld.
- A distribution difference is not shown to affect the model. Predictions can shift without any
  performance change and vice versa; each dimension is reported separately.

## Evidence statuses

`OBSERVED` (a value read from the data: counts, window metrics), `DERIVED` (computed by a stated
method), `INCONCLUSIVE`, `UNDEFINED` (the statistic does not exist for this input), `UNAVAILABLE` (the
data does not exist) and `INSUFFICIENT_EVIDENCE`. A slice with **no members** in a window is
`UNDEFINED` with `reason_code: NO_MEMBERS` (an absence of samples); a slice with a few members is
`INSUFFICIENT_EVIDENCE`. Every value of a window is accounted for: `n_total = n_valid + n_missing +
n_nonfinite + n_invalid`.

## Multiple comparisons

Per-feature tests create many simultaneous hypotheses. `config.correction` is `NONE`, `BONFERRONI` or
`BENJAMINI_HOCHBERG` (Phase 10 `adjust_pvalues`), `alpha` the level. Each result stores `raw_p`,
`adjusted_p`, the `method` and its `family`; each document lists every family's members, size and the
results excluded for lacking a valid p-value. A family is one window pair × one population × one
result group (`covariate`, `target_and_output`, `performance`); `correction_scope:
"ACROSS_WINDOWS"` makes it one family over every window pair (recommended for long rolling analyses).
Magnitude (`measures`) and evidence (`test`) are separate fields, and `adjusted_p_below_alpha` is a
statement about a p-value, never a verdict.

## Slices

`slices` lists explicit Phase 11 slice definitions. Membership is evaluated *independently inside each
window*; it is never assumed identical over time. Windows' slice membership, counts and digests are in
`windows.json`. Slices are never enumerated automatically.

## Failure modes and reliability

`failure_modes` (`fmd_…`) adds, per window, the observed prevalence of each mode's affected samples
(the Phase 11 failure-slice integration). That is an evidence link (window → evaluation → observed
failure prevalence), not an explanation. A reliability profile can reference drift analyses
(`drift_analyses`) as the `DISTRIBUTION_SHIFT` dimension (`UNAVAILABLE` when none is referenced,
`INSUFFICIENT_EVIDENCE` when no result had enough samples), copied as recorded; it never merges with
another dimension and forms no score. A benchmark spec can request one drift analysis (`drift`, its
`baseline_run` is filled in with the benchmark's baseline); benchmarks without one are unchanged.

## Provenance, replay, artifacts

The provenance fingerprint covers dataset, split, evaluation configuration, model fingerprint,
baseline, ordering, every window's identity and membership digest, skipped windows, features and types,
the full configuration (methods, seed, correction, thresholds, metrics), slice identities and per-window
membership digests, failure-mode content hashes and software versions. Changing any of them changes it;
reordering features or renaming a window/slice does not. The run's own Provenance record holds the
source revision and environment.

`drift replay` re-executes the run as a new run and compares every document within 1e-12; a document that
no longer matches its stored digest is refused rather than reported as a difference. Artifacts
(`drift/`): `spec.json`, `windows.json`, `feature_results.json`, `distribution_results.json`,
`performance_results.json` (also failure evidence), `summary.json`.

## Records and commands

`DriftWindow` (`twn_`) and `DriftAnalysis` (`dan_`), schema v11 (`temporal_windows`, `drift_analyses`).

```
experionyx drift validate SPEC [--preflight]   # identity, window plan; --preflight checks the data
experionyx drift windows SPEC [--ids]          # members and digests per window (nothing stored)
experionyx drift compare SPEC [--full]         # compute without storing
experionyx drift evaluate SPEC                 # run and record (exit 3 if PARTIAL)
experionyx drift list [--windows]              # analyses / registered windows
experionyx drift inspect ID [--full]           # dan_ with provenance, or twn_
experionyx drift replay ID                     # exit 1 on any difference
```

Example spec:

```json
{"baseline_run": "run_…", "ordering": {"field": "feature:ts"},
 "reference": {"start": 0, "end": 1000}, "comparisons": [{"start": 1000, "end": 2000}],
 "features": [{"name": "age", "type": "NUMERIC"}, {"name": "region", "type": "CATEGORICAL"}],
 "config": {"min_samples": 30, "correction": "BENJAMINI_HOCHBERG", "seed": 0}}
```

## Limitations

The engine analyzes one evaluated split of one baseline run; it does not ingest a stream. Only
univariate distributions are compared (no joint or multivariate shift, no embeddings, no text).
Windows are numeric intervals. There is no drift alarm, threshold recommendation or change-point
detection. Concept drift (a changed relation between inputs and targets) is not tested; only marginal
label, prediction and performance differences are reported. Small windows give wide uncertainty and
`INSUFFICIENT_EVIDENCE`. The tests are permutation tests and cost grows with window size and features.
