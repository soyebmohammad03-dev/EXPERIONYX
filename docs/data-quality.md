# Data quality laboratory (Phase 13)

Is this dataset fit for the use it was registered for? The data-quality engine answers with
**multidimensional, auditable evidence**: typed checks with explicit configuration, each producing a
status on one scope, its measurements, its thresholds, any statistical evidence and the affected rows.
It does **not** produce a data-quality score, grade or ranking, does not decide that data is "good" or
"bad", does not infer domain rules, and does not claim that a data property caused any model behaviour.

## The specification

`QualitySpec` (`dqs_…`, schema v1) names a registered dataset (`dst_…`) and declares everything the
checks may rely on. Nothing is inferred:

| Field | Meaning |
|---|---|
| `splits` | the dataset splits to analyze (empty: every split it declares, or the whole dataset) |
| `features` | `{name, type}` with type `NUMERIC`, `CATEGORICAL` or `BOOLEAN`; types are never guessed |
| `target` | `{task: CLASSIFICATION \| REGRESSION}` |
| `id_column` | a column the author declares to be a sample identifier |
| `ordering` | `index` or `feature:<name>` (the Phase 12 ordering), needed for windows and `temporal_order` |
| `missing_values` | extra string tokens that mean *missing* (`None` and NaN always do; the empty string is allowed) |
| `checks` | the checks below, each a type plus a configuration |
| `slices` | explicit Phase 11 slices; the checks listed in `config.slice_checks` are re-run inside each |
| `config` | `min_members`, `confidence`, `resamples`, `permutations`, `seed`, `correction`, `alpha`, `max_examples`, `slice_checks` |

A **check** is normalized (defaults spelled out, lists sorted, `5.0` = `5`) and its identity `qcs_…` is a
hash of its type and normalized configuration, so equivalent spellings share it and any real change
alters it. An invalid rule is refused before anything runs: unknown check types, unknown keys, rates
outside [0, 1], `min > max`, unsupported feature types, checks naming undeclared features, a
`target_leakage` with no named candidates, windows without an ordering, splits not listed, and so on.
Data-dependent refusals (unknown split, feature, id or ordering column, a dataset whose content no
longer matches its registered fingerprint) are made by a *preflight* before a run is created, so a
refused request executes nothing.

## Statuses

| Status | Meaning |
|---|---|
| `PASS` | no configured expectation was violated (observations are still recorded) |
| `FAIL` | a **configured** rule was violated: a declared type, range, allowed set, threshold, minimum |
| `WARNING` | a phenomenon that needs interpretation was observed (see below); never a verdict |
| `INCONCLUSIVE` | too little valid data (e.g. fewer than `min_members` rows, nothing valid) |
| `UNAVAILABLE` | the data the check needs does not exist (no aligned target, a slice reading an absent field) |
| `NOT_APPLICABLE` | the check does not apply to this scope (an empty table, a slice with no members) |

`FAIL` requires a rule *you* configured. `WARNING` is used only for findings that are informative
in themselves: duplicate rows or ids, identifier-like columns, non-finite values, constant and
near-constant features, unseen categories, singleton classes, leakage indicators and order violations.
A warning is evidence, not an automatic judgment. Results with no configured expectation (missingness,
outliers, imbalance) stay `PASS` and carry their observations. Statuses are never combined: the summary
counts statuses per check type and scope and says nothing more. A check on several features reports
the most attention-needing feature status (`FAIL` > `WARNING` > `INCONCLUSIVE` > `PASS`) and keeps the
per-feature status in its observations.

## The checks

| Check | Measures | Notes |
|---|---|---|
| `schema` | required/extra columns, ragged rows (row width ≠ column count, e.g. wrong metadata), values that do not fit a declared type | cells are never coerced |
| `sample_count` | `min`/`max`/`expected` rows; the adapter's own count, the target length, repeated sample IDs | an empty table warns |
| `missingness` | per-feature and per-row missing counts and rates (Wilson intervals), missing-per-row histogram, top co-occurrence patterns; optional `max_feature_rate`, `max_row_rate` | missingness may be informative; it is measured, not judged |
| `duplicates` | exact duplicate rows (optionally including the target); identical features with different targets | never `FAIL` unless `max_duplicate_rate`/`max_conflicting_rate` is set; a repeated row may be legitimate |
| `identifiers` | a declared `id_column`'s duplicates, conflicting rows sharing an id, missing ids; identifier-like columns (unique ratio ≥ `unique_ratio` among integer-valued and categorical columns) | continuous columns are not candidates |
| `numeric` | counts of missing / non-finite / invalid, constant and near-constant (`near_constant_ratio`), `min_variance`, `bounds`, outliers | see outliers below |
| `categorical` | counts, cardinality, `allowed` set, rare categories (`rare_fraction`), `max_cardinality` | **invalid** (outside the declared set) ≠ **rare** (below a fraction) ≠ **new** (only in the comparison group) |
| `target` | missing/invalid/mixed-type targets, class counts and proportions (Wilson), singleton and rare classes, imbalance ratio; regression: summary and outliers | class imbalance is never a failure by itself; `min_class_count`/`min_class_proportion`/`max_missing_rate` are configured rules |
| `distribution` | skewness, excess kurtosis, modal fraction, integer-valued fraction, distinct values | descriptive; warns only on configured `max_abs_skew`, `max_modal_fraction`, `min_unique` |
| `split_overlap` | sample-ID overlap and identical feature vectors between two declared splits, and whether their labels agree | a leakage **indicator** |
| `target_leakage` | for the **explicitly named** candidates: equality with the target, \|Pearson r\| with a numeric target, purity | conservative; see below |
| `temporal_order` | whether the comparison split lies after the reference split in the declared ordering; unusable ordering keys | ties allowed unless `allow_ties: false` |
| `group_comparison` | between two groups (a split, optionally narrowed by a Phase 12 window): missingness and validity rates, category availability, the target distribution | Phase 10 statistics, corrected per family |

**Outliers** use Tukey fences (`q1 − k·IQR`, `q3 + k·IQR`, `k = 1.5`, type-7 quantiles) or the
modified z-score `0.6745·(x − median)/MAD` (`|z| > 3.5`). They describe extremeness under the method and
are never called errors; they change a status only when `max_outlier_rate` is configured.

**Leakage indicators** are conservative and need interpretation. Purity (the share of rows whose
feature value's majority target agrees, over values with at least two rows) is flagged only when the
feature has at least two values covering at least half of the rows and beats the target's own majority
share, so a constant feature or a lopsided target does not look like leakage. Thresholds
(`max_abs_correlation`, `min_purity`) are configuration: the same data can pass at one setting and warn
at another. No model is trained and nothing is guessed; only named candidates are examined.

## Statistical evidence

Reused from Phase 10: Wilson intervals for proportions, `compare` (unpaired permutation test, effect
sizes, seeded bootstrap interval) on 0/1 indicators for missingness and validity rate differences, and
`adjust_pvalues` for multiple-comparison correction (`NONE`, `BONFERRONI`, `BENJAMINI_HOCHBERG`). From
Phase 12: the cleaning rules and `shift` for the target distribution. Each `group_comparison` result
keeps observations, thresholds and statistics apart; a family is one aspect (one hypothesis per
feature) between two groups, listed with raw and adjusted p-values. A p-value is never a verdict, and
the groups are refused (`INCONCLUSIVE`) if they share samples or have fewer than `min_members` rows.
Distances and rates are computed with independent-sample assumptions; temporal autocorrelation makes
p-values optimistic.

## Slices and windows

`slices` are explicit Phase 11 definitions, evaluated per split; `config.slice_checks` (default
`categorical`, `distribution`, `duplicates`, `missingness`, `numeric`, `target`) are re-run on each
slice's rows with its identity in the scope. A slice with no members is `NOT_APPLICABLE` (`NO_MEMBERS`), a
slice with fewer than `min_members` rows is `INCONCLUSIVE` (values are shown), and a slice reading a
field the dataset lacks is `UNAVAILABLE`. Slices are never enumerated automatically. Windows narrow a
group in `group_comparison` (missingness changed over time, category availability, class proportions);
a sample with no usable ordering key belongs to no window. Nothing here says a temporal change caused
model degradation.

## Artifacts, provenance, replay

`data_quality/`: `spec.json` (spec, dataset identity and adapter, columns, per-split digests, check ids,
thresholds, seed, versions, slice ids and per-split membership digests), `checks.json` (each check's
definition and per-scope status), `observations.json` (measurements and statistics per check and scope),
`violations.json` (every violation with exact counts and at most `examples` affected sample IDs, with
`truncated`), `summary.json` (status counts; no score). The provenance fingerprint covers the dataset
fingerprint, adapter, columns, per-split sample digests, the whole spec, check ids, slice membership and
software versions; changing any of them changes it, reordering columns, rows, checks or splits does not.
The run's Provenance record holds the source revision and environment. `data-quality replay` re-executes
the run as a new run and compares all five documents; a document that no longer matches its digest is
refused, not reported as a difference.

## Records and commands

`QualityAnalysis` (`qan_`) and one `QualityCheck` (`qck_`) per check and scope; schema v12
(`quality_analyses`, `quality_checks`).

```
experionyx data-quality validate SPEC [--preflight]   # identity; --preflight loads the dataset and refuses what it cannot support
experionyx data-quality run SPEC [--investigation ID] # exit 3 if any check FAILED or evidence is incomplete
experionyx data-quality list [--dataset-id ID] [--status COMPLETE|PARTIAL]
experionyx data-quality inspect ID [--full]           # qan_ with provenance and artifacts, or a qck_
experionyx data-quality checks [ANALYSIS] [--type T] [--status S]   # the catalog, or the per-check results
experionyx data-quality replay ID                     # exit 1 on any difference
```

```json
{"dataset_id": "dst_…", "splits": ["train", "test"], "target": {"task": "CLASSIFICATION"},
 "features": [{"name": "age", "type": "NUMERIC"}, {"name": "region", "type": "CATEGORICAL"}],
 "checks": [{"type": "missingness", "config": {"max_feature_rate": 0.2}},
            {"type": "numeric", "config": {"bounds": {"age": [0, 120]}}},
            {"type": "split_overlap", "config": {"reference": "train", "comparison": "test"}}]}
```

## Limitations

Quality is multidimensional and configuration-dependent: thresholds are yours, and the same data can
pass or fail under different ones. A statistical anomaly is not a data error, an outlier is not a bad
row, a duplicate is not necessarily a defect, and class imbalance is context-dependent. Missingness can
carry information. Leakage indicators show a property of these rows under stated thresholds; they do not
prove leakage, and leakage that needs a model or domain knowledge to see is out of scope. The checks
read whole splits in memory (refused above two million rows) and compare univariate properties; there is
no joint or semantic validation, no text or image quality, and no automatic repair. Only registered
datasets served by an adapter that exposes named or indexed feature columns can be analyzed.
