# Slice and subgroup analysis (Phase 11)

How does this model behave across explicitly defined populations of a dataset? The slice engine
answers that with per-slice evidence and uncertainty. It does **not** rank groups, invent protected
attributes, produce a fairness or quality score, say a group is better or worse overall, or explain
causes.

## Slice definition

`SliceSpec(name, condition)` is immutable. A condition is a small typed tree; no expressions are ever
evaluated, unknown operators and stray keys are rejected, nesting is bounded.

| Operator | Meaning |
|---|---|
| `eq` | `field == value` (bool, number or string; kinds are strict: `1` is not `true`, `"1"` is not `1`) |
| `in` | `field` is one of the values |
| `range` | numeric `low <=/< field <=/< high`, either bound optional, inclusivity explicit (default `[low, high)`, like the evaluation engine's feature ranges) |
| `is_true` | a boolean field is `true` |
| `and` / `or` / `not` | logical composition |

`label(v)` is `eq("target", v)`, a class slice. Fields are `target`, `predicted`, `correct`,
`confidence` and dataset features as `feature:<name or column index>`.

**Identity** is the normalized condition (plus the schema version): children of `and`/`or` are
flattened, de-duplicated and sorted, a one-value `in` becomes `eq`, `not not x` becomes `x`, integral
floats become ints. Equivalent spellings therefore share one `sls_` ID whatever they are named, and
changing any part of the condition changes it. Logical equivalence beyond that (De Morgan, containment
of ranges) is not decided; such definitions stay distinct records.

A slice that reads `predicted`, `correct` or `confidence` depends on the model run (**non-static**). It
can be evaluated on one run but is refused in fault and interaction analysis, where membership would
differ between runs.

## Membership

Three-valued logic. A sample's condition is true, false or **unknown** (field absent, `None`, NaN or
infinite, or of the wrong kind). Unknown is never a match; those samples are counted and their IDs
(first 20) listed. AND is false if any argument is false, OR is true if any is true, NOT unknown is
unknown. Statuses: `COMPUTED`, `EMPTY` (a valid slice with no members), `MISSING_FIELD` (a field
exists in no sample: undefined, not empty), `NO_SAMPLES`. Duplicate sample IDs are refused. Values a
condition names but the data never contains are reported (`unseen_values`). Membership is a function of
table content, iterated in sorted ID order: row order cannot change it (tests shuffle rows and edit the
predictions file; a reordered artifact is refused by its digest).

Sample IDs are dataset indices within the evaluated split. Predictions come from the run's
digest-verified `evaluation/predictions.jsonl`; features come from the registered dataset adapter
(fingerprint-verified). Feature slices without the dataset are `UNAVAILABLE`, never guessed.

## Metrics, comparisons

Per slice: every scalar metric of the task through the evaluation metric registry, with the evaluation
bootstrap's interval; error counts (by true class); calibration when scores were stored for every
member; latency is `UNAVAILABLE` (it is measured per batch). A slice below `min_members` is reported
but flagged `INSUFFICIENT_EVIDENCE`.

Comparisons use the Phase 10 statistics core on a per-sample measure (`accuracy` = mean correctness,
or `mae` for regression). `a` minus `b`: raw and relative difference, effect sizes, permutation test,
bootstrap interval, sample counts, status. Slice vs slice, slice vs `REST` (the complement) or slice vs
`POPULATION`. The population contains the slice, so the delta against it is descriptive and the
inference is against the rest. Overlapping groups get the descriptive difference but no test. An
optional correction (`BONFERRONI`, `BENJAMINI_HOCHBERG`) is applied over the tests actually run and
recorded with its family size.

## Integrations

- **Faults:** for each fault experiment and sweep point, per-trial slice and population degradation
  (positive = worse for the metric's direction), paired by trial; the effect classification comes from
  the interval of (slice degradation - population degradation): `SLICE_MORE_DEGRADED`,
  `SLICE_LESS_DEGRADED`, `NO_EVIDENCE_OF_DIFFERENCE` (not evidence of absence) or
  `INSUFFICIENT_EVIDENCE` (fewer than `min_trials`, too few members, a different baseline). Membership
  comes from the original dataset, so a feature fault does not move samples between slices.
- **Failure modes:** per mode, the share of the slice's samples touched by any signal of the mode
  (Wilson interval), presence, samples outside the slice and in the population, runs touching the
  slice, lifecycle state, evidence count. Modes whose signals did not record every sample ID, or come
  from another dataset, are `INSUFFICIENT_EVIDENCE`. Only `CONFIRMED` modes are called confirmed.
- **Interactions:** the slice's per-trial cell values feed the Phase 7 bootstrap and classification
  (`interactions.calc`), giving the contrast on the slice. Missing members, changed ground truth or too
  few trials give `INSUFFICIENT_EVIDENCE` with `fallback: none`: population-level results are never
  substituted.
- **Reliability profiles:** `ProfileSpec.slice_analyses` (omitted from the identity when empty) adds the
  analysis to `SLICE_SENSITIVITY` with each slice's recorded status and evidence copied as is. A profile
  refuses a slice analysis over another baseline run or dataset.
- **Benchmarks:** a benchmark runs a slice analysis only if its spec lists `slices` (with an optional
  `slice_config`); the result then has a `slice_analysis` section and coverage entry, and an
  insufficient analysis makes coverage incomplete with a reason. Specs without slices keep their
  identity and run nothing extra.

## Records and artifacts

`Slice` (`sls_`, the logical definition) and `SliceAnalysis` (`san_`, one analysis run) are immutable
(schema v10). Each analysis is a Run whose artifacts are `slice/spec.json`, `slice/membership.json`,
`slice/results.json` and `slice/summary.json` (member IDs, no dataset copy). Provenance: dataset
fingerprint, split, evaluation config hash, slice IDs and membership digests, metric configuration,
source run and experiment IDs. The analysis identity is investigation + request; a changed condition,
setting or source is a different analysis.

## CLI

```
experionyx slice validate SPEC.json           # normalize and print identities
experionyx slice register SPEC.json           # equivalent definitions are one record
experionyx slice list [--field F --text T --static yes|no --analyses --baseline-run RUN]
experionyx slice inspect <sls_|san_> [--full]
experionyx slice evaluate SPEC.json --baseline-run RUN [--ids]      # nothing stored
experionyx slice analyze ANALYSIS.json         # exit 3 if some evidence is insufficient/unavailable
experionyx slice compare A.json (B.json|POPULATION|REST) --baseline-run RUN
```

`ANALYSIS.json`: `{"baseline_run", "slices": [...], "comparisons": [["a","REST"]], "fault_experiments":
[...], "failure_modes": [...], "interactions": [...], "config": {...}}`.

## Limitations

- Sample IDs are dataset indices; a run that evaluated a different sample set cannot be aligned.
- Only per-sample-decomposable measures (accuracy, MAE) get tests; F1, AUC etc. get values and
  intervals but no slice-vs-slice test.
- Failure-mode prevalence treats samples as independent, which they are not; it describes the sample
  set. Modes need recorded sample IDs.
- No claim objects are created for slice analyses.
- Latency cannot be attributed to a slice.
- Benchmark comparison (`benchmark compare`) does not diff slice sections yet.
- A slice of a few samples has wide intervals; `min_members` and `min_trials` only flag, they do not
  make small evidence adequate.
