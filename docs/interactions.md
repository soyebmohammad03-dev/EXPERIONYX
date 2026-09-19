# Fault Interaction Analysis

Given two faults A and B, the interaction engine measures whether applying them together
departs from what the single-fault results would predict *additively*, under an explicit
experimental design, and reports the raw numbers, their uncertainty, and a rule-based label.
It is deterministic and inspectable: no LLM, no learned model, no hidden score.

## Measures, infers, and what a human may conclude

| | Content |
|---|---|
| **Measures (OBSERVED)** | Metric values of stored evaluation runs, per trial, in each cell of the design; per-sample outcomes from stored predictions. |
| **Infers (DERIVED + INTERPRETED)** | Effects, an additive contrast, a percentile bootstrap interval over trials, a direction-aware reading, and a label (`NO_EVIDENCE`, `POSSIBLE_INTERACTION`, `OBSERVED_INTERACTION`, `ORDER_DEPENDENT_INTERACTION`, `INCONCLUSIVE`, `UNDEFINED`) under thresholds recorded with the result. |
| **A human may conclude** | That, under this design and model/dataset, the compound treatment departed from the additive reference by the reported amount and interval. Anything about mechanism, cause, or generality needs further, purpose-built experiments. |

The system never states that A causes B, that the faults interact mechanistically, that a fault
is responsible for a failure, or that a model is vulnerable because of a fault. The relationship
graph has no causal predicate. `NO_EVIDENCE` is not evidence of absence.

## The four-cell design

Each cell is a set of finished, evaluated runs (its *trials*):

| Cell | Treatment | Value |
|---|---|---|
| CONTROL | none | Y0 |
| A | fault A | YA |
| B | fault B | YB |
| AB | A then B (compound, in this order) | YAB |
| BA | B then A (only for order analysis) | YBA |

```
effect_A = YA - Y0     effect_B = YB - Y0     combined = YAB - Y0
expected_additive = effect_A + effect_B
interaction_contrast = combined - expected_additive = YAB - YA - YB + Y0
order_effect = I_AB - I_BA = YAB - YBA        (I_BA = YBA - YA - YB + Y0)
```

The contrast is a **mathematical quantity**. It is not causal and not a performance
deterioration. Its sign means different things for higher-is-better and lower-is-better
metrics, so the raw value is stored unchanged and a separate `deterioration_contrast` (> 0 means
the combination is worse than additive for that metric's direction) and a relation
(`SUPER_ADDITIVE_DETERIORATION`, `SUB_ADDITIVE_DETERIORATION`, `ADDITIVE`, `UNKNOWN_DIRECTION`)
are given. All intermediate values are persisted, not only the contrast.

Normalization is an explicit configuration (`NONE`, `BASELINE_MAGNITUDE` = |Y0|,
`EXPECTED_ADDITIVE_MAGNITUDE` = |effect_A + effect_B|, `MAX_COMPONENT_MAGNITUDE`). A zero
denominator gives no number and a stated reason; NaN/inf are never persisted (undefined inputs
make the effect `UNDEFINED`).

## Design validation (refusal before analysis)

`interaction validate` and `analyze` load every run and refuse the design, listing every issue
(*required / found / why*), before any number exists or any record is created. Checks: every
cell has a COMPLETED run with a digest-verified evaluation; the control carries no fault and
treatments do; A and B are single faults and AB/BA are their **ordered** compositions with
identical parameters and scope; one fault definition per cell (seeds may differ); A and B differ;
same model fingerprint, source dataset fingerprint, split, evaluation configuration, task, sample
count and classes; same environment (configurable); consistent metric direction; the primary
metric is defined in the control; unique trial seeds per cell; pairing is justified. Runs that are
not COMPLETED are excluded and **listed**, not hidden. Too few trials is not a refusal: the
effect is reported without an interval (`INSUFFICIENT_DATA`, class `INCONCLUSIVE`).

## Repeated trials, pairing and aggregation

A trial is one run of one cell; trials are independent only if they came from independently
seeded runs. The fault seed is the trial key. `PAIRED` (declared, and checked: identical seed
sets in every treatment cell, one control run) pairs by that key, never by position; `UNPAIRED`
treats cells as independent samples; `UNKNOWN` is analyzed as unpaired and says so. Aggregation
is `MEAN` or `MEDIAN`; per cell the trial count, mean, median, standard deviation and standard
error are kept alongside every raw value (`interaction/trials.json`, `effects.json`).

## Bootstrap

A percentile bootstrap over **trials** (whole runs, never individual samples): unpaired resamples
each cell independently; paired resamples seed keys jointly. A single control run is held fixed
(its own variability is not represented; this is warned). Settings (resamples, seed, confidence)
are recorded; different settings are a different analysis identity. Below `min_trials` per
treatment cell no interval is produced. BCa is not implemented. **The interval is a descriptive
spread of trial-level resampling, not a significance test.** With few trials it is imprecise.

## Labels

Per measure, from configured rules: `UNDEFINED` (no contrast); `INCONCLUSIVE` (no interval);
`ORDER_DEPENDENT_INTERACTION` (the order-effect interval excludes zero **and** at least one
ordered contrast's interval excludes zero); `OBSERVED_INTERACTION` (interval excludes zero, sign
stable in >= `min_sign_fraction` of resamples, |contrast| >= `min_abs_contrast`); `POSSIBLE_INTERACTION`
(part of that, including an order effect with no supported ordered contrast); otherwise
`NO_EVIDENCE`. A `min_abs_contrast` of 0 means *no magnitude criterion* and never suffices by
itself. The rule that fired and its inputs are stored with the label. Defaults are conservative;
there are no universal thresholds.

## Order effects

Requested with `order_analysis` and a BA cell (B then A, exact component order verified).
`order_effect = YAB - YBA` is kept with its own interval. If order analysis is not requested it is
reported as not requested; the engine never assumes two faults commute or that they do not.

## Per-sample, class and slice analysis

Per-sample contrasts (`error`, `confidence`, `true_class_probability`, or `abs_error`/`signed_error`
for regression) use the sample's **dataset index within the split** as its ID, and are computed only
after alignment is *verified*: identical index sets in every run, no duplicate indices, identical
ground-truth targets per index (so label faults are refused), plus the design-level dataset/split
checks. Row order in a file never matters. Otherwise the per-sample analysis is **refused with the
reason**, never approximated. Cell values are means over trials; samples are not independent draws,
so no per-sample interval is computed. Per-class recall and per-slice metrics are analyzed with the
same machinery as ordinary metrics (levels `CLASS`, `SLICE`). Per-sample slice membership is not
recorded (slices are reported at metric level). Per-sample analysis is refused beyond `max_samples`.

## Multiplicity

Every measure is a separate contrast; nothing is corrected. `effects.json` and the summary report
the number of hypotheses tested and say so. Treat the primary metric as the pre-declared analysis and
every other result as exploratory.

## Failure-mode interaction

With a `discovery_run_id` (a Phase 6 discovery that covered every treatment cell's fault experiment),
each mode's prevalence per cell (runs with a signal / runs in the cell) is compared: the states
`OBSERVED_UNDER_COMPOUND_TREATMENT`, `NEWLY_OBSERVED`, `PERSISTING`, `INCREASED_PREVALENCE`,
`REDUCED_PREVALENCE` (including absent) and `INSUFFICIENT_EVIDENCE` (mode below `CANDIDATE`). If the
discovery did not cover every treatment run, the result is `NOT_AVAILABLE` with the reason: modes
found in separate discoveries are different clusters and cannot be compared. States describe
observation, never cause.

## Lifecycle, evidence, provenance

`DISCOVERED` (a valid contrast computed) -> `SUPPORTED` (primary label is `OBSERVED_INTERACTION` or
`ORDER_DEPENDENT_INTERACTION`) -> `REPRODUCIBLE` (`interaction reproduce`: an **independent
replicate** with the same structure and disjoint treatment runs agrees in sign and within tolerance)
-> `CONFIRMED_BY_REVIEW` (`interaction confirm`, a named person and a reason). `REJECTED` and
`DEPRECATED` are reachable by validated transitions. Every transition stores previous and new state,
reason, actor, time and evidence references, and evidence is never deleted. Nothing beyond
`SUPPORTED` is automatic.

The analysis identity is the content hash of investigation + spec (cells' run IDs + configuration).
The provenance fingerprint additionally covers every run's own provenance fingerprint (fault, seed,
model, dataset, source, environment), so changing a fault seed, a fault parameter, the order, the
metric configuration or the bootstrap settings changes it (tested). The analysis is a Run; artifacts
`interaction/{spec, design_validation, trials, effects, bootstrap, per_sample, failure_modes, summary}.json`
are digest-registered; claims are deterministic templates linked to the runs and artifacts.

## Registry, graph and matching

`InteractionRegistry` searches by fault, metric, status, class, failure mode, experiment, dataset and
model, and retrieves effects, evidence, raw trials, artifacts and provenance. The graph extends the
Phase 6 relationships: `OBSERVED_WITH`, `AMPLIFIED_UNDER`, `SUPPRESSED_UNDER`, `ORDER_SENSITIVE_WITH`,
`SUPPORTED_BY`, `CO_OCCURS_WITH`; no causal predicate exists. `interaction related` reports
`SAME_STRUCTURE` (identical run-independent structure) or `POTENTIALLY_RELATED` (same fault
types/versions and metric) with every parameter/dataset/model difference visible; nothing is merged.

## Reproducibility

Replay (`interaction replay`) re-runs the analysis as a new Run and compares specification identity,
provenance fingerprint, every effect, aggregate, bootstrap interval and per-sample result within 1e-12
(relative). The bootstrap uses a seeded Mersenne Twister; bitwise equality across platforms or
Python versions is not claimed.

## CLI

```
experionyx interaction validate SPEC.json [--format text]
experionyx interaction analyze  SPEC.json [--investigation I --seed N --format text]
experionyx interaction list [--fault F --metric M --status S --class C --failure-mode fmd_... --experiment E --dataset FP --model FP]
experionyx interaction inspect ian_... [--full]
experionyx interaction failures | replay | export [--out FILE] | related  ian_...
experionyx interaction reproduce ian_... --replicate ian_... [--abs-tol X --rel-tol Y]
experionyx interaction confirm ian_... --by WHO --reason WHY
experionyx interaction set-status ian_... --to REJECTED|DEPRECATED --by WHO --reason WHY
```

`SPEC.json`: `{"control": [run,...], "a": [...], "b": [...], "ab": [...], "ba": [...], "discovery_run_id": null, "config": {...}}`.

## Limitations

- One control run holds Y0 fixed; its variability is not in the interval.
- Interval quality depends on the number of trials; the defaults need >= 3 per treatment cell.
- Percentile bootstrap only; the label rules are heuristics, not a hypothesis test.
- The additive contrast is one reference model; a non-zero contrast does not say why.
- Per-sample IDs are dataset indices; label faults cannot be analyzed per sample.
- No `compare` command between two analyses; use `export` and `related`.
- Order analysis is opt-in and applies to the two named faults only.
- Storage: analyses, effects and evidence are three tables; raw trials, bootstrap and per-sample data
  live in digest-verified artifacts, not in tables.
