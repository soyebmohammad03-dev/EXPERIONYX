# Reliability Profiles

A **reliability profile** summarizes the reliability *evidence* EXPERIONYX has accumulated about one
evaluated system (a model on a dataset under an evaluation configuration). It is built only from
persisted results of earlier phases, keeps every observation next to its source, and has **no score,
ranking or verdict**. It answers "what evidence exists, and what does it show?", not "is this model
reliable?". Nothing in it generalizes beyond the evaluated model, dataset and configuration.

## What is measured, derived, and interpreted

| Label | Meaning | Where it comes from |
|---|---|---|
| **OBSERVED** | values read directly from stored evaluation results | baseline metrics, calibration, latency, class recall, slices |
| **DERIVED** | computed by an earlier phase and read back, not recomputed | fault aggregation and effect classes (Phase 5), failure modes and prevalence (Phase 6), interaction contrasts (Phase 7) |
| **INTERPRETED** | deterministic sentence templates over the values above | `interpreted.statements`, each with its `basis` references |

## Dimensions (each has an explicit status)

`BASELINE_PERFORMANCE`, `FAULT_SENSITIVITY`, `FAILURE_PREVALENCE`, `FAILURE_SEVERITY`,
`INTERACTION_SENSITIVITY`, `SLICE_SENSITIVITY`, `REPRODUCIBILITY`, `UNCERTAINTY`, `LATENCY`,
`CALIBRATION`. Status is one of `OBSERVED`, `DERIVED`, `UNAVAILABLE` (the evidence does not exist; the
reason is stated) or `INSUFFICIENT_EVIDENCE` (it exists but is too thin to summarize). Missing
dimensions are never fabricated, and no universal threshold is applied.

- **Baseline**: every metric with value, status, direction, sample count and stored interval; error counts; the baseline artifact IDs.
- **Fault sensitivity**: per fault experiment: fault type, version, parameters, scope, components, trial counts (completed, failed, skipped, seeds), and per sweep point the faulted and deterioration aggregates with their bootstrap intervals and the *diagnostic* effect classification. Faults are listed, never ranked.
- **Failure prevalence / severity**: per included mode: lifecycle state, `established` (true only for `CONFIRMED`), prevalence with its denominator, the severity *vector*, classes, slices, fault types, supporting experiments and whether a passing reproduction check exists. A mode that is not `CONFIRMED` is flagged as a grouping of signals, not an established finding.
- **Interaction sensitivity**: per analysis: component faults, metric, contrast, normalized contrast, interval, order effect, label, lifecycle state, trial counts, reproduction. Never a causal claim.
- **Slice sensitivity**: reuses the stored slice results and per-class recall of the baseline, plus class/slice-level interaction effects; what is absent is reported `UNAVAILABLE`, never assumed uniform. No slice engine was added.
- **Reproducibility**: trial and seed counts, mode reproduction checks, interaction lifecycle, environments seen, and an explicit list of *unresolved issues*. Replay is reported as `NOT_RUN` at construction (use `reliability replay`). There is no reliability percentage.
- **Uncertainty**: every stored interval collected with its method, confidence, n and resamples, plus the caveats the sources carried (e.g. a single control run). Intervals from different methods are neither combined nor treated as comparable.
- **Latency, calibration**: observed from the baseline when computed, otherwise `UNAVAILABLE` with the evaluation's reason.

## Scopes and compatibility

| Scope | Must match across every source |
|---|---|
| `MODEL_DATASET` | model fingerprint, dataset fingerprint |
| `MODEL_DATASET_SPLIT` | ... and the evaluated split |
| `MODEL_DATASET_EVALUATION` | ... and the evaluation configuration |

There is no cross-dataset scope: aggregating across unrelated datasets is deliberately not offered. An
extension would need a new scope *and* its compatibility rules. Incompatible sources (a fault
experiment, interaction or failure mode from another model, dataset, or - per scope - split/evaluation
configuration; a faulted or unfinished baseline) are **refused before any run is created**, listing every
issue with what was required, what was found and why. Failure modes are checked by the model and dataset
fingerprints of their signals (their evaluated split is not recorded per mode).

## Identity, provenance, persistence

A profile's identity is the content hash of investigation + spec (scope + the source IDs). The
provenance fingerprint additionally covers every included run's provenance fingerprint, each fault
experiment's design, each interaction's provenance fingerprint and each failure mode's full record
(including its lifecycle state), so it changes when any of them does. A profile is an immutable
**snapshot**: later lifecycle changes do not alter it; build a new profile (a new fingerprint) to see them.

Schema v7 adds two append-only tables, `reliability_profiles` (metadata and dimension statuses only) and
`reliability_references` (which run, artifact, fault experiment, failure mode or interaction supports which
dimension). The observations live in the digest-verified artifact `reliability/profile.json` (and
`reliability/sources.json`), never in a database blob. Building a profile is a Run with a deterministic
claim ("assembled from stored evidence ... no assessment of overall reliability") linked to the baseline
run and the profile artifact.

## Comparison

`reliability compare A B` requires the same scope and dataset (and split / evaluation configuration where
the scope demands); the model may differ. It reports raw differences (B minus A) per dimension: matched
baseline metrics and calibration/latency values, faults matched by their full definition (type, version,
parameters, scope, components), modes matched only by structure (category, classes, slices, fault types),
interactions matched by their component faults and metric, slices and classes by name, plus side-by-side
reproducibility and uncertainty counts. Unmatched items are listed, not forced together. There is no
winner, ranking, or better/worse language; both profiles remain independently inspectable.

## Replay

`reliability replay` re-runs the profile as a new Run and compares the whole document. If the included
evidence changed since the profile was built (for instance a failure mode was rejected), the fingerprints
differ and the result is reported as `sources_changed` with `deterministic: null`, not as nondeterminism.

## CLI

```
experionyx reliability profile SPEC.json [--investigation I --format text]
experionyx reliability list [--model FP --dataset FP --scope S --split S --evaluation H --ref ID --dimension D --dimension-status S]
experionyx reliability inspect rpf_... [--dimension D | --full] [--format text]
experionyx reliability evidence rpf_... [--dimension D]
experionyx reliability compare rpf_... rpf_... [--format text]
experionyx reliability replay rpf_...
```

`SPEC.json`: `{"scope": "MODEL_DATASET_EVALUATION", "baseline_run": "run_...", "fault_experiments": [...], "interactions": [...], "failure_modes": [...]}`.

## Limitations

- A profile describes only the included evidence; it says nothing about data or faults not evaluated.
- Phase 7 limitations carry through unchanged and are shown in the uncertainty caveats: a single control
  run is held fixed in interaction intervals; no BCa bootstrap; per-sample label-fault analysis is refused;
  failure-mode comparison needs Phase 6 coverage of every treatment cell; only the primary metric drives
  interaction lifecycle labels; there is no multiplicity correction. They belong to a later
  statistics-focused phase and were not expanded here.
- Fault-experiment aggregates use the aggregation stored by Phase 5; profiles do not recompute them.
- Failure-mode splits are not recorded, so split compatibility is not enforced for modes.
- Comparison across profiles matches structure, not identity; a match is a pointer for a human.
