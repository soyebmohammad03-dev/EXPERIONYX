# Robustness Benchmarks

A **benchmark** is a standardized, versioned robustness *protocol*: a model/dataset pair, an
evaluation configuration, explicit fault grids, seeds, optional interaction pairs, and analysis
settings. Running it expands the definition into explicit experiment **units**, executes them
through the existing machinery (fault laboratory, failure discovery, interaction analysis,
reliability profile), and produces a structured **result** with a **coverage account**. It never
produces a robustness score, ranking or verdict, and it never treats missing coverage as robustness.

## Definition (`BenchmarkSpec`)

| Field | Meaning |
|---|---|
| `name`, `version` | the definition's identity; `version` is `major.minor.patch` |
| `model`, `dataset` | registered record IDs (`mdl_...`, `dst_...`) |
| `stress` | optional Phase 14 stress plan run on the benchmark's own model, dataset, evaluation and baseline; adds a `stress_analysis` section that references the analysis (planned/executed/unsupported/failed/insufficient-evidence counts) and a profile reference; every earlier benchmark identity is unchanged |
| `calibration` | optional Phase 15 calibration request (a calibration spec without `baseline_run`) run on the benchmark's baseline; adds a `calibration_analysis` section that references the analysis (planned/executed/unavailable/insufficient-evidence/failed context counts, not duplicated) and a profile reference, and leaves every earlier benchmark identity unchanged |
| `resources` | optional Phase 16 resource measurements: a list of resource specs (each over the benchmark's own model and dataset; a batch-size sweep is several specs); adds a `resource_analysis` section whose coverage counts `planned`, `executed`, `unsupported`, `failed`, `timed_out` and `insufficient_evidence` units and references the analyses (not duplicated), and a profile reference; every earlier benchmark identity is unchanged |
| `drift` | optional Phase 12 drift request (a drift spec without `baseline_run`); adds a `drift_analysis` section and a profile reference, and leaves every earlier benchmark identity unchanged |
| `evaluation` | an `EvaluationConfig` (metrics, split, slices, bootstrap); the control uses it too |
| `faults` | grids: `name`, `fault`, `parameters`, optional `sweep_parameter` + `values`, optional `fault_version` pin, optional `scope_fraction` |
| `seeds` | trial seeds; every (grid point x seed) is a real run |
| `interactions` | pairs of grids at one point each (`a`, `b`, `a_value`, `b_value`, `order_analysis`) |
| `primary_metric`, `aggregation` | the metric effects are classified on; confidence/resamples/seed of the aggregation bootstrap |
| `interaction_config` | the Phase 7 `InteractionConfig` (bootstrap, thresholds) |
| `discovery`, `profile` | run Phase 6 discovery over the runs; assemble a Phase 8 profile |
| `limits` | `max_units`, `max_failed_trials`, `allow_unsupported` |

Identity is the content hash of the canonical form (`bsp_...`): changing anything scientifically
meaningful changes it, while listing grids, pairs or seeds in a different order does not. Example:

```json
{"name": "noise-dropout", "version": "1.0.0", "model": "mdl_...", "dataset": "dst_...",
 "faults": [{"name": "noise", "fault": "gaussian_noise", "sweep_parameter": "sigma", "values": [0.1, 0.25, 0.5, 1.0]},
            {"name": "dropout", "fault": "feature_dropout", "sweep_parameter": "probability", "values": [0.05, 0.1, 0.25]}],
 "seeds": [1, 2, 3], "interactions": [{"a": "noise", "b": "dropout", "a_value": 0.5, "b_value": 0.25}]}
```

## Protocol expansion and validation

Every requested point becomes an explicit unit; nothing is interpolated or skipped:
`baseline`, `fault:<grid>:<point>:<seed>`, and for interactions the compound cells
`interaction:<pair>:AB|BA:<seed>`. Interaction cells **A and B are the same fault specifications (same
seeds) as the grid points they name**, so the interaction analysis reuses those grid trials instead of
re-running identical evaluations; only the compound cells are new. Each unit records its exact seeded
fault and fault ID.

`benchmark validate` (and `run`) refuse invalid definitions **before any run or record exists**, listing
every issue (required / found / why): unknown or unimplemented faults, invalid parameters or sweep
values, unknown metrics, missing model or dataset, interactions naming unknown grids or untested points,
`limits.max_units` exceeded, faults the dataset cannot support. A fault that is merely *unsupported* by the
dataset is refused unless `allow_unsupported` is set, in which case it is excluded and recorded in the
coverage. Warnings (for example too few seeds for an interaction interval) are reported.

## Coverage (an account of what ran)

`coverage.json` states, without any score: fault families requested / tested / not tested with per-family
point and trial counts; parameter points requested, with any completed, and fully completed; trials
requested, completed, failed, skipped, not run; seeds; metrics requested, available and undefined (with
reasons); configured slices present or missing; interactions requested, analyzed, with an interval, and
inconclusive; failure modes discovered by lifecycle state; every failed, skipped or not-run unit with its
reason; unsupported combinations. `complete` is true only if every requested unit completed and every
requested analysis ran, and `incomplete_reasons` lists what is missing. An incomplete benchmark says so in
its first observation ("Coverage is INCOMPLETE ... Missing coverage is not evidence of robustness."), and
`benchmark run` exits 3 in that case.

## Results (`results.json`) and honesty about evidence

Sections: baseline, fault responses, interactions, failure modes, reliability profile (a reference),
uncertainty, reproducibility. Values are **read from persisted evidence**, not recomputed: baseline
metrics from the stored evaluation, fault responses from the Phase 5 analysis, interaction contrasts
from Phase 7, modes from Phase 6, the profile from Phase 8. Each section has a status: `OBSERVED`,
`DERIVED`, `UNAVAILABLE` (with a reason, e.g. a stage disabled in the definition) or
`INSUFFICIENT_EVIDENCE`. Interpreted statements are deterministic templates with their basis, for
example "Under fault family gaussian_noise (sigma=6) across 3 completed trial(s), accuracy changed from
0.85 to 0.608; mean deterioration 0.242 [0.2, 0.275]"; there is no "robust", "unreliable" or "safer".
Failure modes that are not `CONFIRMED` are flagged as groupings of signals, not established findings.

## Versioning and comparison

The **protocol hash** covers the engine version, everything in the spec except the model, the *resolved*
fault versions, the expanded unit list and any unsupported grids. Changing the benchmark version, a
grid, a seed, the evaluation, the analysis settings, or a fault implementation's version changes it.
Every result stores its exact definition, so old results stay interpretable. Collection re-expands the
protocol and refuses if it no longer matches what was executed (for example the fault library changed in
between). `benchmark compare` requires identical protocol hashes (only the model may differ); otherwise it
refuses with `PROTOCOL_DIFFERS` and each differing field. The comparison reports raw differences (B minus
A) for coverage, baseline metrics, fault responses by grid and point, interactions by pair, failure modes
matched only by structure (category, classes, slices, fault types), uncertainty and reproducibility.
There is no winner and no ranking.

## Execution, persistence, provenance

`benchmark run` validates, then executes: fault experiments per grid (the first creates the baseline; the
rest reuse it), the compound interaction cells, failure discovery over all of them, an interaction
analysis per pair, and a reliability profile; then a `collect` Run reads the persisted evidence and
writes `benchmark/{spec,units,coverage,results,summary}.json`, registers the records and a
deterministic claim linked to the baseline, every completed unit run and the result artifacts. Running
a definition that already has a result in the registry returns it (idempotent); it is not re-run.
A failure of one grid, pair or stage is recorded with its reason and the rest continues; only a missing
baseline aborts.

Schema v8 adds `benchmarks` (the definition), `benchmark_results` and `benchmark_units` (each unit with
its status and, when it ran, its run: benchmark -> unit -> run is a foreign-key chain). Observations stay
in digest-verified artifacts. The result's provenance fingerprint covers the protocol hash, every unit's
status and run provenance fingerprint, each interaction's provenance, every discovered mode's full
record (including lifecycle state) and the profile's fingerprint.

**Replay.** `benchmark replay` re-runs the collect Run and compares all artifacts. If the persisted
evidence changed since collection (say a mode was rejected) the fingerprints differ and the result is
reported as `sources_changed` (`deterministic: null`), not as nondeterminism. **Reproduction** in the
strong sense is executing the same definition in an independent registry: on the same deterministic
adapters the units, outcomes, coverage and every measured number agree (tested).

## CLI

```
experionyx benchmark validate SPEC.json [--format text]
experionyx benchmark run      SPEC.json [--format text]     # exit 0 complete, 3 incomplete coverage, 2 refused
experionyx benchmark list [--name N --version V --model mdl_ --dataset dst_ --protocol H --engine V --fault F --coverage COMPLETE|INCOMPLETE]
experionyx benchmark inspect  bmk_...|brs_... [--full] [--format text]
experionyx benchmark coverage bmk_...|brs_... [--format text]
experionyx benchmark compare  brs_... brs_... [--format text]
experionyx benchmark replay   brs_...
```

## Limitations

- Coverage counts executed units; it says nothing about faults, seeds or conditions outside the definition.
- Interaction analysis inherits the Phase 7 limits (a single control run, percentile bootstrap only,
  no multiplicity correction; only the primary metric drives lifecycle labels) and needs at least
  `min_trials` seeds for an interval.
- Only single-parameter sweeps and single-point interaction pairs are expressed; multi-parameter grids
  would be separate grids.
- Failure discovery runs over the benchmark's own experiments with the default (or supplied) configuration;
  its similarity weights and criteria are documented heuristics.
- Cross-registry reproduction is exact only for deterministic adapters and the same library versions.
- Comparison requires identical protocols; comparing different definitions is refused, not approximated.
