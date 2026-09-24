# Reproducibility

Every engine in this project already captures what reproduction needs (provenance, digests,
seeds, environment) and already offers its own `replay`. Phase 20's contribution
([reproducibility.md#advanced-reproducibility-framework-phase-20](#advanced-reproducibility-framework-phase-20))
is a single, honest classification layer over all of them — it distinguishes exact, deterministic,
tolerance-based, statistical, and provenance-only agreement instead of one binary "reproducible"
flag, and it never claims a guarantee the underlying platform (floating point, threading, hardware)
cannot actually make.

## Definitions

- **Repeatability:** same team, same setup, same code and environment, same results.
- **Reproducibility:** different team or environment, same method and artifacts, same results.
- **Replicability:** independent implementation or new data, same conclusion.

## Planned capture

Source revision; dependency versions; Python version; OS; hardware; model version; dataset
version; configuration; random seeds; execution parameters; generated artifacts; timestamps;
environment identity (hash of the above).

## Reproduction check (Phase 6)
`experionyx failure reproduce` replays a failure mode's supporting runs as new runs and compares
the reproduced signal magnitude to the original within configured tolerances, recording the
original result, reproduction result, tolerance, pass/fail and provenance as evidence. This
tests **repeatability in the current environment** of one recorded procedure, seed and input; it
is not independent reproduction or replication ([failures.md](failures.md)).

## Status after Phase 2
The execution engine now captures, per run: seed, configuration, environment (Python, OS,
architecture, dependency versions), git commit and cleanliness, executor version, artifact SHA-256
digests, and a provenance fingerprint ([provenance.md](provenance.md)). It can request a
**replay** as a new run. Still missing: any verification that results reproduce (REPRODUCTION),
hardware/accelerator capture, data and model-weight digests, deterministic-execution guarantees,
and storage of uncommitted source diffs. Repeatability, reproducibility and replicability are
therefore **not** yet established by this repository.

## Drift analyses (Phase 12)

A drift analysis is a Run. Its provenance fingerprint covers the dataset, model, baseline, declared
ordering, every window's identity and membership digest, skipped windows, features and their declared
types, methods, seed, correction and thresholds, slice identities and software versions
([drift.md](drift.md)). `experionyx drift replay` re-executes it as a new run and compares all six
stored documents; a document that no longer matches its digest is refused, not reported as a
difference. The permutation tests and bootstraps are seeded, and results do not depend on input order.

## Data quality analyses (Phase 13)

A quality analysis is a Run. Its provenance fingerprint covers the dataset fingerprint, adapter,
columns, per-split sample digests, the whole specification, check identities, slice membership digests
and software versions ([data-quality.md](data-quality.md)). `experionyx data-quality replay` re-executes it
as a new run and compares all five stored documents; a document that no longer matches its digest is
refused, not reported as a difference. Permutation tests and bootstraps are seeded and do not depend on
row or column order.

## Resource analyses (Phase 16)

Resource timing is **not** reproducible bit for bit and is never promised to be: it depends on the machine, the
clock, the scheduler, thermal state and other load. What reproduces is the **definition** (`spec.json`, exactly)
and, where the model is deterministic, the **model outputs** (a digest over the per-sample outputs, independent of
batching). `experionyx resources replay` re-executes the Run as a new Run and compares those two things; the
median trial time of both runs is shown for information only. A repeated measurement of one definition in one
environment is a new analysis with the same provenance fingerprint. Comparing measurements from different
environments is refused unless explicitly allowed and is then labelled as confounded. See
[resources.md](resources.md).

## Scheduled experiments (Phase 17)

A `Schedule` reproduces its own exact compact definition (every unit's kind, parameters, dependencies,
retry policy, resources and timeout); replaying it re-runs each unit through its own engine and compares
outcomes the same way that engine's own replay would. `RESOURCE` units are excluded from the pass/fail
verdict of `experionyx scheduler replay`, for the same reason resource timing is excluded above; every
other unit kind is a deterministic computation given the same seeds and evidence. See
[scheduler.md](scheduler.md).

## Graph construction (Phase 19)

`experionyx graph replay` reconstructs the original collect Run as a new Run and compares every
stored artifact (spec, nodes, edges, summary) byte-for-byte. If the registry's content changed
since the snapshot was collected, the source fingerprint differs and this is reported as
`sources_changed`, never as nondeterminism — replay cannot judge determinism when its inputs moved
between the two runs. See [graph.md](graph.md).

## Calibration analyses (Phase 15)

A calibration analysis is a Run over digest-verified stored predictions; it never loads or calls the model.
Every bootstrap and permutation uses its recorded seed, the calibration split is a seeded hash of the sample
ID (independent of row order), and both Platt (Newton iterations from a fixed start) and isotonic (pool
adjacent violators) fitting are deterministic, so the same stored data and spec reproduce identity, bins,
metrics, intervals, comparisons and candidate signals exactly (floats within `REPLAY_TOLERANCE`).
`experionyx calibration replay` re-executes the run and compares all eight documents; `calibration compare`
recomputes from the stored predictions and stores nothing. See [calibration.md](calibration.md).

## Stress analyses (Phase 14)

A stress analysis is a Run, and so is every trial. Its provenance fingerprint covers the whole design,
the baseline, each trial's outcome and artifact digests, slice membership and software versions
([stress.md](stress.md)). `experionyx stress replay` re-executes the analysis and every completed trial as
new runs and compares all seven documents, each trial's identity and its metric values within tolerance;
`stress compare` recomputes the analysis from the stored trials without storing anything. Seeded stress
uses only its recorded seed; deterministic stress takes none.

## Advanced reproducibility framework (Phase 20)

Every engine above already exposes its own `replay_check(reg, store, executor, id) ->
dict[str, object]` with the same shape (`original_run`, `replay_run`, `replay_status`,
`deterministic`, `sources_changed`, `differences`). `experionyx/reproducibility/` never
re-implements any of that; it dispatches to whichever `replay_check` the target's kind already
has, classifies the result against an explicitly requested mode, and persists one immutable
`ReproductionAttempt` (`rpa_...`). It never promises bit-for-bit reproduction the platform cannot
guarantee — see "Modes" below.

### Target kinds

A `ReproductionSpec` names a `TargetKind` (`RUN`, `BENCHMARK_RESULT`, `SCHEDULE`,
`STRESS_ANALYSIS`, `CALIBRATION_ANALYSIS`, `RESOURCE_ANALYSIS`, `DRIFT_ANALYSIS`,
`QUALITY_ANALYSIS`, `RELIABILITY_PROFILE`, `INTERACTION_ANALYSIS`, `GRAPH_SNAPSHOT`) and a target
ID. `RUN` is compared by artifact digest only (no fixed document schema exists for an arbitrary
run); every other kind is compared document-by-document using the exact list of documents that
kind's own engine already writes. `SCHEDULE` is different again: it re-runs the schedule through
`scheduler.engine.run_schedule` and diffs every unit's final status and `primary_ref`, excluding
`RESOURCE` units from the verdict for the same reason `scheduler replay` does
([scheduler.md](scheduler.md)). A fault experiment or interaction has no single replay-worthy
record of its own kind here; reproduce the `Run` it produced directly instead — a known limitation,
not an oversight.

### Modes

`ReproductionMode` is requested by the caller, never inferred:

- `EXACT` / `DETERMINISTIC` — the target kind's own `replay_check` verdict, unmodified. (The two
  are the same check today: every engine's own replay already excludes environment-only noise
  like timing before reporting `deterministic`.)
- `NUMERIC_TOLERANCE` — when the underlying replay reports a difference, every differing document
  is re-opened and compared field-by-field with `experionyx.reproducibility.compare` (a generic,
  recursive JSON walk: numbers within `relative_tolerance`/`absolute_tolerance` are
  `APPROXIMATELY_EQUAL`, a type mismatch at the same path is `INCOMPARABLE`, everything else that
  differs is `DIFFERENT`). This can only ever *soften* a byte-level difference into an
  approximately-equal one, never manufacture agreement that was not there.
- `STATISTICAL` — the same tolerance-aware re-comparison, offered as its own mode name so a caller
  can request "does this agree within noise" without conflating it with `EXACT`'s stricter
  contract; it does not (yet) reach into a stored bootstrap interval to test overlap specifically.
- `PROVENANCE_ONLY` — **nothing is re-executed.** Only the target's own recorded `Provenance` and
  stored artifact digests are checked (`reproduce.md`'s `verify`, under the hood). This is the one
  mode always available, even when the platform genuinely cannot guarantee re-execution will match
  (a different accelerator, an unavailable dependency): it answers "is the original evidence
  itself still intact," not "does it reproduce."

A `ReproductionAttempt.outcome` is one of `EQUAL`, `APPROXIMATELY_EQUAL`, `DIFFERENT`,
`UNAVAILABLE` (the replay failed, or the persisted evidence changed since collection —
`sources_changed`), or `INCOMPARABLE` (a tolerance re-comparison found two differently-shaped
values at the same path). There is no composite score and no single reproducible/non-reproducible
flag: `outcome`, `mode`, `sources_changed`, `differences`, `field_differences`, `environment_diff`
and `artifact_verification` are reported together, and a reader decides what they mean.

### Environment and artifact checks

Every non-`PROVENANCE_ONLY` attempt also compares the original and replay runs'
`Provenance`/`EnvironmentSnapshot` records field-by-field (Python version, OS, machine, package
versions, dependency digest, executor version, seed, source commit) and reports what changed —
`environment_diff`. If a difference is found, this is reported as data, never automatically
attributed to one cause: a changed package version and a changed metric are both just listed.
Every attempt also re-hashes the target's own stored artifacts (`artifact_verification`), the same
check `experionyx verify` performs, so a corrupted or missing artifact is caught even before a
replay is attempted.

### Replay safety

A reproduction attempt never mutates the target: `ReproductionAttempt` identity is
`(target_id, attempt)` (numbered 0, 1, 2, ... per target, like `scheduler.entities.ExecutionAttempt`),
so a repeated attempt adds a new immutable record and the first one is never overwritten. `reproduce
replay` re-attempts a prior attempt's exact recorded spec as a **new** attempt referencing the
prior one, never replacing it.

### Integration

- **Scheduler:** `UnitKind.REPRODUCTION` dispatches through `run_reproduction` like any other unit;
  a reproduction unit's `depends_on` is enforced by the scheduler's own dependency mechanism, so it
  cannot execute before the evidence it targets exists ([scheduler.md](scheduler.md)). It cannot
  target a `SCHEDULE` itself (no nested scheduling).
- **Graph:** `ReproductionAttempt` is in `graph.build.ENTITY_TYPES`, so it appears in a constructed
  graph automatically, with a `REPRODUCED_BY` edge to the target it reproduces and a `COMPARED_WITH`
  edge to the replay run it compared against — no reproducibility-specific traversal code exists in
  the graph module ([graph.md](graph.md)).
- **Benchmark:** `BENCHMARK_RESULT` is a first-class `TargetKind`; an unreproducible result is
  persisted with `outcome=DIFFERENT`/`UNAVAILABLE` and its full evidence, never silently dropped.

### CLI

`experionyx reproduce validate | run | inspect | compare | verify | replay | diff` (see
[cli.md](cli.md)). `validate` checks a target exists without reproducing anything. `run` exits 0
for `EQUAL`/`APPROXIMATELY_EQUAL`, 1 otherwise. `compare` is the one ad hoc command: a
tolerance-aware comparison of two stored JSON artifacts that need not come from the same run or
target, for exploratory use outside the attempt/classification machinery.

### Known limitations

- `NUMERIC_TOLERANCE`/`STATISTICAL` re-comparison only covers documents the target's own engine
  already persists as JSON; a `RUN` target (arbitrary, non-JSON artifacts) is always compared by
  digest only.
- `STATISTICAL` mode does not yet test stored confidence/bootstrap interval overlap specifically;
  it uses the same field-tolerance comparator as `NUMERIC_TOLERANCE`.
- A `SCHEDULE` reproduction re-runs the **entire** schedule; there is no unit-level reproduction
  scoped to one unit of an existing schedule.
- No `FAULT_EXPERIMENT` or generic analysis-independent `TargetKind` exists; reproduce the `Run`
  it produced directly.
