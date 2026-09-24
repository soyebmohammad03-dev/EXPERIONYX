# Reproducibility

**Target architecture. The repository does not yet capture or guarantee any of this.**

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
