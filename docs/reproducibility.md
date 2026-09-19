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
