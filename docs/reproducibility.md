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

## Status after Phase 2
The execution engine now captures, per run: seed, configuration, environment (Python, OS,
architecture, dependency versions), git commit and cleanliness, executor version, artifact SHA-256
digests, and a provenance fingerprint ([provenance.md](provenance.md)). It can request a
**replay** as a new run. Still missing: any verification that results reproduce (REPRODUCTION),
hardware/accelerator capture, data and model-weight digests, deterministic-execution guarantees,
and storage of uncommitted source diffs. Repeatability, reproducibility and replicability are
therefore **not** yet established by this repository.
