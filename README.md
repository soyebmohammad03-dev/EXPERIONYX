# EXPERIONYX

**AI Experimental Forensics & Reliability Laboratory**

> **Status: Phase 3 (model + dataset adapters).** The typed domain model, a local SQLite registry,
> an execution engine that records provenance and artifact digests, and framework-agnostic model and
> dataset adapters (concrete: scikit-learn and PyTorch) exist. No laboratory capability (autopsy, fault injection, drift, statistics, reports) is
> implemented yet. Everything below marked *planned* is an architecture target, not a feature.

## Problem

AI/ML results are routinely reported from a single seed, a single run and an unrecorded
environment. Failure modes are found anecdotally, reproducibility is assumed rather than measured,
and conclusions are rarely traceable to observations. EXPERIONYX aims to be a laboratory for
investigating the behavior, failure modes, reproducibility, provenance and longitudinal
reliability of AI/ML systems, with every conclusion traceable to real measurements.

## Philosophy

Evidence over assertion; reproducibility over convenience; negative results are valid;
no fabricated evidence; statistical humility; explicit uncertainty; resource awareness.
See [docs/methodology.md](docs/methodology.md).

## Architecture target (planned)

```
MODEL → BASELINE → AUTOMATED AUTOPSY → CONTROLLED FAULT INJECTION → FAILURE DISCOVERY
→ EXPERIMENT TRACE → REPRODUCIBILITY ANALYSIS → LONGITUDINAL DRIFT ANALYSIS
→ STATISTICAL EVIDENCE → RELIABILITY ANALYSIS → EVIDENCE-BACKED DOSSIER
```

See [docs/architecture.md](docs/architecture.md) and
[docs/experiment-lifecycle.md](docs/experiment-lifecycle.md).

## What exists today

- `src/` layout Python package `experionyx` (zero runtime dependencies)
- `experionyx --version` and `experionyx info` (report real package/environment facts)
- Immutable, validated domain model: Investigation, Experiment, Run, Observation, Artifact,
  Claim, Evidence, and content-addressed configuration/environment records
  ([docs/domain-model.md](docs/domain-model.md))
- Deterministic canonical hashing and IDs ([docs/identity.md](docs/identity.md))
- Append-only registry protocol with a SQLite backend and transactions
  ([docs/registry.md](docs/registry.md))
- Execution engine ([docs/execution.md](docs/execution.md)): runs a Python procedure as a traced
  Run, captures environment/source/seed/configuration ([docs/provenance.md](docs/provenance.md)),
  hashes artifacts ([docs/artifacts.md](docs/artifacts.md)), records failures, and can request
  replays as new runs. Replay is not reproduction verification.
- Adapter layer ([docs/adapters.md](docs/adapters.md)): explicit typed capabilities, model/dataset
  identity and fingerprints ([docs/model-dataset-identity.md](docs/model-dataset-identity.md)),
  registered models/datasets verified before every run and recorded in provenance, contract
  tests every adapter must pass. sklearn and PyTorch are the *initial* integrations, optional
  extras; the core imports neither.
- CLI over a real workspace: `status`, `execute`, `replay`, `run`, `provenance`, `verify`, ...
  ([docs/cli.md](docs/cli.md))
- Tests, Ruff, strict mypy, and a GitHub Actions workflow (not yet run on GitHub)

## Hardware philosophy

Laptop-first (developed on an 8 GB Apple Silicon machine): CPU-first, small public datasets,
resumable and bounded-parallel execution. No paid APIs, GPUs or clusters in the scientific core.

## Development

Requires Python 3.11+. The core has no runtime dependencies; frameworks are extras:
`pip install -e ".[sklearn]"`, `".[torch]"` (add `dev` for the test tools).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,sklearn,torch]"
pytest && ruff check . && ruff format --check . && mypy
```

Details: [docs/development.md](docs/development.md), [CONTRIBUTING.md](CONTRIBUTING.md).

## Roadmap

Provisional; see [docs/roadmap.md](docs/roadmap.md). Research directions are drafts:
[docs/research-questions.md](docs/research-questions.md).

## License

MIT. See [LICENSE](LICENSE).
