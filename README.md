# EXPERIONYX

**AI Experimental Forensics & Reliability Laboratory**

> **Status: Phase 11 (slice analysis).** The typed domain model, a local SQLite registry,
> an execution engine that records provenance and artifact digests, framework-agnostic model and
> dataset adapters (concrete: scikit-learn and PyTorch), and a baseline evaluation engine
> (metrics, calibration, bootstrap intervals, slices, error records, rule-based findings) and a fault injection laboratory (typed, seeded faults; control vs
> treatment runs; degradation with direction-aware metrics) and a deterministic failure registry and discovery pipeline (signals, similarity, clustering, evidence criteria, reproduction check, lifecycle) exist. A rigorous fault interaction analysis engine (four-cell additive contrast, trial-level bootstrap, order effects, per-sample and failure-mode analysis) exists. Evidence-first reliability profiles summarize what has been observed about an evaluated system without a score or ranking. Versioned, deterministic robustness benchmarks with an explicit coverage account exist. Drift, general statistics and reports are
> not implemented yet. Everything below marked *planned* is an architecture target, not a feature.

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
- Baseline evaluation and autopsy ([docs/evaluation.md](docs/evaluation.md)): a metric registry,
  confidence/calibration, bootstrap intervals, class-imbalance and slice analysis, error records,
  latency, transparent rule-based findings linked to evidence, and structured comparison. It
  reports measurements and observations, not causes
  ([docs/observation-vs-conclusion.md](docs/observation-vs-conclusion.md)).
- Fault injection ([docs/faults.md](docs/faults.md)): a versioned fault registry (noise, dropout,
  missingness, scaling, occlusion, label corruption, compound faults), deterministic seeding and
  scopes, real parameter sweeps and repeated seeds against a baseline control, degradation
  measurement, transparent effect thresholds, and evidence links. It measures effects; it does not
  declare models failed.
- Failure registry and discovery ([docs/failures.md](docs/failures.md)): normalized failure signals
  extracted from stored evaluations and fault experiments, interpretable similarity with reasons,
  deterministic clustering with stability diagnostics, configurable evidence criteria, a
  reproduction check, and a validated lifecycle (`DISCOVERED` to `CONFIRMED`) in which nothing is
  confirmed automatically. It groups measured signals; it makes no causal claim and no LLM is
  involved.
- Fault interaction analysis ([docs/interactions.md](docs/interactions.md)): a validated four-cell
  design (control, A, B, A+B, optionally B+A), the additive interaction contrast with every
  intermediate value, repeated trials with explicit pairing, a seeded bootstrap over trials, order
  effects, verified per-sample alignment, failure-mode comparison, a lifecycle needing an
  independent replicate and a human decision, and deterministic replay. It reports observed
  contrasts; it never claims that one fault causes another.
- Reliability profiles ([docs/reliability.md](docs/reliability.md)): an evidence-first summary of one
  evaluated model/dataset/configuration across ten dimensions (baseline, fault, failure prevalence and
  severity, interaction, slice, reproducibility, uncertainty, latency, calibration), each with an explicit
  status and a source reference for every observation. Incompatible sources are refused, profiles compare
  by raw differences, and there is no score, ranking or verdict.
- Robustness benchmarks ([docs/benchmarks.md](docs/benchmarks.md)): a versioned protocol (fault grids,
  seeds, interaction pairs, analysis settings) expanded into explicit experiment units, executed through
  the fault laboratory, failure discovery, interaction analysis and reliability profiles, and reported as
  raw evidence plus a coverage account of what did and did not run. Two results compare only under an
  identical protocol, and there is no score, ranking or verdict.
- Statistical analysis ([docs/statistics.md](docs/statistics.md)): deterministic bootstrap intervals
  (percentile and BCa), paired and unpaired comparison with effect sizes and exact or seeded permutation
  tests, explicit multiple-comparison correction, Wilson intervals, and reproducible, registered analyses
  over persisted evidence. Statuses are explicit; a p-value is never a claim.
- Slice and subgroup analysis ([docs/slices.md](docs/slices.md)): typed, deterministic slice definitions,
  three-valued membership that never treats missing metadata as a match, per-slice metrics with uncertainty,
  explicit comparisons, and per-slice fault, failure-mode and interaction evidence. No group is ranked and no
  fairness score is produced.
- CLI over a real workspace: `status`, `execute`, `replay`, `run`, `provenance`, `verify`, ...
  ([docs/cli.md](docs/cli.md))
- Tests, Ruff, strict mypy, and a GitHub Actions workflow (not yet run on GitHub)

## Hardware philosophy

Laptop-first (developed on an 8 GB Apple Silicon machine): CPU-first, small public datasets,
resumable and bounded-parallel execution. No paid APIs, GPUs or clusters in the scientific core.

## Development

Requires Python 3.11+. The core has no runtime dependencies; frameworks are extras:
`pip install -e ".[sklearn]"`, `".[torch]"`, `".[faults]"` (numpy, for fault injection); add `dev` for the test tools.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,sklearn,torch,faults]"
pytest && ruff check . && ruff format --check . && mypy
```

Details: [docs/development.md](docs/development.md), [CONTRIBUTING.md](CONTRIBUTING.md).

## Roadmap

Provisional; see [docs/roadmap.md](docs/roadmap.md). Research directions are drafts:
[docs/research-questions.md](docs/research-questions.md).

## License

MIT. See [LICENSE](LICENSE).
