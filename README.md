# EXPERIONYX

**AI Experimental Forensics & Reliability Laboratory**

EXPERIONYX is an AI experimental forensics and reliability laboratory for controlled
experimentation, failure discovery, statistical analysis, reproducibility, provenance, and
evidence-backed research reporting.

> **Status: Phase 23–24 (visualization/analysis API+UI and final release hardening) complete.**
> Every subsystem listed under "What exists today" below is implemented, tested, and reachable
> both from the CLI and from a read-only HTTP API + small vanilla-JS UI
> (`experionyx viz serve`, see [docs/visualization.md](docs/visualization.md) and
> [docs/api.md](docs/api.md)). See [docs/roadmap.md](docs/roadmap.md) for the phase-by-phase
> history and [docs/limitations.md](docs/limitations.md) for what is genuinely still bounded.

## Problem

AI/ML results are routinely reported from a single seed, a single run and an unrecorded
environment. Failure modes are found anecdotally, reproducibility is assumed rather than measured,
and conclusions are rarely traceable to observations. EXPERIONYX aims to be a laboratory for
investigating the behavior, failure modes, reproducibility, provenance and longitudinal
reliability of AI/ML systems, with every conclusion traceable to real measurements.

## The research lifecycle

```
register model/dataset → run baseline → apply controlled fault
  → discover/analyze failure → statistical analysis → reliability evidence
  → inspect graph → reproducibility check → benchmark (optional)
  → generate report → build evidence dossier → immutable snapshot
  → inspect via API/UI → export Markdown
```

`examples/end_to_end_workflow.py` runs this exact sequence against real (small, deterministic)
data — no mocks — and prints the API/UI paths a researcher would open to inspect each result:

```bash
pip install -e ".[dev,sklearn,faults,viz]"
python examples/end_to_end_workflow.py
experionyx --workspace .experionyx-example viz serve   # then open http://127.0.0.1:8420/
```

## What EXPERIONYX does not claim to solve

- It does not decide whether one model is "better" than another. Benchmarks and reliability
  profiles never produce a composite score, ranking, or "best model" verdict — see
  [leaderboard.md](docs/leaderboard.md) and [reliability.md](docs/reliability.md).
  Comparisons are protocol-specific, metric-specific, and always show their own scope.
- It does not infer causation from an observational comparison, a fault effect, or a failure
  cluster — see [observation-vs-conclusion.md](docs/observation-vs-conclusion.md). A finding is a
  statement about what was measured, never a claim about why.
- It does not treat missing or unavailable evidence as a negative result; unavailable is a
  distinct, explicit status everywhere in the system, never a `0` or a silent omission.
- It does not promise bit-for-bit reproducibility the underlying platform cannot guarantee, and it
  does not measure portable, cross-machine resource/timing numbers — see
  [reproducibility.md](docs/reproducibility.md) and [resources.md](docs/resources.md).
- It is not a GPU-scale or cluster-scale system: it is laptop-first, CPU-first, and built around
  small public datasets. See [docs/limitations.md](docs/limitations.md) for the complete list.

## Philosophy

Evidence over assertion; reproducibility over convenience; negative results are valid;
no fabricated evidence; statistical humility; explicit uncertainty; resource awareness.
See [docs/methodology.md](docs/methodology.md).

## Architecture

```
MODEL → BASELINE → AUTOMATED AUTOPSY → CONTROLLED FAULT INJECTION → FAILURE DISCOVERY
→ EXPERIMENT TRACE → REPRODUCIBILITY ANALYSIS → LONGITUDINAL DRIFT ANALYSIS
→ STATISTICAL EVIDENCE → RELIABILITY ANALYSIS → EVIDENCE-BACKED DOSSIER → VISUALIZATION/API
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
- Temporal and distribution shift ([docs/drift.md](docs/drift.md)): explicitly ordered, reproducible windows;
  covariate (per feature), label, prediction and performance differences with uncertainty and Phase 10
  multiple-comparison correction; slice-aware; evidence links to failure modes, reliability profiles and
  benchmarks. No drift score, no causal or concept-drift claim.
- Data quality laboratory ([docs/data-quality.md](docs/data-quality.md)): a versioned, deterministic quality
  specification and typed checks (schema, missingness, duplicates and identifiers, numeric and categorical
  validity, target quality, leakage indicators, split and temporal checks, group comparisons) with explicit
  statuses, exact affected rows, Phase 10 statistics, optional slices and windows, and replay. No quality
  score; warnings are evidence, not verdicts.
- Model stress laboratory ([docs/stress.md](docs/stress.md)): controlled stress of inputs (reusing the Fault
  Laboratory), parameters, decision thresholds and evaluation conditions, expanded into explicit
  replayable trials against a validated baseline, with paired Phase 10 statistics, slices, failure
  discovery and interaction analysis. No robustness score, no causal claim.
- Calibration & uncertainty laboratory ([docs/calibration.md](docs/calibration.md)): whether a model's
  probabilities correspond to observed correctness (top-label and classwise reliability, ECE/MCE, Brier,
  log loss), post-hoc Platt/isotonic calibration with a strict calibration/evaluation separation,
  statistical uncertainty of the metrics (Wilson and seeded bootstrap), descriptive predictive dispersion,
  and explicit slice, window and stress contexts, all replayable. Confidence is not automatically
  uncertainty; unsupported uncertainty is reported unavailable; no universal calibration or uncertainty score.
- Resource & systems reliability laboratory ([docs/resources.md](docs/resources.md)): real measurements of
  latency, throughput, batch-size and worker-count behaviour, timeouts and failures, process CPU time and
  peak memory, from repeated warmed-up trials with raw timings preserved, Phase 10 intervals and corrected
  comparisons, provenance and replay of the definition and the model outputs. Every number is an
  environment-specific engineering measurement; unsupported measurements are reported unavailable; no
  composite resource score, and performance measurement is not reliability.
- Experiment scheduler & orchestration ([docs/scheduler.md](docs/scheduler.md)): a durable, resumable,
  dependency-aware orchestration layer over every existing engine (fault injection, failure discovery,
  interaction analysis, benchmarks, statistical analysis, slices, drift, data quality, stress, calibration,
  resources) via a thin dispatcher that never duplicates their logic. Compact specs expand deterministically
  into an explicit, validated dependency graph with deterministic identity, ordering and a real state
  machine; retries create new attempts without overwriting earlier ones; a resumed run never re-dispatches
  already-succeeded work; concurrency is bounded and deterministic; dry-run executes zero experiments; every
  plan, dispatch, retry, skip, block and outcome is recorded and auditable. No hidden work, no fabricated
  resource guarantees, no forced kill of a running dispatch.
- Evidence and failure knowledge graph ([docs/graph.md](docs/graph.md)): a deterministic, queryable graph
  connecting every existing entity (models, datasets, splits, experiments, runs, faults, stresses, temporal
  windows, data quality, calibration, resources, failure signals/clusters/modes, statistics, interactions,
  reliability profiles, benchmarks, claims, evidence, artifacts, scheduler units) through a closed, versioned
  relationship vocabulary derived generically from the registry's own references. Edges are
  provenance/evidence metadata, never an inferred causal claim; unresolved references are kept as explicit
  nodes, never dropped; construction is a real, replayable Run; every traversal is bounded and reports
  truncation. No universal graph or reliability score.
- Advanced reproducibility framework ([docs/reproducibility.md](docs/reproducibility.md)): a classification
  layer over every engine's own `replay_check`, distinguishing EXACT/DETERMINISTIC/NUMERIC_TOLERANCE/
  STATISTICAL/PROVENANCE_ONLY agreement (never one binary "reproducible" flag), with environment and
  artifact-integrity checks, scheduler and graph integration (a `REPRODUCTION` unit kind; a
  `ReproductionAttempt` node connected by `REPRODUCED_BY`/`COMPARED_WITH` edges). Never promises
  bit-for-bit reproduction the platform cannot guarantee; a reproduction attempt never mutates its target.
- Benchmark protocol & leaderboard system ([docs/leaderboard.md](docs/leaderboard.md)): a reporting/
  organization layer over the existing Phase 9 benchmark engine — model-independent protocol identity
  (the real expanded `protocol_hash`), provenance-linked submissions (never anonymous), immutable
  leaderboard snapshots, protocol-constrained metric-specific comparisons, explicit multiple-comparison
  correction (reusing Phase 10), reproducibility-state and resource-context reporting kept separate from
  predictive metrics. No universal score, no "best model" verdict, no combined ranking.
- Research reporting ([docs/reporting.md](docs/reporting.md)): versioned, deterministic reports
  assembled entirely from evidence every prior engine already persisted — it never re-executes an
  experiment or recomputes a statistic. Claims cannot exist without a resolvable evidence
  reference; a section without supporting evidence is recorded as an explicit gap rather than
  omitted or invented; statistical results are rendered verbatim (effect size, interval, sample
  size, method, p-value, correction) never reinterpreted; templates are versioned so an old
  report stays reproducible after a template changes; Markdown is the canonical export, with a
  thin derived HTML export.
- Evidence dossiers ([docs/dossier.md](docs/dossier.md)): structured, persisted research packages
  layered on reporting and the evidence graph — deterministic construction from an investigation,
  report, run, failure mode, benchmark result, reliability profile, or graph snapshot; an explicit
  evidence-sufficiency analysis (missing/unavailable/conflicting evidence, provenance and
  reproducibility gaps — never hidden); conflicting statistical evidence is preserved and
  cross-referenced, never auto-resolved; immutable, content-addressed snapshots that a later
  experiment cannot silently alter.
- Visualization / analysis API+UI ([docs/visualization.md](docs/visualization.md),
  [docs/api.md](docs/api.md)): a read-only FastAPI surface plus a small hand-written vanilla-JS SPA
  over every engine above's own read APIs, served with `experionyx viz serve`. Every route calls
  an existing `get`/`find`/`search`/`compare`/`document` method — no analysis is recomputed here,
  no aggregate reliability score or "best model" verdict is ever introduced, missing evidence is
  always an explicit "unavailable", and every list/graph view is bounded and paginated.
- CLI over a real workspace: `status`, `execute`, `replay`, `run`, `provenance`, `verify`, ...
  ([docs/cli.md](docs/cli.md))
- Tests, Ruff, strict mypy, and a GitHub Actions CI workflow (Python 3.11 and 3.12)

## Hardware philosophy

Laptop-first (developed on an 8 GB Apple Silicon machine): CPU-first, small public datasets,
resumable and bounded-parallel execution. No paid APIs, GPUs or clusters in the scientific core.

## Development

Requires Python 3.11+. The core has no runtime dependencies; frameworks are extras:
`pip install -e ".[sklearn]"`, `".[torch]"`, `".[faults]"` (numpy, for fault injection),
`".[viz]"` (FastAPI + uvicorn, for `experionyx viz serve`); add `dev` for the test tools.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,sklearn,torch,faults,viz]"
pytest && ruff check . && ruff format --check . && mypy
```

Details: [docs/development.md](docs/development.md), [CONTRIBUTING.md](CONTRIBUTING.md).

## Roadmap

Provisional; see [docs/roadmap.md](docs/roadmap.md). Research directions are drafts:
[docs/research-questions.md](docs/research-questions.md).

## License

MIT. See [LICENSE](LICENSE).
