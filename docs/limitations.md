# Known limitations

EXPERIONYX makes narrow, checkable claims on purpose (see
[observation-vs-conclusion.md](observation-vs-conclusion.md)). This page collects the genuine
boundaries of what it can measure or guarantee, gathered from the per-subsystem docs rather than
restated as marketing copy. If a limitation below is ever silently removed from a subsystem's own
doc, that doc is the source of truth, not this page.

## Models and datasets

- Adapters are explicit and typed ([adapters.md](adapters.md)); the two concrete integrations
  today are scikit-learn and PyTorch, both optional extras — the core imports neither. A model or
  dataset with no adapter cannot be evaluated.
- Calibration analysis needs real per-class probability scores; classification with hard labels
  only is refused (`UNAVAILABLE`), never approximated from scores that don't exist
  ([calibration.md](calibration.md)).
- Stress and fault analyses that need `ParameterAccess` (parameter-level perturbation) degrade to
  `UNAVAILABLE` for adapters that don't expose floating-point state — restoration afterward is
  exact, but the perturbation itself is never approximated.

## Packaging and CLI

- The importable library (domain, registry, graph, reporting) has no runtime dependencies. The
  `experionyx` CLI wires every subsystem into one dispatcher at import time, so `pip install
  experionyx` with no extras fails on the first command — install at least `.[faults]` (numpy),
  which nearly every command needs transitively even when it doesn't touch fault injection
  directly. This is a packaging rough edge in the CLI's import structure, not a hidden runtime
  dependency of the library itself.

## Execution environment

- Laptop-first: developed and validated on an 8 GB Apple Silicon machine, CPU-first, small public
  datasets, resumable and bounded-parallel execution. No paid APIs, GPUs, or clusters in the
  scientific core (README, "Hardware philosophy").
- Resource/timing measurements are environment-specific engineering numbers, not portable
  reliability claims: GPU memory, enforced OS-level resource limits, and cross-machine variability
  are not measured ([resources.md](resources.md)).
- The scheduler's concurrency is threads-only; a timeout cannot force-kill a unit already running,
  and "unlimited concurrency" is a real value, not an OS guarantee it enforces
  ([scheduler.md](scheduler.md)).

## Statistics

- Every statistical procedure reports an explicit status (`UNDEFINED`, `UNAVAILABLE`,
  `INSUFFICIENT_EVIDENCE`, ...) rather than silently falling back to an approximation; a p-value or
  interval that cannot be computed for the given input is never manufactured
  ([statistics.md](statistics.md)).
- BCa intervals for interaction contrasts and control-variability modelling remain open work
  (roadmap item 10).
- A statistically significant difference is never rendered as a causal or "better model" claim
  anywhere in the codebase, including the API/UI layer.

## Drift, calibration, stress, data quality

- Drift analysis operates on one evaluated split of one baseline run; it does not ingest a
  streaming feed, and windows/slices are never enumerated automatically — they must be declared
  ([drift.md](drift.md)).
- Calibration measures whether stored probabilities matched observed correctness; it never claims
  epistemic/aleatoric uncertainty decomposition, and leaky calibration setups are refused outright,
  not merely flagged ([calibration.md](calibration.md)).
- Stress analysis perturbs a registered dataset's view, decision threshold, or evaluation
  condition — the registered data and model are never modified; an incomplete stress design is
  refused and recorded, never silently skipped ([stress.md](stress.md)).
- Data quality checks report explicit statuses and exact affected rows; there is no aggregate
  quality score, and a warning is evidence, not a verdict ([data-quality.md](data-quality.md)).

## Reproducibility

- Reproduction is classified (`EXACT`/`DETERMINISTIC`/`NUMERIC_TOLERANCE`/`STATISTICAL`/
  `PROVENANCE_ONLY`), never collapsed into one binary "reproducible" flag, and never claims a
  guarantee the underlying platform (floating point, threading, hardware) cannot make
  ([reproducibility.md](reproducibility.md)).
- Resource/timing is explicitly **not** reproducible bit-for-bit and is never promised to be.
- `STATISTICAL` reproduction mode does not yet test a stored bootstrap/confidence interval for
  overlap specifically — it uses the contract described in `reproducibility.md`, nothing stronger.

## Benchmark and leaderboard

- Two benchmark results compare only under an identical protocol; a protocol mismatch is a
  rejection (`422` at the API layer), never a best-effort comparison
  ([leaderboard.md](leaderboard.md)).
- There is no universal score, ranking, or "best model" verdict anywhere in the benchmark/
  leaderboard system, at any layer including the UI.

## Graph and knowledge graph scale

- The graph is a derived read-side layer, rebuilt from the registry's own references — it is not
  an independently maintained graph database, and it carries no graph-wide reliability or trust
  score ([graph.md](graph.md)).
- Every traversal (neighbors/ancestors/descendants/path) is bounded by `max_depth`/`max_visited`
  and reports `truncated` when the bound was hit; there is no unbounded "load the whole graph"
  operation, in the engine or the API/UI.

## UI / API scale and trust boundary

- List endpoints are paginated (default page 50, hard cap 500); very large investigations must be
  narrowed with filters rather than browsed unbounded (see [api.md](api.md)).
- `SqliteRegistry.find` itself has no server-side `LIMIT` — the API bounds what is *returned* to
  the caller, not the size of the intermediate query against a single already-scoped relationship
  (e.g. "runs of this one experiment"). For a registry with an extreme number of rows under one
  parent id, that intermediate fetch is the current ceiling; narrower filtering at the registry
  layer would be the next step if this ever matters in practice.
- The API/UI has no authentication or authorization layer. It is meant for the same trust boundary
  as running the CLI directly against a workspace you already control — do not expose
  `experionyx viz serve` on an untrusted network without a reverse proxy in front of it.

## Evidence and inference boundaries

- No engine ever infers negative evidence from absent evidence: missing/unavailable data is always
  an explicit status, never treated as "this failed" or rendered as a plain `0`
  ([observation-vs-conclusion.md](observation-vs-conclusion.md)).
- Reports and dossiers assemble only evidence prior engines already persisted; they never
  re-execute an experiment or recompute a statistic, and a claim without a resolvable evidence
  reference cannot exist ([reporting.md](reporting.md), [dossier.md](dossier.md)).
