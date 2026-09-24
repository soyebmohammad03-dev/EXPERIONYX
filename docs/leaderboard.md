# Benchmark protocol and leaderboard (Phase 20)

A leaderboard is a **reporting/organization mechanism, not a universal scientific ranking**. It
never produces an overall score, a composite quality number, or a "best model" verdict. Every
comparison stays protocol-specific (only submissions to the *identical* protocol are compared) and
metric-specific (each metric is reported on its own, with its own direction and uncertainty,
never averaged with another). `experionyx/leaderboard/` is a thin layer over the existing Phase 9
benchmark engine ([benchmarks.md](benchmarks.md)) and Phase 10 statistics
([statistics.md](statistics.md)); it never re-implements benchmark expansion, execution, or
significance testing.

```
BenchmarkSpec (per model) --expand()--> protocol_hash --> BenchmarkProtocol (model-independent)
                                                                |
                       submit() --> run_benchmark() --> BenchmarkResult --> BenchmarkSubmission
                                                                |
                                              build_snapshot() --> LeaderboardSnapshot + LeaderboardEntry (one per submission)
```

## Protocol identity is model-independent

A `BenchmarkSpec` already has a `protocol_key()` (everything except `model`) and, once expanded,
an exact `protocol_hash` covering the resolved fault versions, the expanded unit list and
interaction pairs -- the same identity `benchmark.protocol.expand` already computes for every
`Benchmark`/`BenchmarkResult` ([benchmarks.md](benchmarks.md), "Versioning and comparison").
`BenchmarkProtocol` (`bpr_`) is that identity promoted to its own registry row:
`leaderboard.protocol.register_protocol` calls the real `expand()` (the same validation
`benchmark validate`/`benchmark run` perform -- refused, with every issue, before anything is
registered) and stores the resulting hash. A protocol's identity therefore always matches what a
real submission's `BenchmarkResult.protocol_hash` will be; the two are never computed by two
different code paths.

Protocol identity covers: dataset, split (via the evaluation config), model task (via the
metric/adapter compatibility `expand` checks), metrics, fault/stress conditions, slices,
statistical configuration (aggregation confidence/resamples/seed, interaction config),
resource policy (`ResourceSpec`s), calibration/drift requirements, evaluation configuration,
seeds, the benchmark version, and the engine version. It deliberately excludes the model.

## Submissions are provenance-linked, never anonymous

`leaderboard.submission.submit` takes a full `BenchmarkSpec` (including the model) and a target
`BenchmarkProtocol`. It:

1. Computes the spec's real protocol hash (`protocol_hash_of`, the same `expand()` call) and
   refuses with a `ValidationError` if it does not match the target protocol -- before executing
   anything.
2. Calls `benchmark.engine.run_benchmark` directly -- the exact entry point `experionyx benchmark
   run` uses. No benchmark expansion or execution logic is duplicated here.
3. Double-checks the resulting `BenchmarkResult.protocol_hash` still matches (defense in depth:
   `expand()` is deterministic, so this should never fire, but a submission is never recorded
   against evidence that turned out not to match).
4. Registers a `BenchmarkSubmission` (`bsb_`) referencing the real `Benchmark` and `BenchmarkResult`
   -- never copying their data. Identity is `(protocol_id, result_id)`: submitting the same
   evidence twice is idempotent, never duplicated.

A submission always carries the model's registered ID, its real provenance-fingerprinted
`BenchmarkResult`, and (transitively, through that result's `run_id`) the model and dataset
fingerprints, environment, and dependency digest ([provenance.md](provenance.md)). There is no path
to an anonymous or untraceable leaderboard result.

## Leaderboard snapshots are immutable and idempotent

`leaderboard.snapshot.build_snapshot` reads every `BenchmarkSubmission` to a protocol, applies
explicit filtering rules (`require_complete_coverage`) and evidence requirements
(`require_reproduction`), and persists one `LeaderboardSnapshot` (`lbs_`) plus one
`LeaderboardEntry` (`lbe_`) per included submission. A submission excluded by a rule is recorded
in `excluded` with its reason -- never silently dropped. Identity is `(protocol_id,
source_fingerprint)`, where `source_fingerprint` hashes every included submission's ID and its
result's own content hash, the metric selection, and the rules -- the same idempotent-construction
convention as `GraphSnapshot` ([graph.md](graph.md)): rebuilding from unchanged evidence returns
the existing snapshot, never a duplicate. `experionyx leaderboard replay` rebuilds from current
evidence and reports whether the identity is still the same.

Each `LeaderboardEntry.metrics` is read directly from its submission's stored
`benchmark/results.json` (`results.baseline.metrics`) -- the exact per-metric `{value, status,
higher_is_better, interval}` documents `benchmark.collect` already writes from the real evaluation
engine ([evaluation.md](evaluation.md)). Nothing is recomputed, and there is no `score` or
`overall` key anywhere in an entry.

## Metrics: explicit, never combined

Every metric an entry reports already carries, from `evaluation.metrics.MetricSpec` (Phase 4):
`name`, `higher_is_better` (direction), a `description` (definition), `scale`, an aggregation
(the evaluation engine's own bootstrap/Wilson interval), and an explicit `UNDEFINED` status with a
reason when a metric could not be computed for that model/dataset -- never silently omitted.
Metrics of different families (accuracy, calibration ECE, fault deterioration) are never averaged
or combined into a single number.

## Resource context stays separate

If a submission's benchmark requested resource measurements (`BenchmarkSpec.resources`), its
`resource_analysis` block (latency, throughput, memory/CPU where available, timeout/failure rate --
see [resources.md](resources.md)) is carried into `LeaderboardEntry.resource_context` as its own
field, never merged into `metrics`. Resource numbers are environment-specific engineering
measurements, not predictive/robustness evidence, and comparing them across submissions is subject
to the same confounding caveats as `experionyx resources compare`.

## Reproducibility state is an evidence state, not a quality rating

`LeaderboardEntry.reproducibility_state` is one of `REPRODUCED`, `PARTIALLY_REPRODUCED`,
`NOT_REPRODUCED`, `UNAVAILABLE` -- derived from the most recent `ReproductionAttempt`
([reproducibility.md](reproducibility.md)) targeting the submission's `BenchmarkResult`, if any
(`EQUAL` -> `REPRODUCED`, `APPROXIMATELY_EQUAL` -> `PARTIALLY_REPRODUCED`, `DIFFERENT`/
`INCOMPARABLE` -> `NOT_REPRODUCED`). No attempt at all is `UNAVAILABLE`, never assumed reproduced.
Nothing here ranks submissions by this state; `require_reproduction` only filters which
submissions a snapshot includes, when explicitly requested.

## Comparison: protocol-constrained, no winner

`leaderboard.compare.compare_submissions` wraps `benchmark.registry.BenchmarkRegistry.compare`
directly ([benchmarks.md](benchmarks.md), "Versioning and comparison") -- the exact function
`benchmark compare` already uses, which refuses (`BenchmarkRefusal`, `PROTOCOL_DIFFERS`) two
results of different protocols before computing anything. Its output is raw per-metric, per-grid,
per-pair differences (B minus A), uncertainty availability, and reproducibility status; there is no
winner, ranking, or verdict field anywhere in it.

### Multiple-comparison correction: reused, never silent

`leaderboard.compare.correct_family` reuses `stats.core.adjust_pvalues` end to end via the
existing `stats.store` `CORRECTION` analysis kind (its `{"kind": "analyses", "ids": [...]}` source,
see [statistics.md](statistics.md)) over an **explicitly named family** of already-registered
`COMPARE` analyses (built with `experionyx stats compare`, typically one per pairwise submission
comparison on a chosen metric). `method` defaults to `NONE`; `BONFERRONI` and `BENJAMINI_HOCHBERG`
are the other two supported methods, matching `stats.core.CORRECTIONS` exactly. Nothing here
computes a p-value itself -- the caller supplies real, already-computed analyses, and the family is
always exactly what the caller named, never inferred.

## Evidence graph integration

Every leaderboard entity is registered like any other and participates in Phase 17-18's graph
generically: a `BenchmarkSubmission`'s `result_id`/`benchmark_id`/`protocol_id` fields become
`DERIVED_FROM`/`MEMBER_OF` edges to its real `BenchmarkResult`/`Benchmark`/`BenchmarkProtocol`
nodes, and (through the result's own `run_id`) all the way to the real collect `Run` and its
artifacts, exactly the same generic field-walk that already connects every other entity kind (see
[graph.md](graph.md)). No new graph traversal code exists for the leaderboard layer.

## CLI

```
experionyx benchmark protocol SPEC.json [--format text]
experionyx benchmark submit   SPEC.json --protocol bpr_... [--format text]

experionyx leaderboard list
experionyx leaderboard inspect  bpr_...|bsb_...|lbs_...|lbe_...
experionyx leaderboard snapshot bpr_... [--metric M ... --require-complete-coverage --require-reproduction]
experionyx leaderboard compare  bsb_... bsb_...
experionyx leaderboard correct  sta_... sta_... [--method NONE|BONFERRONI|BENJAMINI_HOCHBERG --alpha A]
experionyx leaderboard verify   bsb_...
experionyx leaderboard replay   lbs_...
```

`benchmark submit` exits 2 (a clean refusal, nothing recorded) if the spec does not match the
protocol, or if the benchmark itself is refused. `leaderboard compare` exits 2 if the two
submissions are not protocol-comparable. `leaderboard verify`/`replay` follow the same exit
conventions as `reproduce verify`/`graph replay`.

## Known limitations

- Protocol registration calls the real `expand()`, which needs the model already declared in
  the spec even though the model plays no role in the resulting identity; there is no
  model-free way to validate a protocol shape before at least one real spec exists.
- Multiple-comparison correction requires the caller to build the individual `COMPARE` analyses
  first (via `experionyx stats compare`); `leaderboard correct` does not derive per-sample
  significance from stored evaluation predictions on its own.
- `LeaderboardEntry` metrics are exactly whatever the underlying `BenchmarkResult`'s baseline
  evaluation computed; a metric unavailable for a given model/dataset combination is reported
  `UNDEFINED` with its reason, never fabricated.
- There is no per-metric ranking or ordering exposed anywhere in the persisted schema; sorting, if
  a caller wants it, is a presentation-layer concern over the raw entry values, never a stored
  verdict.
