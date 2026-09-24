# Evidence and failure knowledge graph (Phase 19)

The graph answers one question: **given everything already in the registry, how does it connect?**
It is a read-side layer over the existing entities (models, datasets, splits, experiments, runs,
faults, stresses, temporal windows, data-quality analyses, calibration analyses, resource
analyses, failure signals/clusters/modes, statistical analyses, interactions, reliability
profiles, benchmarks, claims, evidence, artifacts, scheduler units, ...). It never duplicates a
domain entity's data and never re-derives a science result; it only records and traverses the
references those entities already carry.

```
Registry (existing entities) -> Graph Construction -> GraphSnapshot (nodes + edges)
```

## Not a causality engine

**An edge is provenance/evidence metadata, not an inferred causal claim.** If a `StressAnalysis`
references a `Run`, the graph records `DERIVED_FROM`; it never asserts that the stress caused
anything observed in that run. The distinction the rest of this project already draws — failure
*signal* vs. candidate *mode* vs. *confirmed* mode vs. *claim* vs. *evidence* (see
[failures.md](failures.md), [observation-vs-conclusion.md](observation-vs-conclusion.md)) — is
preserved by keeping every one of those as its own typed node, connected by the edge that actually
exists in the source record (`EVIDENCE_FOR`, `SUPPORTS`, `REFUTES`, `CONTEXTUALIZES`, ...), never
collapsed into one generic "related to" relationship. There is no graph-wide reliability or
trust score: a node's or path's meaning comes from what kind it is and what specific edges reach
it, read by a person, not a number the graph computes on its own.

## Deterministic identity

A `GraphSpec` (`gsp_`) is a small, content-addressed construction request: `name`, `version`, an
optional `investigation_id` scope, `included_kinds` (which `NodeKind`s to include; empty means
every kind), and `max_nodes`/`max_edges` bounds. An `EvidenceGraph` (`grh_`) is the versioned
*definition* — its identity is just `spec_id`, so re-registering the same spec is a no-op. A
`GraphSnapshot` (`gsn_`) is one *construction* of that definition against the registry's content at
a point in time; its identity is `(graph_id, source_fingerprint)`, where `source_fingerprint` is a
content hash of every `(kind, id, content_hash)` row actually walked. Constructing the same
definition again against **unchanged** registry content always yields the same snapshot —
re-collection is idempotent, never duplicated, the same way `BenchmarkResult` is keyed on
`(benchmark_id, provenance_fingerprint)`. New evidence added to the registry (a new run, a new
analysis, ...) changes the fingerprint and produces a new snapshot; nothing is overwritten.

## Generic construction, not per-phase hand-wiring

`graph.build.construct` never hand-writes "read a fault trial and link it to its baseline run" for
every phase individually. Every registry entity is already, by this project's own convention,
content-addressed and refers to what it depends on by ID (`<prefix>_<32 hex>`, see
[identity.md](identity.md)). Construction walks each entity's dataclass fields (recursively
through nested dataclasses, never into free-form JSON blobs like `spec`/`detail`/`summary`/
`parameters`) and turns every string that looks like an entity reference into a typed edge,
resolving the target's kind from the reference's own prefix (`graph.taxonomy.PREFIX_TO_KIND`). A
future entity kind needs no new construction code to appear in the graph — only an entry in
`graph.build.ENTITY_TYPES`. Three things need a few lines of special handling because they are not
literal registry IDs:

* `Evidence.relation` selects the edge label (`SUPPORTS`/`REFUTES`/`CONTEXTUALIZES`) instead of a
  fixed default, so an edge is never mislabeled `SUPPORTS` when the source record said otherwise.
* A `split` value paired with a `dataset_fingerprint` value on any entity becomes a synthetic
  `SPLIT` node (`spl_...`, content-addressed on the pair), handled once, generically.
* `FailureRelationship`'s `FAULT`/`CLASS`/`SLICE` endpoints are descriptor strings, not registry
  IDs (`failures.taxonomy.NodeKind`); they become synthetic `DESCRIPTOR` nodes (`dsc_...`).

A reference field not given a specific label in `FIELD_RELATION` still becomes a real, typed edge
— `REFERENCES`, the least specific member of the vocabulary, never a dropped or generic-looking
one.

## Node and relationship vocabulary

`NodeKind` (`graph.taxonomy`) has one member per registry entity `PREFIX`, plus three synthetic,
non-registry kinds: `SPLIT`, `DESCRIPTOR`, and `UNRESOLVED`. `RelationType` is the closed edge
vocabulary: `DERIVED_FROM`, `USES_MODEL`, `USES_DATASET`, `USES_SPLIT`, `PRODUCED_RUN`,
`PRODUCED_ARTIFACT`, `SUPPORTS`, `REFUTES`, `CONTEXTUALIZES`, `EVIDENCE_FOR`, `OBSERVED_IN`,
`MEMBER_OF`, `DEPENDS_ON`, `TRIGGERED_BY`, `REPRODUCED_BY`, `COMPARED_WITH`, `DISCOVERED_FROM`,
`ANALYZED_BY`, `REFERENCES`. Both enums are versioned by `graph.spec.ENGINE_VERSION`: a meaningful
change to either changes the engine version, which every future `EvidenceGraph` records.

## Unresolved references are explicit, never dropped

A `GraphNode.resolved=False` means the referenced ID does not currently resolve in this registry —
a dangling reference is recorded as its own node, not silently skipped. `GraphSnapshot` reports
`unresolved_count` at the top level so a caller never has to scan every node to notice one. A
missing reference is data (the source record really does point somewhere that no longer resolves,
or never did), never fabricated as if it resolved.

## Bounded, deterministic traversal

`GraphQuery` (`max_depth`, `max_visited`, optional `kinds`/`relations` filters) bounds every
traversal call. `SnapshotIndex` (`graph.query`) builds an in-memory adjacency index of one
snapshot's nodes and edges once, then answers `neighbors`, `ancestors` (follow edges backward —
what points AT this node), `descendants` (follow edges forward — what this node's own fields point
at) and `path` (bounded BFS with deterministic tie-breaking) against it. Every traversal stops at
whichever bound it hits first and reports `truncated=True`; a truncated result is never presented
as complete. `path` additionally distinguishes "no path exists within the bound" (`found=False,
truncated=False`) from "the bound was hit before ruling a path out" (`truncated=True`) — those are
different claims and the graph never conflates them.

Most edges in this graph point from the referencing record to the record it references (a
`FailureCluster` points at its `FailureSignal`s, not the reverse), so "what supports X" is usually
an *ancestor* query, not a descendant one — `evidence_for_claim` below is the concrete example.

### Composed queries

Built entirely from the traversal primitives above, in `graph.query`:

* `evidence_for_claim(idx, claim_node)` — the `Evidence` row(s) with an `EVIDENCE_FOR` edge into
  the claim, plus transitively what each one draws on (an artifact, a run, an analysis). A claim
  with no such edge is reported as unsupported in this graph, not hidden.
* `runs_contributing_to_failure_mode(idx, mode_node)` — every `RUN` reachable by following the
  mode's own fields forward (mode → cluster → signal → run).
* `analyses_depending_on_run(idx, run_node)` — every record that names this run (an ancestor
  query: they point at the run, not the reverse).
* `artifacts_for_analysis(idx, analysis_node)` — every artifact connected to an analysis via its
  run (forward to the run the analysis names, then backward to whatever else names that run).
* `model_to_failure_paths` / `dataset_to_failure_paths(idx, ref_id, mode_ref_id)` — a bounded path
  between a model or dataset node and a failure mode node, searched in both directions.

## CLI

`experionyx graph build | list | inspect | neighbors | path | query | diff | replay` (see
[cli.md](cli.md)). `build` executes a real, provenance-carrying `Run` (below); every other command
is a read-only query against an already-persisted `GraphSnapshot`. `path`/`query` exit 1 if no path
or evidence was found within the given bounds; `replay` exits 1 unless verified deterministic.
JSON is the default output (`--format text` for a short human line); invalid requests (an unknown
snapshot, a node not present, a malformed spec file) are clean `error:` messages, never a crash.

## Provenance: construction is a real Run

`graph.engine.run_graph` never writes a snapshot directly. It registers a `ConfigurationRef`/
`Experiment` for the spec (if not already registered) and dispatches through the normal execution
engine (`resolve_procedure`, `Executor.execute` — the same path every other engine in this project
uses, see [execution.md](execution.md)), so a `GraphSnapshot` carries the exact same
[provenance.md](provenance.md) any other run would: environment snapshot, fingerprint, artifacts,
a `Claim` ("Graph *name* *version* was constructed over *N* node(s) and *M* edge(s) ...; *k*
reference(s) did not resolve. This records what the registry's own references are; it makes no
causal claim.") backed by `Evidence` pointing at the stored node/edge/summary artifacts. Calling
`run_graph` again with the exact same spec against unchanged registry content returns the existing
snapshot (`already_built=True`) rather than executing a second time.

### Excluding the graph's own bookkeeping

A graph construction never treats its own `ConfigurationRef`/`Experiment`/`Run`/`Claim` (or an
earlier graph construction's) as evidence to include — without this exclusion, every graph would
permanently absorb the bookkeeping of the graph built before it and never reach a stable,
idempotent fingerprint. `graph.build._excluded` identifies these rows structurally (a
`ConfigurationRef` whose `parameters` contains `"graph_collect"`, the `Experiment`s and `Run`s
built from it, and `Claim`s asserted by `experionyx.graph/<version>`) and `graph.build._iter_entities`
skips them. `EvidenceGraph`/`GraphSnapshot`/`GraphNode`/`GraphEdge` are never in
`graph.build.ENTITY_TYPES` at all, so a "graph of graphs" is excluded structurally, not by this
filter.

## Replay and diff

`graph.engine.replay_check` (`graph replay <gsn_…>`) reconstructs the original collect `Run` as a
new run and compares every stored artifact (`spec`, `nodes`, `edges`, `summary`). If the
registry's content changed since the snapshot was collected (new evidence was added, say), the
source fingerprint differs and this is reported as `sources_changed`, never as nondeterminism —
determinism cannot be judged from a replay whose inputs moved. Otherwise a byte-identical replay is
`deterministic=True`; any difference is `deterministic=False`, and the exact artifacts that
differed are named.

`graph.diff.diff` (`graph diff <a> <b>`) reports raw added/removed nodes and added/removed edges
between two snapshots, matched by what each one actually represents (kind + referenced ID for a
node; endpoint identity + relation for an edge) rather than by internal row ID, so the same logical
node or edge across two snapshots is recognized as unchanged even though its row ID differs. Like
`benchmark.compare` and `resources.compare`, it never labels a difference an improvement or a
regression — new evidence appearing is reported, not judged.

## Schema and artifacts

Schema **v17** adds `evidence_graphs` (`grh_`, immutable), `graph_snapshots` (`gsn_`, immutable),
`graph_nodes` (`gnd_`, immutable) and `graph_edges` (`ged_`, immutable) — four new tables only;
every existing table and every migration `v1→v16` is unchanged. Delete triggers refuse to remove a
row, like every other table in the registry. A collect run's artifacts live under
`graph/{spec,nodes,edges,summary}.json` in the run's artifact directory.

## Phase 20 integration: no new traversal code

`ReproductionAttempt` ([reproducibility.md](reproducibility.md)) and every leaderboard entity
(`BenchmarkProtocol`, `BenchmarkSubmission`, `LeaderboardSnapshot`, `LeaderboardEntry`, see
[leaderboard.md](leaderboard.md)) are in `graph.build.ENTITY_TYPES` like anything else, so they
appear in a constructed graph automatically through the same generic field walk this module already
uses for every other entity kind. `ReproductionAttempt.target_id`/`replay_run_id` become
`REPRODUCED_BY`/`COMPARED_WITH` edges (`FIELD_RELATION` overrides); a `BenchmarkSubmission`'s
`result_id`/`benchmark_id`/`protocol_id` fields become `DERIVED_FROM`/`MEMBER_OF` edges to its real
`BenchmarkResult`/`Benchmark`/`BenchmarkProtocol` nodes — and, transitively, to the real `Run` and
artifacts behind them. Neither integration added a single line to `graph.query` or `graph.build`
beyond a handful of new `ENTITY_TYPES` entries and two `FIELD_RELATION` overrides.

## Known limitations

* `GraphSpec.investigation_id` scopes entities that themselves carry an `investigation_id`; an
  entity with no such attribute (a `Run`, an `Artifact`, a `Provenance` record, ...) is included
  regardless of scope — a scoped graph can still reference records technically outside the
  requested investigation if another in-scope record points at them.
* `max_nodes`/`max_edges` (construction) and `max_depth`/`max_visited` (traversal) are hard,
  reported bounds, not soft limits: a construction or traversal that hits one stops and reports
  `truncated=True` rather than silently returning a partial result as if it were complete.
* The graph is read-only over what the registry's own fields already reference; it discovers no
  new relationships beyond what a source record already names, and it computes no similarity,
  ranking, or trust score of any kind.
* `RESOURCE`-kind entities inside a replayed graph carry the same environment-dependent
  measurements [resources.md](resources.md) already documents; a graph replay compares the graph's
  own structure (nodes/edges/spec), not whether an included resource measurement matches to the
  microsecond.
