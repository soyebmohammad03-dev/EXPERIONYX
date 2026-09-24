# Evidence dossiers (Phase 22)

An `EvidenceDossier` is **structured, persisted research evidence, not merely a document**. It
sits on top of the reporting layer ([reporting.md](reporting.md)): where a `Report` is one
generated, rendered artifact, a dossier is the underlying research package — the scope, the
included evidence, the reports that drew on it, and an explicit sufficiency analysis of what is
and is not supported — built from the registry's own persisted state.

```
DossierSpec (investigation + research question + source scope) --build_dossier()--> EvidenceDossier
                                                                          |
                      collect_evidence() (reused from reporting) --> DossierItem(s) + DossierFinding(s)
                                                                          |
                                                        build_snapshot() --> DossierSnapshot (immutable)
```

## Built from evidence and reports, never re-executed

`dossier.builder.build_dossier` calls the exact same `reporting.collector.collect_evidence` the
reporting layer uses — no second evidence-collection path exists. It also reuses every `Report`
already registered for the investigation, importing their `ReportFinding`s by reference
(`report_finding_id`) rather than duplicating a claim's statement or evidence a report already
established. Nothing here executes or reruns an experiment, a statistical test, or an analysis;
`STANDARD_LIMITATIONS` states this explicitly on every dossier.

## Deterministic identity and construction

`EvidenceDossier`'s identity (`_identity()`) is `(spec_id, source_evidence_digest)` — the same
pattern as `Report`: an unchanged registry and the same `DossierSpec` always produce the identical
dossier; a later experiment that changes the evidence in scope produces a new, distinct dossier
rather than mutating the old one (`build_dossier` is idempotent — a second call against unchanged
evidence returns the same `dossier_id` with a `warnings` entry, never a duplicate row).

`DossierSpec` supports construction from an investigation, a report, a run, a failure mode, a
benchmark result, a reliability profile, or a graph snapshot (`DossierSourceKind`), plus an
explicit `(source_kind, source_id)` list for evidence a caller wants included regardless of
automatic scoping. Every included item is recorded as a `DossierItem` with its `reason` — exactly
what was included and why is always inspectable, never implicit.

## Evidence sufficiency analysis: nothing is hidden

`build_dossier` runs a `DossierFinding` per claim it examines, each carrying one
`SufficiencyStatus`:

| Status | Meaning |
|---|---|
| `SUPPORTED` / `UNSUPPORTED` | carried over verbatim from an included report's own `ClaimStatus` |
| `MISSING_EXPECTED_EVIDENCE` | an evidence category the reporting layer knows how to collect (`reporting.collector.EVIDENCE_SOURCES`) has no persisted records for this investigation, or a report already recorded the gap |
| `PROVENANCE_GAP` | a `Run` in scope has no matching `Provenance` record |
| `REPRODUCIBILITY_GAP` | the investigation has runs in scope but no `ReproductionAttempt` targets any of them |
| `CONFLICTING_EVIDENCE` | two or more `StatisticalAnalysis` records of the same `analysis_kind`, included explicitly, disagree (different `result_hash`) |
| `UNAVAILABLE_EVIDENCE` / `STALE_EVIDENCE` | reserved statuses for evidence a future source explicitly marks unavailable or superseded |

A gap, a provenance hole, or a conflict is never silently omitted from the dossier; it is a
first-class `DossierFinding` a caller can list (`dossier findings`) and act on.

## Conflicting evidence is preserved, never resolved

When statistical analyses of the same kind disagree, `build_dossier` keeps **both** as their own
`CONFLICTING_EVIDENCE` findings and cross-references them through `conflicting_with` (each
finding names every other finding in the same disagreement group). No result is discarded, chosen,
or averaged; resolving the disagreement is left to a human reviewing both pieces of evidence side
by side.

## Evidence lineage

Every `DossierItem` retains `source_kind`/`source_id` (the concrete entity it points at),
`captured_content_hash` (that entity's own `content_hash()` at inclusion time, so a later change to
the underlying record is detectable as stale evidence without rebuilding the dossier), and
`provenance_fingerprint` (the matching `Provenance.fingerprint` for a `run` item, when one exists).
This is the same evidence lineage the graph and provenance systems already expose
([graph.md](graph.md), [provenance.md](provenance.md)) — the dossier does not introduce a second
lineage model.

## Immutable snapshots

`dossier.builder.build_snapshot` freezes a dossier's *current* item and finding ID sets into a
`DossierSnapshot` (`dsn_`). Identity is derived from the dossier's ID, its
`source_evidence_digest`, and the exact sorted item/finding ID tuples captured — so a later
experiment that changes the investigation's evidence produces a new `EvidenceDossier` (and could be
snapshotted again) without altering an existing snapshot at all. Snapshotting an unchanged dossier
again is idempotent and returns the same snapshot ID.

## Reproducibility integration

`EvidenceDossier.reproduction_attempt_ids` names every `ReproductionAttempt`
([reproducibility.md](reproducibility.md)) collected for the investigation — classification,
environment differences, artifact-digest verification, and replay references included by
reference, never re-executed. A dossier with runs but no reproduction attempts is flagged
explicitly (`REPRODUCIBILITY_GAP`), never assumed reproduced.

## Validation

`experionyx dossier validate` lists every unresolved `DossierFinding` (any status other than
`SUPPORTED`) and exits 1 if any evidence is `UNAVAILABLE_EVIDENCE` or `STALE_EVIDENCE` — the same
exit convention as `report validate`. Every `EvidenceReference` a dossier finding carries must
resolve to a real registry entity; every included item's provenance is checked where a run is
involved; dossier and snapshot IDs are always the deterministic content hash of their declared
inputs, so two independent builds from the same spec and evidence state always agree.

## CLI

```
experionyx dossier build <--investigation --question> [--source-kind --source-id --evidence k:v ... --format]
experionyx dossier list [--investigation]
experionyx dossier show <dsr_...>
experionyx dossier items <dsr_...>
experionyx dossier findings <dsr_...>
experionyx dossier snapshot <dsr_...>
experionyx dossier validate <dsr_...>
experionyx dossier export <dsr_...> [--output PATH]
```

## Known limitations

- Sufficiency analysis for `MISSING_EXPECTED_EVIDENCE` checks every evidence category the
  reporting layer knows how to collect, even ones a particular research question does not care
  about; a caller filters the resulting findings by relevance rather than the dossier omitting
  categories on its own judgment.
- Conflict detection compares `StatisticalAnalysis` records by `analysis_kind` and `result_hash`
  only, and only over analyses explicitly named in the dossier's evidence; it does not
  automatically discover every statistic that might bear on a claim.
- `STALE_EVIDENCE`/`UNAVAILABLE_EVIDENCE` are reserved statuses with no automatic detector yet;
  today they are only ever set by a future evidence source that explicitly reports them.
