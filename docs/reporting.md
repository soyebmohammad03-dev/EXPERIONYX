# Research reporting (Phase 21)

A report is an **evidence consumer, not a new experimental engine**. `experionyx/reporting/`
assembles a structured, versioned, provenance-linked research document from evidence the registry
already holds; it never re-executes an experiment, recomputes a statistic, or invents a
measurement to fill a gap.

```
ReportSpec (investigation + report type + template) --generate_report()--> Report
                                                              |
                              collect_evidence() --> EvidenceBundle --> ReportSection(s) + ReportFinding(s)
```

## Claims never exist without evidence

Every `ReportFinding` is `Report`'s unit of claim: a `statement`, an evidence tuple
(`EvidenceReference`, each naming a concrete `source_kind`/`source_id`), and a `status` from the
existing `ClaimStatus` vocabulary (`experionyx.domain`) — reused, not redefined, so "supported"
means the same thing here as everywhere else in the project. A finding built from zero evidence is
always `UNSUPPORTED`; the generator never sets `SUPPORTED` without a resolvable evidence
reference. There is no code path that fabricates a measurement, a p-value, a benchmark value, or a
causal claim to complete a section.

Sections follow the same discipline: a template names the sections a report type calls for
(`taxonomy.BUILTIN_TEMPLATE_SECTIONS`), but the generator includes a non-synthesized section only
if its evidence bucket (`taxonomy.SECTION_SOURCES`) is non-empty. A section the evidence does not
support is recorded as an entry in `Report.evidence_gaps` instead of being silently dropped or
padded with a generic sentence. `Report.status` is `INCOMPLETE` whenever any gap exists,
`GENERATED` otherwise — never a hidden default.

## Evidence assembly reuses the graph's own lookup pattern

`reporting.collector.collect_evidence` pulls every evidence-source entity
(`EVIDENCE_SOURCES`: fault experiments, failure modes, interaction analyses, reliability
profiles, benchmark results/submissions, leaderboard snapshots, slice/drift/quality/stress/
calibration/resource analyses, schedule runs, graph snapshots, reproduction attempts) by its
existing `investigation_id` column — the same generic, investigation-scoped lookup the graph
builder and CLI already use ([graph.md](graph.md)). `StatisticalAnalysis` has no investigation
scope of its own (a statistic is computed over values, not owned by an investigation), so it is
only included when its ID is named explicitly in the `ReportSpec` — deliberately, rather than
guessed from its opaque `sources` blob. Nothing here duplicates evaluation, fault, reliability, or
statistics logic; it only reads what those engines already persisted.

`evidence_digest` content-hashes exactly which records (by ID and content) went into a report, so
regenerating against unchanged evidence returns the identical `Report`, and any change to any
included record's content produces a new, distinct report — the same idempotent-construction
convention as `GraphSnapshot` and `LeaderboardSnapshot`.

## Statistical results are rendered, never reinterpreted

When a section draws on a `StatisticalAnalysis`, the report shows that record's own fields
verbatim — effect size, uncertainty interval, sample size, test/method, p-value where applicable,
correction method where applicable, assumption/status — through the same `_scalar_fields`
verbatim-field extraction every evidence item uses. No section recomputes a statistic, adjusts a
p-value, or re-labels an assumption; `experionyx stats verify` remains the only way to confirm a
statistical result.

## Templates are versioned; changing one changes report identity

`ReportTemplate` (`rtp_`) is a versioned, content-addressed description of a report type's section
list and ordering. `Report`'s identity (`_identity()`) is `(spec, template_id, evidence_digest)`:
regenerating a report from the same spec against the same template and unchanged evidence yields
the identical `Report`; changing either the template or the evidence produces a new, distinct
report. Old reports stay reproducible because their template is never mutated in place — a
template change is a new `ReportTemplate` row, addressed by its own content hash
(`taxonomy.TEMPLATE_ENGINE_VERSION` tracks the section vocabulary/ordering this module knows how
to build; a caller can also register a custom `ReportTemplate` via `ensure_template`).

## Evidence graph integration

`Report`, `ReportFinding`, and `ReportTemplate` are registered like any other entity and
participate in the Phase 19 graph generically: a `Report`'s `investigation_id` and a
`ReportFinding`'s `evidence` references become the same generic field-walk edges that already
connect every other entity kind (see [graph.md](graph.md)) — no second graph or traversal system.

## CLI

```
experionyx report generate <TYPE> --investigation inv_... [--template-id --run --statistical-analysis --format]
experionyx report list [--investigation]
experionyx report show <rpt_...>
experionyx report sections <rpt_...>
experionyx report claims <rpt_...>
experionyx report validate <rpt_...>
experionyx report export <rpt_...> [--export-format markdown|html --output PATH]
```

`report validate` exits 1 if any claim lacks resolvable evidence, any evidence reference does not
resolve, or any statistical claim references something other than a `StatisticalAnalysis` —
following the same exit convention as `reproduce verify`/`graph replay`.

## Export

Markdown is the canonical, deterministic export (`render.render_markdown`): identical report
content always produces byte-identical Markdown. HTML (`render.render_html`) is a thin derived
wrapper around the same content — no templating engine or frontend framework was introduced for
it. Both preserve the report's ID, template ID, and every evidence reference verbatim; an export
is a rendering of a persisted `Report`, never a second copy of its evidence.

## Known limitations

- Report generation only ever reads existing evidence; a report generated before an experiment
  ran will show that section as an explicit evidence gap, not an inferred placeholder.
- `StatisticalAnalysis` evidence must be named explicitly in a `ReportSpec`; there is no automatic
  inference of "the statistics that apply to this investigation."
- There is no universal "best model" conclusion, composite reliability score, or ranking anywhere
  in a generated report — consistent with every other engine in the project
  ([leaderboard.md](leaderboard.md), [reliability.md](reliability.md)).
