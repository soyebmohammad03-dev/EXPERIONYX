# Visualization / analysis UI

`experionyx viz serve` starts a small, hand-written vanilla-JS single-page app
(`src/experionyx/api/static/`) served alongside the read-only API in [api.md](api.md). There is no
build step, no npm dependency, and no bundler — `index.html` loads `app.js`/`charts.js` directly as
ES modules and every view calls `fetch()` against `/api/*`.

## Why this way

The plan for Phase 23 (see [roadmap.md](roadmap.md)) was explicit: build an analysis *surface*
over the existing engines, not a second implementation of them. A framework-heavy frontend would
have meant either duplicating every entity's shape in a client-side schema layer, or pulling in a
build toolchain the rest of this dependency-free core doesn't have. Plain `fetch()` + hand-written
DOM construction keeps the UI as thin as the API underneath it, and keeps `pip install
experionyx[viz]` the only thing a researcher needs beyond the core package.

## Views

Hash-routed (`#/investigations/{id}`, `#/graph/snapshots/{ref}`, ...):

- **Investigations** — identity, scope, experiments, runs, report/dossier references.
- **Reliability** — per-dimension status with supporting/missing evidence; deliberately rendered
  as a per-dimension bar, never summed into one score.
- **Failures** — clusters, modes, signals, evidence, relationships.
- **Faults / robustness** — trials, degradation curve (parameter value vs. deterioration, with its
  confidence band and classification, read from the persisted `fault/analysis.json` artifact).
- **Drift** — windows and analyses, with feature/label/prediction/performance results from the
  persisted drift documents.
- **Statistics** — effect size, interval, p-value, correction, sample size and comparison identity,
  rendered verbatim from the stored `StatisticalAnalysis`.
- **Benchmark / leaderboard** — protocol identity, submissions, the most recent leaderboard
  snapshot for a protocol, protocol-constrained comparisons.
- **Reproducibility** — attempts, classifications, environment/artifact-verification detail.
- **Knowledge / failure graph** — a node/edge table plus a small deterministic SVG rendering of a
  bounded neighborhood (`GET /api/graph/snapshots/{ref}/nodes/{id}/neighbors`); node positions are
  a pure function of each node's index in the (bounded) result set, not a force simulation, so the
  same query always draws the same picture.
- **Reports / dossiers** — list, open, inspect claims/evidence/limitations/sufficiency, export
  Markdown (a link straight to the API's `/export.md` route).

## Charts (`charts.js`)

Four deterministic SVG primitives — `bar`, `groupedBar`, `lineWithBand` (line plus an optional
confidence band, used for degradation curves, drift trajectories and calibration curves alike),
and `scatter` — cover every visualization the roadmap called for by composition, so there is one
rendering path to keep correct rather than one per chart type. Every chart is handed a `metric`,
`unit`, `population`, `comparison`, and `uncertainty` (or the literal `"unavailable"`) by the
`/api/viz/*` endpoints it reads from, and renders that framing as visible text — never a bare
number with no context.

No chart samples randomly and no chart is decorative: everything drawn is a reshaping of a value
that already exists on a persisted entity (see `src/experionyx/api/routers/viz.py`'s module
docstring). Calling the backing endpoint twice against unchanged evidence returns byte-identical
JSON — `tests/test_api_*` assert this for every `/api/viz/*` route the UI uses.

## Evidence tracing

Every detail view renders a `traceChain(...)` of clickable ids back through
run → experiment → investigation (or the equivalent chain for the entity in question), using the
same foreign-key fields already present on the API payload — there is no separate "trace" endpoint
or duplicated lineage logic; the chain is just the entity's own references, made clickable.

## Scale limits

The UI never loads a whole investigation's evidence or a whole graph snapshot into the browser: list
views are paginated (default page size 50, hard cap 500) and graph views are bounded traversals
(`max_depth`/`max_visited`, with `truncated` surfaced when the bound was hit). For very large
investigations or snapshots, use the `investigation`/`profile_id`/`node_id` filters and narrower
`max_depth` rather than expecting an unbounded view — see also [limitations.md](limitations.md).
