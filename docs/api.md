# Read-only HTTP API

`src/experionyx/api/` is a FastAPI surface over an existing workspace's registry and artifact
store. It is a **viewer**, not a second engine: every endpoint calls an existing read method
(`registry.get`/`find`/`exists`, or a subsystem's own `search`/`compare`/`document` helper) and
returns the entity's own `to_dict()`. No endpoint computes a new statistic, recomputes an
analysis, or mutates anything — `tests/test_api_read_only.py` mechanically greps every router
source file and fails the build if `.add(` or `.update_status(` ever appears there.

Run it:

```bash
pip install -e ".[viz]"
experionyx --workspace .experionyx viz serve --host 127.0.0.1 --port 8420
```

Then open `http://127.0.0.1:8420/` for the UI, or call `/api/*` directly for JSON.

## Conventions

- **Pagination.** Every list endpoint accepts `limit` (default 50, capped at 500) and `offset`,
  and returns `{"items": [...], "total": ..., "limit": ..., "offset": ...}`. Bounded queries scoped
  to a single parent id (e.g. runs of one experiment) are not separately paginated — the true
  bound is the parent relationship, not an artificial page.
- **IDs.** Path parameters that name an entity id are validated against that entity's
  `<prefix>_<32 hex>` shape before any registry call; a malformed id is a `422`, an unknown id is
  a `404` (`experionyx.errors.NotFoundError`/`ValidationError`, the same taxonomy the CLI uses —
  see [errors.py](../src/experionyx/api/errors.py)).
- **Unavailable evidence.** A field that has no persisted value is always returned as the literal
  string `"unavailable"`, never `0`, `null`-by-omission, or a fabricated default.
- **No score, no verdict.** `dimension_status` on a reliability profile is returned per-dimension;
  leaderboard entries are returned in deterministic id order, never ranked by a computed score; a
  benchmark/submission comparison rejects (`422`) two results from incompatible protocols instead
  of silently comparing them.
- **Graph bounds.** Every traversal endpoint (`neighbors`/`ancestors`/`descendants`/`related`/
  `path`) forwards `max_depth`/`max_visited` straight into `graph.query`'s existing bounded BFS and
  reports `truncated` when the bound was hit — no endpoint loads a whole snapshot into one
  response.

## Endpoints by area

| Area | Prefix | Notable routes |
|---|---|---|
| Investigations | `/api/investigations` | list/get, `/experiments`, `/experiments/{id}`, `/experiments/{id}/runs`, `/runs/{id}` |
| Reliability | `/api/reliability` | `/profiles` (search by model/dataset/scope/split), `/profiles/{id}`, `/profiles/{id}/provenance`, `/compare?a=&b=` |
| Failures | `/api/failures` | `/clusters`, `/clusters/{id}`, `/modes`, `/modes/{id}` (evidence + relationships) |
| Faults / robustness | `/api/faults` | `/experiments`, `/experiments/{id}` (trials + analysis ids), `/analyses/{id}` |
| Drift | `/api/drift` | `/windows`, `/analyses`, `/analyses/{id}`, `/analyses/{id}/document?name=` |
| Statistics | `/api/stats` | `/analyses`, `/analyses/{id}` (effect size, CI, p-value, correction, config, sources verbatim) |
| Benchmark / leaderboard | `/api/benchmark` | `/benchmarks`, `/results/{id}`, `/results/compare?a=&b=`, `/protocols/{id}/leaderboard`, `/submissions/compare?a=&b=` |
| Reproducibility | `/api/reproducibility` | `/attempts`, `/attempts/{id}`, `/resolve/{kind}/{id}`, `/runs/{id}/artifact-verification` |
| Graph | `/api/graph` | `/snapshots`, `/snapshots/{ref}`, `/snapshots/{ref}/nodes/{id}[/neighbors|/ancestors|/descendants|/related]`, `/snapshots/{ref}/path?from_id=&to_id=` |
| Reports | `/api/reports` | list/get, `/findings`, `/export.md` |
| Dossiers | `/api/dossiers` | list/get, `/items`, `/findings`, `/export.md` |
| Chart-ready views | `/api/viz` | `/reliability/{id}/dimensions`, `/failures/frequency?investigation=`, `/faults/{id}/degradation`, `/drift/{id}/trajectory`, `/calibration/{id}/curve`, `/resources/{id}/measurements`, `/stats/{id}/effect` |

`/api/viz/*` endpoints only reshape fields that already exist on a persisted entity into a
`{metric, unit, population, comparison, points, uncertainty, provenance_ref}` envelope for
charting — see [visualization.md](visualization.md). Calling the same `/api/viz/*` endpoint twice
against unchanged evidence returns byte-identical JSON (no random sampling, no wall-clock content).

## What this API deliberately does not do

- It does not trigger execution, replay, fault injection, benchmarking, or report/dossier
  generation — those stay CLI/engine operations. The API is read-only by construction.
- It does not add a new database, a new schema version, or a duplicate of the registry's data —
  every response is derived live from the same `registry.sqlite` the CLI reads.
- It does not implement authentication/authorization; it is meant for a researcher's own machine
  or a workspace they already trust, the same trust boundary as running the CLI directly. Do not
  expose `experionyx viz serve` on an untrusted network without putting a reverse proxy with auth
  in front of it.
