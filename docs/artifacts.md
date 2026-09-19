# Artifacts

`Artifact` (domain entity) records metadata for a file a run produced: logical `name`,
`category` (OUTPUT, DATA, LOG, DIAGNOSTIC), relative `path`, `size_bytes`, `sha256` digest of the
**file's bytes**, `media_type`, producing run, timestamp. The bytes are not stored in SQLite.

## Lifecycle
1. The store prepares `metadata/` and `artifacts/` for the run.
2. The procedure writes files directly into `ctx.artifact_dir` (no copying).
3. `ctx.register_artifact(path)` streams the file through SHA-256 in 1 MiB chunks (bounded
   memory) and records the artifact. A file that cannot be hashed is never registered.
4. Before COMPLETED, every registered artifact is re-hashed. A missing or changed file fails the
   run (ARTIFACT_REGISTRATION). Re-registering after a change yields a *new* artifact, so the
   older registration still fails verification, by design.
5. `experionyx verify <run-id>` re-checks a run's artifacts at any later time.

Paths must be normalized relative POSIX paths; `..`, absolute paths and symlinks pointing outside
the artifact directory are refused.

## Storage layout
Default workspace `./.experionyx` (CLI `--workspace`; library: any directory):

```
<workspace>/registry.sqlite
<workspace>/experiments/<experiment-id>/runs/<run-id>/metadata/{provenance,outcome}.json
<workspace>/experiments/<experiment-id>/runs/<run-id>/artifacts/...
```
`.experionyx/` and `/experiments/` are git-ignored. Nothing is written under `src/`.

## Replaceable backend
`ArtifactStore` is a `Protocol` (`prepare`, `register`, `verify`, `write_metadata`).
`LocalArtifactStore` is the only implementation; an object-store backend can replace it without
changing the executor. Limits: single-writer assumptions, no garbage collection of orphan files,
no dataset/model-weight hashing.
