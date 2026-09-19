# Registry and Persistence

## Architecture
`experionyx.registry.Registry` is a `Protocol`: `add`, `get`, `exists`, `find`,
`update_status`, `transaction`, `close`. `experionyx.sqlite.SqliteRegistry` implements it with the standard
library `sqlite3` and explicit SQL (no ORM). Domain modules never import SQLite, so another
backend can implement the protocol without touching domain logic.

## Semantics
- **Append-only, immutable records.** `add` of an existing ID raises `DuplicateError`. There is
  no delete and no general update.
- **One exception:** `update_status` persists a legal `Experiment`/`Run` status transition;
  any other difference from the stored record is rejected.
- **References are validated** before insert (`MissingReferenceError`), including the polymorphic
  evidence target. `get` of an unknown ID raises `NotFoundError`.
- `find(cls, **filters)` matches indexed columns (foreign keys, `status`, evidence target
  fields) with equality, ordered by ID.

## Storage model
One table per entity: `id` (primary key), indexed FK/status columns, `payload` (canonical JSON of
`to_dict()`, includes `kind` and `schema_version`), and `content_hash`. Relationships are real
foreign key columns (`PRAGMA foreign_keys = ON`); nothing is inferred from file names.
SQLite triggers additionally reject `DELETE` everywhere and `UPDATE` on immutable tables (on
`runs`/`experiments` only `status`, `payload`, `content_hash` may change), so immutability holds
even for code that bypasses the Python layer.
Evidence targets are polymorphic and are checked in Python, not by a foreign key.

## Transactions and concurrency
`transaction()` makes a group of operations atomic (`BEGIN IMMEDIATE`, nestable, rollback on any
exception); `add` and `update_status` are transactional by themselves. `add` also enforces
cross-record invariants for provenance and outcomes (see [execution.md](execution.md)). Status
updates are compare-and-swap. Single connection per object, one thread; concurrent writers
serialize on the SQLite lock. See [execution.md](execution.md#concurrency).

## Schema versioning
`PRAGMA user_version` holds `DB_SCHEMA_VERSION` (currently 6; v6 added `interaction_analyses`, `interaction_effects` and `interaction_evidence`; v4 added the fault-laboratory tables, v5 the failure tables `failure_signals`, `failure_clusters`, `failure_modes`, `failure_evidence`, `failure_relationships`): v2 added `provenance`/`outcomes`
(and gave artifacts a name and category), v3 added `models`/`datasets` (and `inputs` on
provenance).  A new database is created at the latest version. An older database is **migrated in place**,
one version at a time inside a single transaction (all steps or none), after copying the file to
`<name>.v<old>.bak` (in-memory databases are not backed up). Steps are explicit functions in
`sqlite.py` (`_migrate_1_to_2`, `_migrate_2_to_3`); payload rewrites keep record IDs and
recompute content hashes, and migrated records keep their immutability triggers. Migrated data
is only as rich as the old data: v1 artifacts get `name = path` and category OUTPUT, and old
provenance gets `inputs = null` (its fingerprint is unchanged by design). A newer version, or a
non-empty database with no version, raises `SchemaVersionError`. Migration tests build real v1
and v2 databases.
