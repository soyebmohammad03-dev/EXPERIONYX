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
`PRAGMA user_version` holds `DB_SCHEMA_VERSION` (currently 2; version 2 added the `provenance` and
`outcomes` tables and changed `Artifact`/`Observation` payloads). Version-1 (Phase 1) databases are
rejected: there is no migration path yet and no data was ever published. A new database is initialized
atomically; a matching version opens; any other version, or a non-empty database with no version,
raises `SchemaVersionError`. Payloads carry their own `schema_version`, so record format and table
layout can evolve independently. No migrations exist yet.

## Limitations
No migrations. Not a reproducibility guarantee by itself: it stores what the execution engine
and callers report.
