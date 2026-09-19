"""SQLite implementation of the Registry protocol. Explicit SQL, no ORM.

Table and column names come only from the static `_SPECS` below; all values are bound parameters.
"""

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import TracebackType
from typing import Self

from experionyx.domain import (
    EVIDENCE_TARGET_TYPES,
    Artifact,
    Claim,
    ConfigurationRef,
    Entity,
    EnvironmentSnapshot,
    Evidence,
    Experiment,
    Investigation,
    Observation,
    Run,
    RunStatus,
)
from experionyx.errors import (
    ConcurrentModificationError,
    CorruptRecordError,
    DuplicateError,
    MissingReferenceError,
    NotFoundError,
    SchemaVersionError,
    ValidationError,
)
from experionyx.hashing import canonical_json, content_hash
from experionyx.provenance import Provenance, RunOutcome
from experionyx.registry import E

DB_SCHEMA_VERSION = 2  # stored in PRAGMA user_version; 2 added provenance + outcomes


@dataclass(frozen=True)
class _Spec:
    table: str
    refs: tuple[tuple[str, type[Entity]], ...] = ()  # (column, referenced entity type)
    plain: tuple[str, ...] = ()  # other indexed columns
    mutable: bool = False  # status may be updated
    optional: tuple[str, ...] = ()  # ref columns that may be NULL

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(c for c, _ in self.refs) + self.plain


# Order matters: referenced tables come first.
_SPECS: dict[type[Entity], _Spec] = {
    Investigation: _Spec("investigations"),
    ConfigurationRef: _Spec("configurations"),
    EnvironmentSnapshot: _Spec("environments"),
    Experiment: _Spec(
        "experiments",
        refs=(("investigation_id", Investigation), ("configuration_id", ConfigurationRef)),
        plain=("status",),
        mutable=True,
    ),
    Run: _Spec(
        "runs",
        refs=(("experiment_id", Experiment), ("environment_id", EnvironmentSnapshot)),
        plain=("status",),
        mutable=True,
    ),
    Observation: _Spec("observations", refs=(("run_id", Run),)),
    Artifact: _Spec("artifacts", refs=(("run_id", Run),)),
    Provenance: _Spec(
        "provenance",
        refs=(
            ("run_id", Run),
            ("experiment_id", Experiment),
            ("environment_id", EnvironmentSnapshot),
            ("configuration_id", ConfigurationRef),
            ("replay_of", Run),
        ),
        optional=("replay_of",),
    ),
    RunOutcome: _Spec("outcomes", refs=(("run_id", Run),), plain=("status",)),
    Claim: _Spec("claims", refs=(("investigation_id", Investigation),), plain=("status",)),
    Evidence: _Spec(
        "evidence",
        refs=(("claim_id", Claim),),
        plain=("target_kind", "target_id", "relation"),
    ),
}


def _ddl() -> str:
    out = ["BEGIN;"]
    for spec in _SPECS.values():
        t = spec.table
        body = [
            "id TEXT PRIMARY KEY",
            *(f"{c} TEXT" + ("" if c in spec.optional else " NOT NULL") for c in spec.columns),
            "payload TEXT NOT NULL",
            "content_hash TEXT NOT NULL",
            *(f"FOREIGN KEY ({c}) REFERENCES {_SPECS[e].table}(id)" for c, e in spec.refs),
        ]
        out.append(f"CREATE TABLE {t} ({', '.join(body)});")
        out.extend(f"CREATE INDEX ix_{t}_{c} ON {t}({c});" for c in spec.columns)
        msg = "registry records cannot be deleted"
        out.append(
            f"CREATE TRIGGER {t}_no_delete BEFORE DELETE ON {t} "
            f"BEGIN SELECT RAISE(ABORT, '{msg}'); END;"
        )
        guarded = ", ".join(("id", *(c for c, _ in spec.refs))) if spec.mutable else ""
        of = f" OF {guarded}" if spec.mutable else ""
        out.append(
            f"CREATE TRIGGER {t}_immutable BEFORE UPDATE{of} ON {t} "
            "BEGIN SELECT RAISE(ABORT, 'registry records are immutable'); END;"
        )
    out += [f"PRAGMA user_version = {DB_SCHEMA_VERSION};", "COMMIT;"]
    return "\n".join(out)


def _column_value(entity: Entity, column: str) -> str | None:
    value = getattr(entity, column)
    if value is None:
        return None
    return str(value.value if isinstance(value, Enum) else value)


class SqliteRegistry:
    """Registry backed by a single SQLite file (or ":memory:")."""

    def __init__(self, path: str | Path, *, timeout: float = 5.0) -> None:
        """`timeout`: seconds a writer waits for another writer's transaction before raising
        `sqlite3.OperationalError`. A registry object must be used from one thread only."""
        # isolation_level=None: we control transactions explicitly (see `transaction`).
        self._conn = sqlite3.connect(str(path), timeout=timeout, isolation_level=None)
        self._depth = 0
        try:
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._init_schema()
        except Exception:
            self._conn.close()
            raise

    def _init_schema(self) -> None:
        (found,) = self._conn.execute("PRAGMA user_version").fetchone()
        if found == 0:
            (tables,) = self._conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
            if tables:
                raise SchemaVersionError(
                    "database is non-empty but has no EXPERIONYX schema version"
                )
            self._conn.executescript(_ddl())
        elif found != DB_SCHEMA_VERSION:
            raise SchemaVersionError(
                f"unsupported database schema version {found} (supported: {DB_SCHEMA_VERSION})"
            )

    # -- Registry protocol --------------------------------------------------------------------

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Group operations atomically. Takes the database write lock up front (BEGIN IMMEDIATE),
        so concurrent writers serialize; nesting joins the outermost transaction. Any exception
        rolls everything back."""
        if self._depth == 0:
            self._conn.execute("BEGIN IMMEDIATE")
        self._depth += 1
        try:
            yield
        except BaseException:
            self._depth -= 1
            if self._depth == 0:
                self._conn.execute("ROLLBACK")
            raise
        else:
            self._depth -= 1
            if self._depth == 0:
                self._conn.execute("COMMIT")

    def add(self, entity: Entity) -> None:
        spec = _SPECS[type(entity)]
        with self.transaction():
            if self.exists(type(entity), entity.id):
                raise DuplicateError(f"{spec.table}: {entity.id} already exists")
            for column, target in spec.refs:
                value = _column_value(entity, column)
                if value is not None:
                    self._require(target, value, f"{spec.table}.{column}")
            if isinstance(entity, Evidence):
                self._require(
                    EVIDENCE_TARGET_TYPES[entity.target_kind],
                    entity.target_id,
                    "evidence.target_id",
                )
            elif isinstance(entity, Provenance):
                self._check_provenance(entity)
            elif isinstance(entity, RunOutcome):
                self._check_outcome(entity)
            payload = entity.to_dict()
            cols = ["id", *spec.columns, "payload", "content_hash"]
            values = [
                entity.id,
                *(_column_value(entity, c) for c in spec.columns),
                canonical_json(payload),
                content_hash(payload),
            ]
            self._conn.execute(
                f"INSERT INTO {spec.table} ({', '.join(cols)}) "  # noqa: S608
                f"VALUES ({', '.join('?' * len(cols))})",
                values,
            )

    def _check_provenance(self, p: Provenance) -> None:
        run = self.get(Run, p.run_id)
        exp = self.get(Experiment, run.experiment_id)
        if (run.experiment_id, run.environment_id, run.seed) != (
            p.experiment_id,
            p.environment_id,
            p.seed,
        ) or exp.configuration_id != p.configuration_id:
            raise ValidationError("provenance is inconsistent with its run and experiment")
        if run.status is not RunStatus.PENDING:
            raise ValidationError("provenance must be recorded while the run is PENDING")
        if (
            p.replay_of is not None
            and self.get(Run, p.replay_of).experiment_id != run.experiment_id
        ):
            raise ValidationError("a replay must belong to the same experiment as its original")

    def _check_outcome(self, o: RunOutcome) -> None:
        run = self.get(Run, o.run_id)
        run.with_status(o.status)  # raises ValidationError if the transition is illegal
        for artifact_id in o.artifact_ids:
            self._require(Artifact, artifact_id, "outcomes.artifact_ids")
            if self.get(Artifact, artifact_id).run_id != o.run_id:
                raise ValidationError(f"artifact {artifact_id} belongs to a different run")
        if run.status is RunStatus.RUNNING and not self.find(Provenance, run_id=o.run_id):
            raise ValidationError("a RUNNING run must have provenance before it can finish")
        recorded = len(self.find(Observation, run_id=o.run_id))
        if recorded != o.observation_count:
            raise ValidationError(
                f"outcome claims {o.observation_count} observations, registry has {recorded}"
            )

    def get(self, cls: type[E], entity_id: str) -> E:
        spec = _SPECS[cls]
        row = self._conn.execute(
            f"SELECT payload, content_hash FROM {spec.table} WHERE id = ?",  # noqa: S608
            (entity_id,),
        ).fetchone()
        if row is None:
            raise NotFoundError(f"{spec.table}: {entity_id} not found")
        return self._decode(cls, entity_id, row[0], row[1])

    def exists(self, cls: type[Entity], entity_id: str) -> bool:
        row = self._conn.execute(
            f"SELECT 1 FROM {_SPECS[cls].table} WHERE id = ?",  # noqa: S608
            (entity_id,),
        ).fetchone()
        return row is not None

    def find(self, cls: type[E], **filters: str) -> list[E]:
        spec = _SPECS[cls]
        unknown = set(filters) - set(spec.columns)
        if unknown:
            raise ValidationError(f"cannot filter {spec.table} by {sorted(unknown)}")
        where = " AND ".join(f"{c} = ?" for c in filters) or "1"
        rows = self._conn.execute(
            f"SELECT id, payload, content_hash FROM {spec.table} "  # noqa: S608
            f"WHERE {where} ORDER BY id",
            [str(x) for x in filters.values()],
        ).fetchall()
        return [self._decode(cls, r[0], r[1], r[2]) for r in rows]

    def update_status(self, entity: Experiment | Run) -> None:
        with self.transaction():
            current: Experiment | Run
            expected: Experiment | Run
            if isinstance(entity, Experiment):
                current = self.get(Experiment, entity.id)
                expected = current.with_status(entity.status)
            else:
                current = self.get(Run, entity.id)
                expected = current.with_status(entity.status)
            if expected != entity:
                raise ValidationError("only `status` may change in a status update")
            payload = entity.to_dict()
            cursor = self._conn.execute(
                f"UPDATE {_SPECS[type(entity)].table} "  # noqa: S608
                "SET status = ?, payload = ?, content_hash = ? WHERE id = ? AND status = ?",
                (
                    entity.status.value,
                    canonical_json(payload),
                    content_hash(payload),
                    entity.id,
                    current.status.value,
                ),
            )
            if cursor.rowcount != 1:
                raise ConcurrentModificationError(f"{entity.id} changed during the update")

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # -- internals ----------------------------------------------------------------------------

    def _require(self, cls: type[Entity], entity_id: str, where: str) -> None:
        if not self.exists(cls, entity_id):
            raise MissingReferenceError(f"{where} references missing {cls.KIND} {entity_id}")

    @staticmethod
    def _decode(cls: type[E], entity_id: str, payload: str, stored_hash: str) -> E:
        data = json.loads(payload)
        if not isinstance(data, dict):
            raise CorruptRecordError(f"{entity_id}: payload is not an object")
        entity = cls.from_dict(data)
        if entity.id != entity_id or content_hash(data) != stored_hash:
            raise CorruptRecordError(f"{entity_id}: stored record does not match its ID/hash")
        return entity
