"""SQLite implementation of the Registry protocol. Explicit SQL, no ORM.

Table and column names come only from the static `_SPECS` below; all values are bound parameters.
"""

import json
import shutil
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import TracebackType
from typing import Self

from experionyx.adapters.records import RegisteredDataset, RegisteredModel
from experionyx.benchmark.entities import Benchmark, BenchmarkResult, BenchmarkUnit
from experionyx.calibration.entities import CalibrationAnalysis, CalibrationResult
from experionyx.data_quality.entities import QualityAnalysis, QualityCheck
from experionyx.domain import (
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
    evidence_target_type,
)
from experionyx.drift.entities import DriftAnalysis, DriftWindow
from experionyx.errors import (
    ConcurrentModificationError,
    CorruptRecordError,
    DuplicateError,
    MissingReferenceError,
    NotFoundError,
    SchemaVersionError,
    ValidationError,
)
from experionyx.failures.entities import (
    FailureCluster,
    FailureEvidence,
    FailureMode,
    FailureRelationship,
    FailureSignal,
)
from experionyx.failures.taxonomy import NodeKind
from experionyx.faults.entities import FaultAnalysis, FaultExperiment, FaultTrial
from experionyx.graph.entities import EvidenceGraph, GraphEdge, GraphNode, GraphSnapshot
from experionyx.hashing import canonical_json, content_hash
from experionyx.interactions.entities import (
    InteractionAnalysis,
    InteractionEffect,
    InteractionEvidence,
)
from experionyx.provenance import Provenance, RunOutcome
from experionyx.registry import E
from experionyx.reliability.entities import ReliabilityProfile, ReliabilityReference
from experionyx.resources.entities import ResourceAnalysis, ResourceTrial
from experionyx.scheduler.entities import (
    ExecutionAttempt,
    Schedule,
    ScheduleRun,
    ScheduleUnit,
    UnitStateTransition,
)
from experionyx.slices.entities import Slice, SliceAnalysis
from experionyx.stats.entities import StatisticalAnalysis
from experionyx.stress.entities import StressAnalysis, StressTrial

# PRAGMA user_version. 2: provenance+outcomes. 3: models+datasets. 4: faults. 5: failures.
# 6: interactions. 7: reliability profiles. 8: benchmarks. 9: statistical analyses.
# 10: slices and slice analyses. 11: temporal windows and drift analyses.
# 12: data quality analyses and check results. 13: stress analyses and trials.
# 14: calibration analyses and per-context calibration results.
# 15: resource analyses and per-pass resource trials.
# 16: schedules, schedule runs, schedule units, execution attempts and unit state transitions.
# 17: evidence graphs, graph snapshots, graph nodes and graph edges.
DB_SCHEMA_VERSION = 17


@dataclass(frozen=True)
class _Spec:
    table: str
    refs: tuple[tuple[str, type[Entity]], ...] = ()  # (column, referenced entity type)
    plain: tuple[str, ...] = ()  # other indexed columns
    mutable: bool = False  # status may be updated
    optional: tuple[str, ...] = ()  # ref columns that may be NULL
    since: int = 1  # schema version that introduced the table

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
        since=2,
    ),
    RunOutcome: _Spec("outcomes", refs=(("run_id", Run),), plain=("status",), since=2),
    RegisteredModel: _Spec(
        "models", plain=("name", "version", "adapter", "adapter_version", "fingerprint"), since=3
    ),
    RegisteredDataset: _Spec(
        "datasets", plain=("name", "version", "adapter", "adapter_version", "fingerprint"), since=3
    ),
    FaultExperiment: _Spec(
        "fault_experiments",
        refs=(
            ("investigation_id", Investigation),
            ("baseline_experiment_id", Experiment),
            ("baseline_run_id", Run),
        ),
        plain=("status",),
        mutable=True,
        since=4,
    ),
    FaultTrial: _Spec(
        "fault_trials",
        refs=(
            ("fault_experiment_id", FaultExperiment),
            ("baseline_run_id", Run),
            ("treatment_experiment_id", Experiment),
            ("treatment_run_id", Run),
        ),
        plain=("status", "family_id", "fault_id"),
        optional=("treatment_experiment_id", "treatment_run_id"),
        since=4,
    ),
    FaultAnalysis: _Spec(
        "fault_analyses",
        refs=(("fault_experiment_id", FaultExperiment), ("run_id", Run)),
        since=4,
    ),
    FailureSignal: _Spec(
        "failure_signals",
        refs=(("run_id", Run), ("experiment_id", Experiment)),
        plain=("signal_kind", "category", "signature"),
        since=5,
    ),
    FailureCluster: _Spec(
        "failure_clusters",
        refs=(("investigation_id", Investigation),),
        plain=("algorithm",),
        since=5,
    ),
    FailureMode: _Spec(
        "failure_modes",
        refs=(("investigation_id", Investigation), ("cluster_id", FailureCluster)),
        plain=("status", "category"),
        mutable=True,
        since=5,
    ),
    FailureEvidence: _Spec(
        "failure_evidence",
        refs=(("failure_mode_id", FailureMode),),
        plain=("evidence_kind",),
        since=5,
    ),
    FailureRelationship: _Spec(
        "failure_relationships",
        refs=(("investigation_id", Investigation),),
        plain=("subject_id", "predicate", "object_id"),
        since=5,
    ),
    InteractionAnalysis: _Spec(
        "interaction_analyses",
        refs=(("investigation_id", Investigation), ("run_id", Run)),
        plain=(
            "spec_id",
            "status",
            "primary_metric",
            "primary_class",
            "structural_key",
            "fault_a",
            "fault_b",
            "model_fingerprint",
            "dataset_fingerprint",
        ),
        mutable=True,
        since=6,
    ),
    InteractionEffect: _Spec(
        "interaction_effects",
        refs=(("analysis_id", InteractionAnalysis),),
        plain=("level", "measure", "effect_status", "interaction_class"),
        since=6,
    ),
    InteractionEvidence: _Spec(
        "interaction_evidence",
        refs=(("analysis_id", InteractionAnalysis),),
        plain=("evidence_kind",),
        since=6,
    ),
    ReliabilityProfile: _Spec(
        "reliability_profiles",
        refs=(("investigation_id", Investigation), ("run_id", Run)),
        plain=(
            "spec_id",
            "scope",
            "model_fingerprint",
            "dataset_fingerprint",
            "split",
            "evaluation_config_hash",
        ),
        since=7,
    ),
    ReliabilityReference: _Spec(
        "reliability_references",
        refs=(("profile_id", ReliabilityProfile),),
        plain=("dimension", "ref_kind", "ref_id"),
        since=7,
    ),
    Benchmark: _Spec(
        "benchmarks",
        refs=(("model_record_id", RegisteredModel), ("dataset_record_id", RegisteredDataset)),
        plain=("name", "version", "spec_id", "protocol_hash", "engine_version"),
        since=8,
    ),
    BenchmarkResult: _Spec(
        "benchmark_results",
        refs=(("benchmark_id", Benchmark), ("investigation_id", Investigation), ("run_id", Run)),
        plain=("protocol_hash", "provenance_fingerprint", "coverage_status"),
        since=8,
    ),
    BenchmarkUnit: _Spec(
        "benchmark_units",
        refs=(("result_id", BenchmarkResult), ("run_id", Run)),
        plain=("unit_key", "unit_kind", "unit_status", "grid"),
        optional=("run_id",),
        since=8,
    ),
    StatisticalAnalysis: _Spec(
        "statistical_analyses",
        plain=("analysis_kind", "input_hash", "engine_version", "analysis_status"),
        since=9,
    ),
    Slice: _Spec("slices", plain=("name",), since=10),
    SliceAnalysis: _Spec(
        "slice_analyses",
        refs=(("investigation_id", Investigation), ("run_id", Run)),
        plain=("spec_id", "baseline_run_id", "dataset_fingerprint", "analysis_status"),
        since=10,
    ),
    DriftWindow: _Spec("temporal_windows", plain=("ordering_field", "role"), since=11),
    DriftAnalysis: _Spec(
        "drift_analyses",
        refs=(("investigation_id", Investigation), ("run_id", Run)),
        plain=("spec_id", "baseline_run_id", "dataset_fingerprint", "analysis_status"),
        since=11,
    ),
    QualityAnalysis: _Spec(
        "quality_analyses",
        refs=(
            ("investigation_id", Investigation),
            ("run_id", Run),
            ("dataset_id", RegisteredDataset),
        ),
        plain=("spec_id", "dataset_fingerprint", "analysis_status"),
        since=12,
    ),
    QualityCheck: _Spec(
        "quality_checks",
        refs=(("analysis_id", QualityAnalysis),),
        plain=("check_id", "check_type", "status"),
        since=12,
    ),
    StressAnalysis: _Spec(
        "stress_analyses",
        refs=(
            ("investigation_id", Investigation),
            ("run_id", Run),
            ("model_id", RegisteredModel),
            ("dataset_id", RegisteredDataset),
            ("baseline_run_id", Run),
        ),
        plain=("spec_id", "analysis_status"),
        since=13,
    ),
    StressTrial: _Spec(
        "stress_trials",
        refs=(("analysis_id", StressAnalysis), ("run_id", Run)),
        plain=("unit_key", "family", "origin", "status"),
        optional=("run_id",),
        since=13,
    ),
    CalibrationAnalysis: _Spec(
        "calibration_analyses",
        refs=(
            ("investigation_id", Investigation),
            ("run_id", Run),
            ("baseline_run_id", Run),
        ),
        plain=("spec_id", "analysis_status"),
        since=14,
    ),
    CalibrationResult: _Spec(
        "calibration_results",
        refs=(("analysis_id", CalibrationAnalysis),),
        plain=("context_key", "context_kind", "status"),
        since=14,
    ),
    ResourceAnalysis: _Spec(
        "resource_analyses",
        refs=(("investigation_id", Investigation), ("run_id", Run)),
        plain=("spec_id", "analysis_status"),
        since=15,
    ),
    ResourceTrial: _Spec(
        "resource_trials",
        refs=(("analysis_id", ResourceAnalysis),),
        plain=("phase", "status"),
        since=15,
    ),
    Schedule: _Spec("schedules", plain=("spec_id",), since=16),
    ScheduleRun: _Spec(
        "schedule_runs",
        refs=(("schedule_id", Schedule), ("investigation_id", Investigation)),
        plain=("sequence", "status"),
        mutable=True,
        since=16,
    ),
    ScheduleUnit: _Spec(
        "schedule_units",
        refs=(("schedule_id", Schedule),),
        plain=("unit_key", "unit_kind", "status"),
        mutable=True,
        since=16,
    ),
    ExecutionAttempt: _Spec(
        "execution_attempts",
        refs=(("unit_id", ScheduleUnit), ("schedule_run_id", ScheduleRun), ("run_id", Run)),
        plain=("attempt", "outcome"),
        optional=("run_id",),
        since=16,
    ),
    UnitStateTransition: _Spec(
        "unit_state_transitions",
        refs=(("unit_id", ScheduleUnit), ("schedule_run_id", ScheduleRun)),
        plain=("sequence", "from_status", "to_status"),
        since=16,
    ),
    EvidenceGraph: _Spec("evidence_graphs", plain=("spec_id",), since=17),
    GraphSnapshot: _Spec(
        "graph_snapshots",
        refs=(("graph_id", EvidenceGraph), ("investigation_id", Investigation), ("run_id", Run)),
        plain=("source_fingerprint",),
        since=17,
    ),
    GraphNode: _Spec(
        "graph_nodes",
        refs=(("snapshot_id", GraphSnapshot),),
        plain=("node_kind", "ref_id"),
        since=17,
    ),
    GraphEdge: _Spec(
        "graph_edges",
        refs=(
            ("snapshot_id", GraphSnapshot),
            ("from_node_id", GraphNode),
            ("to_node_id", GraphNode),
        ),
        plain=("relation",),
        since=17,
    ),
    Claim: _Spec("claims", refs=(("investigation_id", Investigation),), plain=("status",)),
    Evidence: _Spec(
        "evidence",
        refs=(("claim_id", Claim),),
        plain=("target_kind", "target_id", "relation"),
    ),
}


def _immutable_trigger(spec: _Spec) -> str:
    guarded = ", ".join(("id", *(c for c, _ in spec.refs))) if spec.mutable else ""
    of = f" OF {guarded}" if spec.mutable else ""
    return (
        f"CREATE TRIGGER {spec.table}_immutable BEFORE UPDATE{of} ON {spec.table} "
        "BEGIN SELECT RAISE(ABORT, 'registry records are immutable'); END"
    )


def _ddl(upto: int = DB_SCHEMA_VERSION, since: int = 1) -> list[str]:
    """Statements creating the tables introduced in schema versions since..upto."""
    out: list[str] = []
    for spec in _SPECS.values():
        if not since <= spec.since <= upto:
            continue
        t = spec.table
        body = [
            "id TEXT PRIMARY KEY",
            *(f"{c} TEXT" + ("" if c in spec.optional else " NOT NULL") for c in spec.columns),
            "payload TEXT NOT NULL",
            "content_hash TEXT NOT NULL",
            *(f"FOREIGN KEY ({c}) REFERENCES {_SPECS[e].table}(id)" for c, e in spec.refs),
        ]
        out.append(f"CREATE TABLE {t} ({', '.join(body)})")
        out.extend(f"CREATE INDEX ix_{t}_{c} ON {t}({c})" for c in spec.columns)
        out.append(
            f"CREATE TRIGGER {t}_no_delete BEFORE DELETE ON {t} "
            "BEGIN SELECT RAISE(ABORT, 'registry records cannot be deleted'); END"
        )
        out.append(_immutable_trigger(spec))
    return out


def _rewrite_payloads(
    conn: sqlite3.Connection, cls: type[Entity], fn: Callable[[dict[str, object]], None]
) -> None:
    """Migration helper: rewrite every stored payload of `cls` in place (IDs are unchanged, the
    content hash is recomputed). Lifts the immutability trigger only for the duration."""
    spec = _SPECS[cls]
    conn.execute(f"DROP TRIGGER {spec.table}_immutable")
    for row_id, payload in conn.execute(f"SELECT id, payload FROM {spec.table}").fetchall():  # noqa: S608
        data = json.loads(payload)
        fn(data)
        conn.execute(
            f"UPDATE {spec.table} SET payload = ?, content_hash = ? WHERE id = ?",  # noqa: S608
            (canonical_json(data), content_hash(data), row_id),
        )
    conn.execute(_immutable_trigger(spec))


def _migrate_1_to_2(conn: sqlite3.Connection) -> None:
    """Phase 2: add provenance/outcomes; artifacts gain `name` (= their path) and `category`."""
    for statement in _ddl(upto=2, since=2):
        conn.execute(statement)

    def artifact(data: dict[str, object]) -> None:
        data.setdefault("name", data["path"])  # v1 had no logical name; the path is all we know
        data.setdefault("category", "OUTPUT")

    _rewrite_payloads(conn, Artifact, artifact)


def _migrate_2_to_3(conn: sqlite3.Connection) -> None:
    """Phase 3: add model/dataset records; provenance gains `inputs` (null for existing runs)."""
    for statement in _ddl(upto=3, since=3):
        conn.execute(statement)

    def provenance(data: dict[str, object]) -> None:
        data.setdefault("inputs", None)

    _rewrite_payloads(conn, Provenance, provenance)


def _migrate_3_to_4(conn: sqlite3.Connection) -> None:
    """Phase 5: fault experiments, trials and analyses (new tables only; no payload changes)."""
    for statement in _ddl(upto=4, since=4):
        conn.execute(statement)


def _migrate_4_to_5(conn: sqlite3.Connection) -> None:
    """Phase 6: failure signals, clusters, modes, evidence and relationships (new tables only)."""
    for statement in _ddl(upto=5, since=5):
        conn.execute(statement)


def _migrate_5_to_6(conn: sqlite3.Connection) -> None:
    """Phase 7: interaction analyses, effects and evidence (new tables only)."""
    for statement in _ddl(upto=6, since=6):
        conn.execute(statement)


def _migrate_6_to_7(conn: sqlite3.Connection) -> None:
    """Phase 8: reliability profiles and their evidence references (new tables only)."""
    for statement in _ddl(upto=7, since=7):
        conn.execute(statement)


def _migrate_7_to_8(conn: sqlite3.Connection) -> None:
    """Phase 9: benchmark definitions, results and units (new tables only)."""
    for statement in _ddl(upto=8, since=8):
        conn.execute(statement)


def _migrate_8_to_9(conn: sqlite3.Connection) -> None:
    """Phase 10: statistical analyses (new table only)."""
    for statement in _ddl(upto=9, since=9):
        conn.execute(statement)


def _migrate_9_to_10(conn: sqlite3.Connection) -> None:
    """Phase 11: slice definitions and slice analyses (new tables only)."""
    for statement in _ddl(upto=10, since=10):
        conn.execute(statement)


def _migrate_10_to_11(conn: sqlite3.Connection) -> None:
    """Phase 12: temporal window definitions and drift analyses (new tables only)."""
    for statement in _ddl(upto=11, since=11):
        conn.execute(statement)


def _migrate_11_to_12(conn: sqlite3.Connection) -> None:
    """Phase 13: data quality analyses and check results (new tables only)."""
    for statement in _ddl(upto=12, since=12):
        conn.execute(statement)


def _migrate_12_to_13(conn: sqlite3.Connection) -> None:
    """Phase 14: stress analyses and stress trials (new tables only)."""
    for statement in _ddl(upto=13, since=13):
        conn.execute(statement)


def _migrate_13_to_14(conn: sqlite3.Connection) -> None:
    """Phase 15: calibration analyses and per-context calibration results (new tables only)."""
    for statement in _ddl(upto=14, since=14):
        conn.execute(statement)


def _migrate_14_to_15(conn: sqlite3.Connection) -> None:
    """Phase 16: resource analyses and per-pass resource trials (new tables only)."""
    for statement in _ddl(upto=15, since=15):
        conn.execute(statement)


def _migrate_15_to_16(conn: sqlite3.Connection) -> None:
    """Phase 17: schedules, schedule runs, schedule units, execution attempts and unit state
    transitions (new tables only)."""
    for statement in _ddl(upto=16, since=16):
        conn.execute(statement)


def _migrate_16_to_17(conn: sqlite3.Connection) -> None:
    """Phase 17-18: evidence graphs, graph snapshots, graph nodes and graph edges (new tables
    only)."""
    for statement in _ddl(upto=17, since=17):
        conn.execute(statement)


_MIGRATIONS: dict[int, Callable[[sqlite3.Connection], None]] = {
    1: _migrate_1_to_2,
    2: _migrate_2_to_3,
    3: _migrate_3_to_4,
    4: _migrate_4_to_5,
    5: _migrate_5_to_6,
    6: _migrate_6_to_7,
    7: _migrate_7_to_8,
    8: _migrate_8_to_9,
    9: _migrate_9_to_10,
    10: _migrate_10_to_11,
    11: _migrate_11_to_12,
    12: _migrate_12_to_13,
    13: _migrate_13_to_14,
    14: _migrate_14_to_15,
    15: _migrate_15_to_16,
    16: _migrate_16_to_17,
}


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
            self._init_schema(path)
        except Exception:
            self._conn.close()
            raise

    def _init_schema(self, path: str | Path) -> None:
        (found,) = self._conn.execute("PRAGMA user_version").fetchone()
        if found == DB_SCHEMA_VERSION:
            return
        if found == 0:
            (tables,) = self._conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
            if tables:
                raise SchemaVersionError(
                    "database is non-empty but has no EXPERIONYX schema version"
                )
            self._run_atomically(lambda: [self._conn.execute(s) for s in _ddl()], DB_SCHEMA_VERSION)
        elif found in _MIGRATIONS:
            self._migrate(found, path)
        else:
            raise SchemaVersionError(
                f"unsupported database schema version {found} (supported: 1..{DB_SCHEMA_VERSION})"
            )

    def _run_atomically(self, work: Callable[[], object], version: int) -> None:
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            work()
            self._conn.execute(f"PRAGMA user_version = {version}")
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        self._conn.execute("COMMIT")

    def _migrate(self, found: int, path: str | Path) -> None:
        """Upgrade one version at a time inside a single transaction: all steps or none. A file
        database is copied to `<name>.v<found>.bak` first."""
        if str(path) != ":memory:":
            shutil.copy2(path, Path(path).with_name(f"{Path(path).name}.v{found}.bak"))

        def steps() -> None:
            for version in range(found, DB_SCHEMA_VERSION):
                _MIGRATIONS[version](self._conn)

        self._run_atomically(steps, DB_SCHEMA_VERSION)

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
                    evidence_target_type(entity.target_kind),
                    entity.target_id,
                    "evidence.target_id",
                )
            elif isinstance(entity, Provenance):
                self._check_provenance(entity)
            elif isinstance(entity, RunOutcome):
                self._check_outcome(entity)
            elif isinstance(entity, RegisteredModel | RegisteredDataset):
                self._check_unique_binding(entity)
            elif isinstance(entity, FaultTrial):
                self._check_fault_trial(entity)
            elif isinstance(entity, FailureSignal):
                self._check_failure_signal(entity)
            elif isinstance(entity, FailureCluster):
                for sid in entity.signal_ids:
                    self._require(FailureSignal, sid, "failure_clusters.signal_ids")
            elif isinstance(entity, FailureEvidence):
                self._check_failure_evidence(entity)
            elif isinstance(entity, FailureRelationship):
                self._check_failure_relationship(entity)
            elif isinstance(entity, InteractionAnalysis):
                self._check_interaction_analysis(entity)
            elif isinstance(entity, ReliabilityProfile):
                self._check_reliability_profile(entity)
            elif isinstance(entity, BenchmarkResult):
                self._check_benchmark_result(entity)
            elif isinstance(entity, ScheduleUnit):
                for dep in entity.depends_on:
                    self._require(ScheduleUnit, dep, "schedule_units.depends_on")
            elif isinstance(entity, ExecutionAttempt):
                self._check_execution_attempt(entity)
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

    def _check_execution_attempt(self, a: ExecutionAttempt) -> None:
        unit = self.get(ScheduleUnit, a.unit_id)
        run = self.get(ScheduleRun, a.schedule_run_id)
        if unit.schedule_id != run.schedule_id:
            raise ValidationError(
                "execution attempt's unit and schedule run belong to different schedules"
            )
        if a.attempt != len(self.find(ExecutionAttempt, unit_id=a.unit_id)):
            raise ValidationError(
                f"execution attempt {a.attempt} is not the next attempt for {a.unit_id}"
            )

    def _check_benchmark_result(self, r: BenchmarkResult) -> None:
        run = self.get(Run, r.run_id)
        if self.get(Experiment, run.experiment_id).investigation_id != r.investigation_id:
            raise ValidationError("benchmark result run belongs to a different investigation")
        if self.get(Benchmark, r.benchmark_id).protocol_hash != r.protocol_hash:
            raise ValidationError("benchmark result protocol hash differs from its benchmark")

    def _check_reliability_profile(self, p: ReliabilityProfile) -> None:
        run = self.get(Run, p.run_id)
        if self.get(Experiment, run.experiment_id).investigation_id != p.investigation_id:
            raise ValidationError("reliability profile run belongs to a different investigation")

    def _check_interaction_analysis(self, a: InteractionAnalysis) -> None:
        run = self.get(Run, a.run_id)
        if self.get(Experiment, run.experiment_id).investigation_id != a.investigation_id:
            raise ValidationError("interaction analysis run belongs to a different investigation")

    def _check_failure_signal(self, sig: FailureSignal) -> None:
        run = self.get(Run, sig.run_id)
        if run.experiment_id != sig.experiment_id:
            raise ValidationError("failure signal experiment_id does not match its run")

    def _check_failure_evidence(self, ev: FailureEvidence) -> None:
        if ev.ref_id is None:
            return
        target = {"SIGNAL": FailureSignal, "RUN": Run}.get(ev.evidence_kind.value)
        if target is not None:
            self._require(target, ev.ref_id, "failure_evidence.ref_id")

    def _check_failure_relationship(self, rel: FailureRelationship) -> None:
        nodes: dict[NodeKind, type[Entity]] = {
            NodeKind.MODEL: RegisteredModel,
            NodeKind.DATASET: RegisteredDataset,
            NodeKind.FAILURE_MODE: FailureMode,
            NodeKind.RUN: Run,
            NodeKind.INTERACTION: InteractionAnalysis,
        }  # CLASS, SLICE and FAULT nodes are descriptors, not registry records
        for kind, node_id in ((rel.subject_kind, rel.subject_id), (rel.object_kind, rel.object_id)):
            if kind is NodeKind.EVIDENCE:
                if not (
                    self.exists(FailureEvidence, node_id)
                    or self.exists(InteractionEvidence, node_id)
                ):
                    raise MissingReferenceError(
                        f"failure_relationships EVIDENCE references missing evidence {node_id}"
                    )
            elif kind in nodes:
                self._require(nodes[kind], node_id, f"failure_relationships {kind.value}")
            if (
                kind is NodeKind.FAILURE_MODE
                and NodeKind.INTERACTION not in (rel.subject_kind, rel.object_kind)
                and self.get(FailureMode, node_id).investigation_id != rel.investigation_id
            ):
                raise ValidationError("relationship crosses investigations")

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
        if p.inputs is not None:
            if p.inputs.model is not None:
                model = self.get(RegisteredModel, p.inputs.model.record_id)
                self._check_binding(p.inputs.model.fingerprint, exp.model.digest, model.fingerprint)
            if p.inputs.dataset is not None:
                data = self.get(RegisteredDataset, p.inputs.dataset.record_id)
                self._check_binding(
                    p.inputs.dataset.fingerprint, exp.dataset.digest, data.fingerprint
                )

    @staticmethod
    def _check_binding(bound: str, referenced: str | None, registered: str) -> None:
        if bound != registered or referenced != registered:
            raise ValidationError(
                "provenance input does not match the experiment's registered reference"
            )

    def _check_fault_trial(self, t: FaultTrial) -> None:
        """A trial must share its fault experiment's control run and, if it has a treatment run,
        that run must belong to the treatment experiment it names."""
        fx = self.get(FaultExperiment, t.fault_experiment_id)
        if t.baseline_run_id != fx.baseline_run_id:
            raise ValidationError(
                "a trial's control run must be its fault experiment's baseline run"
            )
        if t.treatment_run_id is not None:
            run = self.get(Run, t.treatment_run_id)
            if t.treatment_experiment_id != run.experiment_id:
                raise ValidationError(
                    "treatment run does not belong to the named treatment experiment"
                )

    def _check_unique_binding(self, record: RegisteredModel | RegisteredDataset) -> None:
        """A (name, version) may only ever refer to one fingerprint per adapter version."""
        clashes = [
            r
            for r in self.find(type(record), name=record.name, version=record.version)
            if r.adapter == record.adapter
            and r.adapter_version == record.adapter_version
            and r.fingerprint != record.fingerprint
        ]
        if clashes:
            raise ValidationError(
                f"{record.name} {record.version} is already registered with a different "
                "fingerprint; use a new version for changed content"
            )

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

    def update_status(
        self,
        entity: (
            Experiment
            | Run
            | FaultExperiment
            | FailureMode
            | InteractionAnalysis
            | ScheduleRun
            | ScheduleUnit
        ),
    ) -> None:
        with self.transaction():
            current: (
                Experiment
                | Run
                | FaultExperiment
                | FailureMode
                | InteractionAnalysis
                | ScheduleRun
                | ScheduleUnit
            )
            expected: (
                Experiment
                | Run
                | FaultExperiment
                | FailureMode
                | InteractionAnalysis
                | ScheduleRun
                | ScheduleUnit
            )
            if isinstance(entity, Experiment):
                current = self.get(Experiment, entity.id)
                expected = current.with_status(entity.status)
            elif isinstance(entity, FaultExperiment):
                current = self.get(FaultExperiment, entity.id)
                expected = current.with_status(entity.status)
            elif isinstance(entity, FailureMode):
                current = self.get(FailureMode, entity.id)
                expected = current.with_status(entity.status)
            elif isinstance(entity, InteractionAnalysis):
                current = self.get(InteractionAnalysis, entity.id)
                expected = current.with_status(entity.status)
            elif isinstance(entity, ScheduleRun):
                current = self.get(ScheduleRun, entity.id)
                expected = current.with_status(entity.status)
            elif isinstance(entity, ScheduleUnit):
                current = self.get(ScheduleUnit, entity.id)
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
