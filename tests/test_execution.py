import hashlib
import importlib.util
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

import procedures
from conftest import Lab, make_lab, make_repo
from experionyx.artifacts import LocalArtifactStore
from experionyx.domain import (
    Artifact,
    ArtifactCategory,
    Entity,
    EnvironmentSnapshot,
    Experiment,
    ExperimentStatus,
    Observation,
    Run,
    RunStatus,
)
from experionyx.errors import (
    NotFoundError,
    PreparationError,
    ReplayError,
    ValidationError,
)
from experionyx.execution import (
    Executor,
    _process_alive,
    resolve_procedure,
    run_states,
)
from experionyx.failures.entities import FailureMode
from experionyx.faults.entities import FaultExperiment
from experionyx.provenance import (
    FailureStage,
    Provenance,
    ResourceLimits,
    RunOutcome,
    SourceState,
    compare_provenance,
)
from experionyx.sqlite import SqliteRegistry


def outcome_of(lab: Lab, run: Run) -> RunOutcome:
    (o,) = lab.registry.find(RunOutcome, run_id=run.id)
    return o


# --- successful execution ---------------------------------------------------------------------


def test_successful_execution_leaves_a_complete_durable_record(lab: Lab) -> None:
    result = lab.executor.execute(lab.experiment.id, procedures.ok, seed=4)
    assert result.status is RunStatus.COMPLETED
    assert result.error is None
    assert result.duration_seconds is not None
    assert result.duration_seconds >= 0

    # reopen from disk: everything below comes from the persisted registry
    reg = SqliteRegistry(lab.workspace / "registry.sqlite")
    run = reg.get(Run, result.run.id)
    assert run.status is RunStatus.COMPLETED
    assert run.seed == 4
    assert run.experiment_id == lab.experiment.id

    env = reg.get(EnvironmentSnapshot, run.environment_id)
    assert env.python_version == sys.version.split()[0]
    assert "experionyx" in env.packages

    (prov,) = reg.find(Provenance, run_id=run.id)
    assert prov.seed == 4
    assert prov.configuration_id == lab.configuration.id
    assert prov.environment_id == env.id
    assert prov.execution.procedure == "procedures:ok"
    assert prov.source.state is SourceState.UNKNOWN  # source root is not a git repo
    assert prov.replay_of is None
    expected_seeded = (
        ("python.random", "numpy.random")
        if importlib.util.find_spec("numpy")
        else ("python.random",)
    )
    assert prov.runtime["seeded_libraries"] == expected_seeded  # frozen: lists become tuples

    obs = {o.name: o for o in reg.find(Observation, run_id=run.id)}
    assert obs["score"].value == 2.0
    assert obs["score"].unit == "ratio"
    assert obs["curve"].value == {"xs": (1, 2, 3)}
    assert obs["draw"].run_id == run.id

    (art,) = reg.find(Artifact, run_id=run.id)
    file = (
        lab.workspace
        / "experiments"
        / run.experiment_id
        / "runs"
        / run.id
        / "artifacts"
        / "result.txt"
    )
    assert art.digest == "sha256:" + hashlib.sha256(file.read_bytes()).hexdigest()
    assert art.size_bytes == file.stat().st_size

    out = outcome_of(lab, run)
    assert out.status is RunStatus.COMPLETED
    assert out.observation_count == 3
    assert out.artifact_ids == (art.id,)
    assert out.error is None
    reg.close()


def test_metadata_files_mirror_the_registry(lab: Lab) -> None:
    result = lab.executor.execute(lab.experiment.id, procedures.ok, seed=1)
    meta = lab.store.run_dir(result.run) / "metadata"
    prov = json.loads((meta / "provenance.json").read_text())
    assert prov["fingerprint"] == result.provenance.fingerprint
    assert prov["run_id"] == result.run.id
    assert json.loads((meta / "outcome.json").read_text())["status"] == "COMPLETED"


def test_repository_state_is_recorded_in_provenance(tmp_path: Path) -> None:
    clean = make_lab(tmp_path / "w1", make_repo(tmp_path / "clean"))
    dirty = make_lab(tmp_path / "w2", make_repo(tmp_path / "dirty", dirty=True))
    a = clean.executor.execute(clean.experiment.id, procedures.nothing, seed=0)
    b = dirty.executor.execute(dirty.experiment.id, procedures.nothing, seed=0)
    assert a.provenance.source.state is SourceState.REPRODUCIBLE_SOURCE
    assert b.provenance.source.state is SourceState.MODIFIED_WORKTREE
    assert a.provenance.source.commit
    assert b.provenance.source.commit
    assert a.provenance.fingerprint != b.provenance.fingerprint
    clean.registry.close()
    dirty.registry.close()


def test_observation_sequences_increase_per_name(lab: Lab) -> None:
    def proc(ctx) -> None:  # type: ignore[no-untyped-def]
        for i in range(3):
            ctx.observe("loss", 1.0 / (i + 1))
        ctx.observe("other", "x")

    result = lab.executor.execute(lab.experiment.id, proc, seed=0, procedure_name="t:proc")
    seqs = sorted((o.name, o.sequence) for o in result.observations)
    assert seqs == [("loss", 0), ("loss", 1), ("loss", 2), ("other", 0)]


def test_resource_limits_and_metadata_are_recorded(lab: Lab) -> None:
    result = lab.executor.execute(
        lab.experiment.id,
        procedures.nothing,
        seed=0,
        resources=ResourceLimits(max_workers=2, timeout_seconds=30.0),
        metadata={"note": "x"},
    )
    assert result.provenance.execution.resources.max_workers == 2
    assert result.provenance.execution.metadata == {"note": "x"}


# --- seeds ------------------------------------------------------------------------------------


def test_same_seed_gives_same_python_randomness_and_different_seed_differs(lab: Lab) -> None:
    def draw(seed: int) -> object:
        r = lab.executor.execute(lab.experiment.id, procedures.ok, seed=seed)
        return next(o.value for o in r.observations if o.name == "draw")

    assert draw(11) == draw(11)
    assert draw(11) != draw(12)


# --- failures ---------------------------------------------------------------------------------


def test_failed_experiment_stays_visible_with_structured_diagnostics(lab: Lab) -> None:
    result = lab.executor.execute(lab.experiment.id, procedures.fails, seed=2)
    assert result.status is RunStatus.FAILED

    run = lab.registry.get(Run, result.run.id)
    assert run.status is RunStatus.FAILED  # not deleted, not hidden
    out = outcome_of(lab, run)
    assert out.error is not None
    assert out.error.stage is FailureStage.EXECUTION
    assert out.error.error_type == "builtins.ValueError"
    assert out.error.message == "boom"
    assert out.error.diagnostic_artifact_id is not None

    diag = lab.registry.get(Artifact, out.error.diagnostic_artifact_id)
    assert diag.category is ArtifactCategory.DIAGNOSTIC
    text = (lab.store.run_dir(run) / "artifacts" / diag.path).read_text()
    assert "ValueError: boom" in text
    assert "procedures.py" in text
    assert str(Path.home()) not in text
    lab.store.verify(run, diag)

    # evidence produced before the failure is retained and consistent
    assert [o.name for o in lab.registry.find(Observation, run_id=run.id)] == ["before_failure"]
    assert out.observation_count == 1
    assert len(out.artifact_ids) == 2  # partial.txt + traceback
    assert lab.registry.find(Run, status="FAILED") == [run]
    assert lab.registry.find(Provenance, run_id=run.id)  # provenance recorded before it ran


def test_registering_a_missing_file_fails_the_run_without_a_phantom_artifact(lab: Lab) -> None:
    result = lab.executor.execute(lab.experiment.id, procedures.registers_missing_file, seed=0)
    assert result.status is RunStatus.FAILED
    assert result.error is not None
    assert result.error.error_type.endswith("ArtifactError")
    arts = lab.registry.find(Artifact, run_id=result.run.id)
    assert [a.category for a in arts] == [ArtifactCategory.DIAGNOSTIC]  # only the traceback


@pytest.mark.parametrize(
    "proc", [procedures.deletes_after_registering, procedures.modifies_after_registering]
)
def test_artifact_changed_after_registration_fails_the_run(lab: Lab, proc: object) -> None:
    result = lab.executor.execute(lab.experiment.id, proc, seed=0)  # type: ignore[arg-type]
    assert result.status is RunStatus.FAILED
    assert result.error is not None
    assert result.error.stage is FailureStage.ARTIFACT_REGISTRATION
    assert result.error.diagnostic_artifact_id is not None
    assert lab.registry.get(Run, result.run.id).status is RunStatus.FAILED


def test_keyboard_interrupt_is_recorded_then_reraised(lab: Lab) -> None:
    with pytest.raises(KeyboardInterrupt):
        lab.executor.execute(lab.experiment.id, procedures.interrupted, seed=0)
    (run,) = lab.registry.find(Run)
    assert run.status is RunStatus.FAILED
    err = outcome_of(lab, run).error
    assert err is not None
    assert err.stage is FailureStage.INTERRUPTED
    assert err.error_type == "builtins.KeyboardInterrupt"
    assert len(lab.registry.find(Observation, run_id=run.id)) == 1


def test_preparation_errors_are_raised_and_create_no_run(lab: Lab) -> None:
    with pytest.raises(NotFoundError):
        lab.executor.execute("exp_" + "0" * 32, procedures.ok, seed=0)
    with pytest.raises(ValidationError):
        lab.executor.execute(lab.experiment.id, procedures.ok, seed=-1)
    with pytest.raises(PreparationError, match="importable name"):
        lab.executor.execute(lab.experiment.id, lambda ctx: None, seed=0)
    assert lab.registry.find(Run) == []


@pytest.mark.parametrize("status", [ExperimentStatus.CANCELLED])
def test_non_executable_experiments_are_refused(lab: Lab, status: ExperimentStatus) -> None:
    lab.registry.update_status(lab.experiment.with_status(status))
    with pytest.raises(PreparationError, match="cannot be executed"):
        lab.executor.execute(lab.experiment.id, procedures.ok, seed=0)


def test_draft_experiment_is_refused(tmp_path: Path) -> None:
    from factories import configuration, experiment, investigation

    reg = SqliteRegistry(tmp_path / "r.sqlite")
    inv, cfg = investigation(), configuration()
    exp = experiment(inv, cfg)
    for e in (inv, cfg, exp):
        reg.add(e)
    executor = Executor(reg, LocalArtifactStore(tmp_path / "x"), source_root=tmp_path)
    with pytest.raises(PreparationError):
        executor.execute(exp.id, procedures.ok, seed=0)
    reg.close()


def test_preparation_failure_after_run_creation_is_recorded(lab: Lab) -> None:
    class BrokenStore(LocalArtifactStore):
        def prepare(self, run: Run) -> Path:
            raise OSError("disk full")

    ex = Executor(lab.registry, BrokenStore(lab.workspace / "x"), source_root=lab.workspace)
    with pytest.raises(PreparationError, match="disk full"):
        ex.execute(lab.experiment.id, procedures.ok, seed=0)
    (run,) = lab.registry.find(Run)
    assert run.status is RunStatus.FAILED
    err = outcome_of(lab, run).error
    assert err is not None
    assert err.stage is FailureStage.PREPARATION
    assert lab.registry.find(Provenance, run_id=run.id) == []


# --- transactional integrity ------------------------------------------------------------------


class FailingRegistry(SqliteRegistry):
    """Real SQLite registry that fails one chosen write, to test partial-failure behaviour."""

    fail_on_outcome = False
    fail_on_final_status = False

    def add(self, entity: Entity) -> None:
        if self.fail_on_outcome and isinstance(entity, RunOutcome):
            raise OSError("simulated persistence failure")
        super().add(entity)

    def update_status(self, entity: Experiment | Run | FaultExperiment | FailureMode) -> None:
        if self.fail_on_final_status and entity.status is RunStatus.COMPLETED:
            raise OSError("simulated persistence failure")
        super().update_status(entity)


def _flaky_lab(tmp_path: Path) -> tuple[Lab, FailingRegistry]:
    from factories import configuration, experiment, investigation

    reg = FailingRegistry(tmp_path / "flaky.sqlite")
    store = LocalArtifactStore(tmp_path / "exp")
    inv, cfg = investigation(), configuration()
    exp = experiment(inv, cfg)
    for e in (inv, cfg, exp):
        reg.add(e)
    reg.update_status(exp.with_status(ExperimentStatus.READY))
    lab = Lab(
        reg,
        store,
        Executor(reg, store, source_root=tmp_path),
        inv,
        cfg,
        reg.get(Experiment, exp.id),
        tmp_path,
    )
    return lab, reg


@pytest.mark.parametrize("which", ["fail_on_outcome", "fail_on_final_status"])
def test_run_is_never_completed_if_finalization_fails(tmp_path: Path, which: str) -> None:
    lab, reg = _flaky_lab(tmp_path)
    setattr(reg, which, True)
    with pytest.raises(OSError, match="simulated"):
        lab.executor.execute(lab.experiment.id, procedures.ok, seed=0)
    (run,) = reg.find(Run)
    assert run.status is RunStatus.RUNNING  # not COMPLETED, not half-recorded
    assert reg.find(RunOutcome) == []  # outcome rolled back together with the status write
    assert len(reg.find(Artifact, run_id=run.id)) == 1  # already-registered evidence survives
    reg.close()


def test_recover_interrupted_closes_runs_whose_process_is_gone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lab, reg = _flaky_lab(tmp_path)
    reg.fail_on_outcome = True
    with pytest.raises(OSError, match="simulated"):
        lab.executor.execute(lab.experiment.id, procedures.ok, seed=0)
    reg.fail_on_outcome = False

    assert lab.executor.recover_interrupted() == []  # our own pid is alive: leave it alone
    monkeypatch.setattr("experionyx.execution._process_alive", lambda pid: False)
    (recovered,) = lab.executor.recover_interrupted()
    assert recovered.status is RunStatus.FAILED
    out = reg.find(RunOutcome, run_id=recovered.id)[0]
    assert out.error is not None
    assert out.error.stage is FailureStage.INTERRUPTED
    assert out.duration_seconds is None
    assert out.observation_count == 3
    assert len(out.artifact_ids) == 1
    assert lab.executor.recover_interrupted() == []  # idempotent
    reg.close()


def test_process_liveness_uses_real_pids() -> None:
    assert _process_alive(os.getpid())
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    assert not _process_alive(proc.pid)


def test_run_states_groups_runs_for_resumption(lab: Lab) -> None:
    lab.executor.execute(lab.experiment.id, procedures.ok, seed=0)
    lab.executor.execute(lab.experiment.id, procedures.fails, seed=1)
    states = run_states(lab.registry, lab.experiment.id)
    assert {s: len(r) for s, r in states.items()} == {
        RunStatus.PENDING: 0,
        RunStatus.RUNNING: 0,
        RunStatus.COMPLETED: 1,
        RunStatus.FAILED: 1,
    }


# --- retries and replay -----------------------------------------------------------------------


def test_repeating_an_execution_creates_a_distinct_run_and_never_overwrites(lab: Lab) -> None:
    first = lab.executor.execute(lab.experiment.id, procedures.fails, seed=3)
    snapshot = lab.registry.get(Run, first.run.id)
    second = lab.executor.execute(lab.experiment.id, procedures.fails, seed=3)
    assert second.run.id != first.run.id
    assert (first.run.attempt, second.run.attempt) == (0, 1)
    assert lab.registry.get(Run, first.run.id) == snapshot
    assert len(lab.registry.find(RunOutcome)) == 2


def test_replay_creates_a_new_run_and_leaves_the_original_untouched(lab: Lab) -> None:
    original = lab.executor.execute(lab.experiment.id, procedures.ok, seed=6)
    before = (
        lab.registry.get(Run, original.run.id),
        lab.registry.get(RunOutcome, original.outcome.id),
        lab.registry.get(Provenance, original.provenance.id),
        lab.registry.find(Observation, run_id=original.run.id),
        lab.registry.find(Artifact, run_id=original.run.id),
    )
    replay = lab.executor.replay(original.run.id)

    assert replay.run.id != original.run.id
    assert replay.run.status is RunStatus.COMPLETED
    assert replay.provenance.replay_of == original.run.id
    assert replay.provenance.is_replay
    assert replay.provenance.seed == original.provenance.seed == 6
    assert replay.provenance.configuration_id == original.provenance.configuration_id
    assert replay.provenance.execution == original.provenance.execution
    assert replay.run.attempt == 1
    # same inputs => same fingerprint; a replay request does not claim more than that
    assert compare_provenance(original.provenance, replay.provenance) == ()
    assert replay.provenance.fingerprint == original.provenance.fingerprint
    # independent results are recorded for the new run
    (new_art,) = replay.artifacts
    (old_art,) = original.artifacts
    assert new_art.id != old_art.id
    assert new_art.digest == old_art.digest  # this procedure happens to be deterministic

    after = (
        lab.registry.get(Run, original.run.id),
        lab.registry.get(RunOutcome, original.outcome.id),
        lab.registry.get(Provenance, original.provenance.id),
        lab.registry.find(Observation, run_id=original.run.id),
        lab.registry.find(Artifact, run_id=original.run.id),
    )
    assert after == before


def test_replay_of_a_failed_run_is_an_explicit_retry(lab: Lab) -> None:
    failed = lab.executor.execute(lab.experiment.id, procedures.fails, seed=0)
    retry = lab.executor.replay(failed.run.id)
    assert retry.provenance.replay_of == failed.run.id
    assert retry.run.id != failed.run.id
    assert lab.registry.get(Run, failed.run.id).status is RunStatus.FAILED


def test_replay_can_override_the_procedure_and_it_shows_in_the_fingerprint(lab: Lab) -> None:
    original = lab.executor.execute(lab.experiment.id, procedures.ok, seed=1)
    other = lab.executor.replay(original.run.id, procedures.nothing)
    assert compare_provenance(original.provenance, other.provenance) == ("execution",)


def test_replay_rejects_unfinished_unknown_and_unresolvable_runs(lab: Lab) -> None:
    with pytest.raises(NotFoundError):
        lab.executor.replay("run_" + "0" * 32)
    closure = lab.executor.execute(
        lab.experiment.id, lambda ctx: None, seed=0, procedure_name="not.a.module:f"
    )
    with pytest.raises(ReplayError, match="cannot replay"):
        lab.executor.replay(closure.run.id)
    pending = Run(lab.experiment.id, closure.run.environment_id, 42, closure.run.created_at)
    lab.registry.add(pending)
    with pytest.raises(ReplayError, match="finished"):
        lab.executor.replay(pending.id)


def test_resolve_procedure() -> None:
    assert resolve_procedure("procedures:ok") is procedures.ok
    for bad in ("procedures", "procedures:", ":ok", "no.such.module:f", "procedures:missing"):
        with pytest.raises(PreparationError):
            resolve_procedure(bad)
    with pytest.raises(PreparationError, match="not callable"):
        resolve_procedure("procedures:random")


def test_random_module_is_the_global_one_seeded_per_run(lab: Lab) -> None:
    lab.executor.execute(lab.experiment.id, procedures.nothing, seed=123)
    a = random.random()
    random.seed(123)
    assert random.random() == a
