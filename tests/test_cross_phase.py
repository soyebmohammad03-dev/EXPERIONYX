"""One REAL workspace, every phase, one chain: the iris/scikit-learn baseline (Phases 1-4), the fault
laboratory (5), failure discovery (6), slices (11), temporal shift (12), data quality (13), model stress
(14) and a reliability profile (8) that references them, then replays and provenance checks across the
phases. Also CLI smoke coverage: every command group must at least load its help, and the read-only
commands must run against the populated workspace. An engineering integration test of the machinery;
not a finding about iris."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("sklearn")

from experionyx.adapters.records import RegisteredDataset, RegisteredModel
from experionyx.adapters.registry import default_registries
from experionyx.artifacts import LocalArtifactStore
from experionyx.cli import main
from experionyx.data_quality.entities import QualityAnalysis
from experionyx.demos import _sklearn_config
from experionyx.domain import Investigation, Run, to_jsonable
from experionyx.drift.entities import DriftAnalysis
from experionyx.execution import Executor
from experionyx.failures.engine import run_discovery
from experionyx.faults.demos import run_fault_demo
from experionyx.faults.entities import FaultExperiment
from experionyx.reliability.engine import run_profile
from experionyx.reliability.registry import ReliabilityProfileRegistry
from experionyx.reliability.spec import ProfileSpec
from experionyx.reliability.taxonomy import Scope
from experionyx.slices.entities import SliceAnalysis
from experionyx.sqlite import SqliteRegistry
from experionyx.stress.entities import StressAnalysis

FEATS = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]


@dataclass
class World:
    ws: Path
    baseline: str
    fxps: tuple[str, ...]
    mid: str
    did: str
    inv: str


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> World:
    ws = tmp_path_factory.mktemp("cross") / "w"
    results = run_fault_demo("tabular", ws)  # baseline + three real fault sweeps on iris
    with SqliteRegistry(ws / "registry.sqlite") as reg:
        fx0 = reg.get(FaultExperiment, results[0].fault_experiment.id)
        return World(
            ws,
            fx0.baseline_run_id,
            tuple(r.fault_experiment.id for r in results),
            reg.find(RegisteredModel)[0].id,
            reg.find(RegisteredDataset)[0].id,
            fx0.investigation_id,
        )


def cli(w: World, capsys: pytest.CaptureFixture[str], *args: str) -> tuple[int, str, str]:
    code = main(["--workspace", str(w.ws), *args])
    out = capsys.readouterr()
    return code, out.out, out.err


def write(tmp_path: Path, name: str, body: Any) -> str:
    p = tmp_path / name
    p.write_text(json.dumps(body), encoding="utf-8")
    return str(p)


def test_every_phase_runs_on_one_workspace_and_replays(
    world: World, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    w = world
    ev = to_jsonable(_sklearn_config(False))
    with SqliteRegistry(w.ws / "registry.sqlite") as reg:
        store = LocalArtifactStore(w.ws / "experiments")
        ex = Executor(
            reg, store, source_root=Path.cwd(), adapters=default_registries(), inputs_root=w.ws
        )
        disc = run_discovery(
            reg, store, ex, w.inv, [], list(w.fxps)
        )  # Phase 6 over the Phase 5 experiments
        assert disc.status.value == "COMPLETED"
    # Phase 11: slices, reading the fault experiments
    sl = {"baseline_run": w.baseline, "slices": [{"name": "wide", "condition": {"op": "range", "field": "feature:petal width (cm)", "low": 1.0, "high": None}}], "fault_experiments": list(w.fxps), "config": {"resamples": 100}}  # fmt: skip
    code, out, _ = cli(w, capsys, "slice", "analyze", write(tmp_path, "sl.json", sl))
    slice_id = json.loads(out)["analysis_id"]
    assert code in (0, 3) and slice_id.startswith("san_")
    # Phase 12: drift between two explicit index windows
    dr = {"baseline_run": w.baseline, "ordering": {"field": "index"}, "reference": {"start": 0, "end": 75}, "comparisons": [{"start": 75, "end": 150}], "features": [{"name": n, "type": "NUMERIC"} for n in FEATS], "config": {"min_samples": 5, "resamples": 100, "permutations": 200}}  # fmt: skip
    code, out, _ = cli(w, capsys, "drift", "evaluate", write(tmp_path, "dr.json", dr))
    drift_id = json.loads(out)["analysis_id"]
    assert code in (0, 3) and drift_id.startswith("dan_")
    # Phase 13: data quality on the registered dataset
    dq = {"dataset_id": w.did, "splits": ["test", "train"], "target": {"task": "CLASSIFICATION"}, "features": [{"name": n, "type": "NUMERIC"} for n in FEATS], "checks": [{"type": "missingness", "config": {}}, {"type": "duplicates", "config": {}}, {"type": "target", "config": {}}, {"type": "split_overlap", "config": {"reference": "train", "comparison": "test"}}], "config": {"min_members": 10, "resamples": 50, "permutations": 100}}  # fmt: skip
    code, out, err = cli(
        w, capsys, "data-quality", "run", write(tmp_path, "dq.json", dq), "--investigation", w.inv
    )
    assert out, err
    quality_id = json.loads(out)["analysis_id"]
    assert code in (0, 3) and quality_id.startswith("qan_")
    # Phase 14: stress, against the SAME baseline the faults used
    st = {"model_id": w.mid, "dataset_id": w.did, "baseline_run": w.baseline, "evaluation": ev, "plan": {"components": [{"family": "PARAMETER_NOISE", "parameters": {"relative_sigma": 0.5}}], "seeds": [1, 2]}, "min_members": 5, "resamples": 100, "permutations": 100, "discover_failures": True}  # fmt: skip
    code, out, _ = cli(w, capsys, "stress", "run", write(tmp_path, "st.json", st))
    stress_doc = json.loads(out)
    stress_id = stress_doc["analysis_id"]
    assert (
        code in (0, 3)
        and stress_id.startswith("sxa_")
        and stress_doc["summary"]["coverage"]["completed"] == 2
    )
    assert stress_doc["summary"]["links"]["failure_discovery"]["status"] == "COMPLETED"
    with SqliteRegistry(w.ws / "registry.sqlite") as reg:
        store = LocalArtifactStore(w.ws / "experiments")
        ex = Executor(
            reg, store, source_root=Path.cwd(), adapters=default_registries(), inputs_root=w.ws
        )
        modes = tuple(
            m.id
            for m in reg.find(
                __import__("experionyx.failures.entities", fromlist=["FailureMode"]).FailureMode
            )
        )
        spec = ProfileSpec(
            Scope.MODEL_DATASET_EVALUATION,
            w.baseline,
            w.fxps,
            (),
            modes[:3],
            (slice_id,),
            (drift_id,),
            (stress_id,),
        )
        prof = run_profile(
            reg, store, ex, w.inv, spec
        )  # Phase 8, one profile referencing four later phases
        assert prof.profile_id
        doc = ReliabilityProfileRegistry(reg, store).document(prof.profile_id)
        dims = doc["dimensions"]
        assert (
            dims["SLICE_SENSITIVITY"]["status"]
            and dims["DISTRIBUTION_SHIFT"]["status"] == "DERIVED"
            and dims["MODEL_STRESS"]["status"] in ("DERIVED", "INSUFFICIENT_EVIDENCE")
        )
        assert "no overall score" in doc["no_score"] and all("score" not in k.lower() for k in dims)
        assert (
            reg.get(SliceAnalysis, slice_id).baseline_run_id
            == reg.get(DriftAnalysis, drift_id).baseline_run_id
            == reg.get(StressAnalysis, stress_id).baseline_run_id
            == w.baseline
        )
        assert (
            reg.get(QualityAnalysis, quality_id).dataset_id
            == w.did
            == reg.get(StressAnalysis, stress_id).dataset_id
        )
        assert (
            len(
                {
                    a.provenance_fingerprint
                    for a in (
                        reg.get(DriftAnalysis, drift_id),
                        reg.get(QualityAnalysis, quality_id),
                        reg.get(StressAnalysis, stress_id),
                    )
                }
            )
            == 3
        )
        n_before = len(reg.find(Run))
    for group, ident in (("drift", drift_id), ("data-quality", quality_id), ("stress", stress_id)):
        code, out, _ = cli(w, capsys, group, "replay", ident)
        assert code == 0 and json.loads(out)["deterministic"] is True, group
    code, out, _ = cli(w, capsys, "stress", "compare", stress_id)
    assert code == 0 and json.loads(out)["reproduced"] is True
    code, out, _ = cli(w, capsys, "reliability", "replay", prof.profile_id)
    assert code in (0, 1) and json.loads(out)["profile_id"] == prof.profile_id
    with SqliteRegistry(w.ws / "registry.sqlite") as reg:
        assert len(reg.find(Run)) > n_before  # replays are new, separately recorded runs
        assert reg.find(Investigation)


GROUPS = ["fault", "failure", "interaction", "reliability", "benchmark", "stats", "slice", "drift", "data-quality", "stress", "dataset", "model", "evaluation", "adapters", "demo"]  # fmt: skip


@pytest.mark.parametrize("group", GROUPS)
def test_every_command_group_loads_its_help(group: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as e:
        main([group, "--help"])
    assert e.value.code == 0
    assert group in capsys.readouterr().out


def test_read_only_commands_run_against_the_populated_workspace(
    world: World, capsys: pytest.CaptureFixture[str]
) -> None:
    for args in (("info",), ("status",), ("faults",), ("stress", "list"), ("data-quality", "list"), ("drift", "list"), ("slice", "list", "--analyses"), ("stress", "families"), ("data-quality", "checks")):  # fmt: skip
        code, out, err = cli(world, capsys, *args)
        assert code == 0 and "Traceback" not in err, (args, err)
        assert out.strip(), args
