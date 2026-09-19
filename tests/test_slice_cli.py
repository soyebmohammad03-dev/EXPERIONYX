"""The slice CLI and a REAL-DATA scenario: a scikit-learn model trained on the bundled iris dataset,
evaluated by the normal engine, attacked by real seeded faults, mined for failure modes and
analyzed for an interaction. Expected values come from numpy/scikit-learn directly (the same split
the evaluation engine uses), not from the slice engine. Real experimental evidence, on a tiny
dataset: these are correctness checks of the machinery, not scientific findings about iris."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("sklearn")

import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

from experionyx.adapters.registry import default_registries
from experionyx.artifacts import LocalArtifactStore
from experionyx.cli import main
from experionyx.demos import _sklearn_config, run_demo
from experionyx.domain import Experiment, Investigation, Run
from experionyx.execution import Executor
from experionyx.failures.config import DiscoveryConfig
from experionyx.failures.engine import run_discovery
from experionyx.failures.entities import FailureMode
from experionyx.faults.design import FaultDesign, FaultLimits
from experionyx.faults.entities import FaultTrial
from experionyx.faults.lab import run_fault_experiment
from experionyx.faults.library import default_fault_registry
from experionyx.faults.spec import FaultScope, ScopeKind
from experionyx.interactions.config import InteractionConfig, InteractionSpec
from experionyx.interactions.engine import run_interaction
from experionyx.interactions.taxonomy import Pairing
from experionyx.provenance import Provenance
from experionyx.slices.entities import SliceAnalysis
from experionyx.slices.registry import SliceRegistry
from experionyx.sqlite import SqliteRegistry

PW = "feature:petal width (cm)"
WIDE = {"name": "wide-petals", "condition": {"op": "range", "field": PW, "low": 1.0, "high": None}}
CLASS0 = {"name": "class-0", "condition": {"op": "eq", "field": "target", "values": [0]}}


@dataclass
class Iris:
    ws: Path
    baseline: str
    investigation: str
    faults: tuple[str, ...]
    interaction: str
    modes: tuple[str, ...]


@pytest.fixture(scope="module")
def iris(tmp_path_factory: pytest.TempPathFactory) -> Iris:
    ws = tmp_path_factory.mktemp("iris") / "w"
    base = run_demo("sklearn-classification", ws)
    assert base.run is not None
    fr, ev, seeds = default_fault_registry(), _sklearn_config(False), (1, 2, 3, 4)
    scope = FaultScope(ScopeKind.ALL)
    with SqliteRegistry(ws / "registry.sqlite") as reg:
        store = LocalArtifactStore(ws / "experiments")
        ex = Executor(
            reg, store, source_root=Path.cwd(), adapters=default_registries(), inputs_root=ws
        )
        (prov,) = reg.find(Provenance, run_id=base.run.id)
        assert (
            prov.inputs is not None
            and prov.inputs.model is not None
            and prov.inputs.dataset is not None
        )
        mid, did = prov.inputs.model.record_id, prov.inputs.dataset.record_id

        def launch(spec: Any, name: str) -> Any:
            return run_fault_experiment(
                reg, store, ex, model_id=mid, dataset_id=did, base_spec=spec, fault_registry=fr,
                design=FaultDesign(spec.to_dict(), ev, seeds=seeds, limits=FaultLimits()),
                name=name, source_root=Path.cwd(), baseline_run_id=base.run.id,
            )  # fmt: skip

        a = fr.make("gaussian_noise", seed=1, scope=scope, sigma=1.0)
        b = fr.make("feature_dropout", seed=1, scope=scope, probability=0.4)
        ra, rb, rab = launch(a, "A noise"), launch(b, "B dropout"), launch(fr.compound(a, b), "AB")

        def runs(r: Any) -> tuple[str, ...]:
            return tuple(
                t.treatment_run_id
                for t in reg.find(FaultTrial, fault_experiment_id=r.fault_experiment.id)
                if t.treatment_run_id
            )

        inv = reg.get(
            Investigation,
            reg.get(Experiment, reg.get(Run, base.run.id).experiment_id).investigation_id,
        ).id
        fx = [ra.fault_experiment.id, rb.fault_experiment.id, rab.fault_experiment.id]
        run_discovery(reg, store, ex, inv, [], fx, DiscoveryConfig())
        spec = InteractionSpec(
            control=(base.run.id,),
            a=runs(ra),
            b=runs(rb),
            ab=runs(rab),
            config=InteractionConfig(pairing=Pairing.PAIRED, bootstrap_resamples=300),
        )
        ia = run_interaction(reg, store, ex, inv, spec)
        assert ia.analysis_id
        modes = tuple(sorted(m.id for m in reg.find(FailureMode)))
    return Iris(ws, base.run.id, inv, tuple(fx), ia.analysis_id, modes)


def cli(iris: Iris, capsys: pytest.CaptureFixture[str], *args: str) -> tuple[int, str, str]:
    code = main(["--workspace", str(iris.ws), "slice", *args])
    out = capsys.readouterr()
    return code, out.out, out.err


def write(tmp_path: Path, name: str, body: Any) -> str:
    p = tmp_path / name
    p.write_text(json.dumps(body), encoding="utf-8")
    return str(p)


def sklearn_truth(iris: Iris) -> tuple[list[int], Any, Any]:
    """The evaluated split, the data and the model's predictions, straight from scikit-learn."""
    data = load_iris()
    order = np.random.RandomState(0).permutation(150)
    test = sorted(int(i) for i in order[150 - round(150 * 0.25) :])
    model = joblib.load(iris.ws / "models" / "iris-logreg.joblib")
    return test, data, model.predict(data.data[test])


# -- real-data scenario ------------------------------------------------------------------------------------------------------


def test_real_iris_membership_and_metrics_match_scikit_learn(
    iris: Iris, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    test, data, pred = sklearn_truth(iris)
    wide_ids = [i for i in test if data.data[i, 3] >= 1.0]
    c0 = [i for i in test if data.target[i] == 0]
    f = write(tmp_path, "s.json", {"slices": [WIDE, CLASS0]})
    code, out, _ = cli(iris, capsys, "evaluate", f, "--baseline-run", iris.baseline, "--ids")
    ms = {m["name"]: m for m in json.loads(out)["memberships"]}
    assert (
        code == 0
        and sorted(ms["wide-petals"]["sample_ids"]) == wide_ids
        and sorted(ms["class-0"]["sample_ids"]) == c0
    )
    assert ms["wide-petals"]["n_total"] == len(test) == 38 and ms["wide-petals"]["n_unknown"] == 0

    spec = {"baseline_run": iris.baseline, "slices": [WIDE, CLASS0], "config": {"resamples": 200}, "comparisons": [["wide-petals", "REST"]],
            "fault_experiments": list(iris.faults), "failure_modes": list(iris.modes), "interactions": [iris.interaction]}  # fmt: skip
    code, out, _ = cli(iris, capsys, "analyze", write(tmp_path, "a.json", spec))
    doc = json.loads(out)
    counts = {
        k: v
        for blk in (
            "fault_status_counts",
            "failure_mode_status_counts",
            "interaction_status_counts",
        )
        for k, v in doc["summary"][blk].items()
    }
    assert doc["analysis_status"] == (
        "COMPLETE" if set(counts) <= {"COMPUTED"} else "PARTIAL"
    )  # the status mirrors the evidence
    assert code == (0 if doc["analysis_status"] == "COMPLETE" else 3)
    reg = SqliteRegistry(iris.ws / "registry.sqlite")
    sr = SliceRegistry(reg, LocalArtifactStore(iris.ws / "experiments"))
    res = sr.document(doc["analysis_id"], "results")

    pos = {i: k for k, i in enumerate(test)}
    truth = data.target
    acc_wide = accuracy_score([truth[i] for i in wide_ids], [pred[pos[i]] for i in wide_ids])
    m = next(x for x in res["slices"]["wide-petals"]["metrics"] if x["metric_id"] == "accuracy")
    assert m["value"] == pytest.approx(acc_wide, abs=1e-12) and m["n_samples"] == len(wide_ids)
    rest = [i for i in test if i not in set(wide_ids)]
    comp = res["comparisons"]["wide-petals vs REST"]
    assert comp["descriptive"]["difference"] == pytest.approx(
        acc_wide - accuracy_score([truth[i] for i in rest], [pred[pos[i]] for i in rest]), abs=1e-12
    )
    cp = [c for k, c in res["comparisons"].items() if k != "wide-petals vs REST"]
    assert cp == []  # only the explicitly requested comparison was run

    # faults, failure modes and the interaction, all on real runs and all reported per slice
    for name in ("wide-petals", "class-0"):
        for fr in res["faults"][name]:
            assert fr["status"] == "COMPUTED" and fr["points"][0]["n_trials"] == 4
        assert {r["status"] for r in res["failure_modes"][name]["modes"]} <= {
            "COMPUTED",
            "INSUFFICIENT_EVIDENCE",
        }
        assert all(r["confirmed"] is False for r in res["failure_modes"][name]["modes"])
        inter = res["interactions"][name][0]
        assert inter["status"] in ("COMPUTED", "INSUFFICIENT_EVIDENCE")
        assert "derived" not in inter or inter["pairing"] == "PAIRED"
    wide_inter = res["interactions"]["wide-petals"][0]
    assert wide_inter["status"] == "COMPUTED" and wide_inter["n_members"] == len(wide_ids)
    reg.close()


# -- CLI ---------------------------------------------------------------------------------------------------------------------------


def test_validate_register_list_inspect(
    iris: Iris, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    f = write(tmp_path, "s.json", {"slices": [WIDE, CLASS0]})
    code, out, _ = cli(iris, capsys, "validate", f)
    doc = json.loads(out)
    assert (
        code == 0
        and doc["valid"]
        and {s["name"] for s in doc["slices"]} == {"wide-petals", "class-0"}
    )
    assert all(s["slice_id"].startswith("sls_") and s["static"] for s in doc["slices"])
    assert "VALID" in cli(iris, capsys, "validate", f, "--format", "text")[1]
    full = write(tmp_path, "full.json", {"baseline_run": iris.baseline, "slices": [WIDE]})
    assert json.loads(cli(iris, capsys, "validate", full)[1])["kind"] == "analysis"

    renamed = write(tmp_path, "r.json", {"slices": [{**WIDE, "name": "another name"}]})
    code, out, _ = cli(iris, capsys, "register", renamed)
    (r1,) = json.loads(out)["slices"]
    code, out, _ = cli(iris, capsys, "register", f)
    rows = {r["name"]: r for r in json.loads(out)["slices"]}
    assert (
        code == 0
        and rows["wide-petals"]["id"] == r1["id"]
        and rows["wide-petals"]["created"] is False
    )  # the same logical slice

    _, out, _ = cli(iris, capsys, "list", "--field", PW)
    assert {r["id"] for r in json.loads(out)["slices"]} == {r1["id"]}
    assert json.loads(cli(iris, capsys, "list", "--static", "no")[1])["slices"] == []
    _, out, _ = cli(iris, capsys, "inspect", r1["id"])
    ins = json.loads(out)
    assert ins["condition"]["op"] == "range" and ins["static"] is True and ins["fields"] == [PW]


def test_analyze_partial_exit_code_inspect_and_compare(
    iris: Iris, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    tiny = {"name": "tiny", "condition": {"op": "range", "field": PW, "low": 2.4, "high": None}}
    cfg = {"resamples": 100, "min_members": 8}
    code, out, _ = cli(
        iris,
        capsys,
        "analyze",
        write(
            tmp_path,
            "p.json",
            {"baseline_run": iris.baseline, "slices": [WIDE, tiny], "config": cfg},
        ),
    )
    doc = json.loads(out)
    assert (
        code == 3 and doc["analysis_status"] == "PARTIAL"
    )  # a slice with too few members is visible in the exit code
    assert (
        doc["summary"]["slices"]["tiny"]["evidence"] == "INSUFFICIENT_EVIDENCE"
        and doc["summary"]["slices"]["wide-petals"]["evidence"] == "SUFFICIENT"
    )
    aid = doc["analysis_id"]
    _, out, _ = cli(iris, capsys, "list", "--analyses", "--baseline-run", iris.baseline)
    assert aid in {a["id"] for a in json.loads(out)["analyses"]}
    _, out, _ = cli(iris, capsys, "inspect", aid, "--full")
    ins = json.loads(out)
    assert ins["provenance"]["baseline_run_id"] == iris.baseline and set(ins["documents"]) == {
        "spec",
        "membership",
        "results",
        "summary",
    }
    assert (
        ins["provenance"]["dataset_fingerprint"].startswith("sha256:")
        and ins["provenance"]["split"] == "test"
    )
    assert {a["path"] for a in ins["artifacts"]} == {
        "slice/spec.json",
        "slice/membership.json",
        "slice/results.json",
        "slice/summary.json",
    }
    sid = ins["provenance"]["slice_ids"]["wide-petals"]
    assert aid in json.loads(cli(iris, capsys, "inspect", sid)[1])["used_by"]
    assert cli(
        iris,
        capsys,
        "analyze",
        write(
            tmp_path,
            "p2.json",
            {"baseline_run": iris.baseline, "slices": [WIDE, tiny], "config": cfg},
        ),
        "--format",
        "text",
    )[1].startswith("COMPLETED:")

    a_file, b_file = write(tmp_path, "a1.json", WIDE), write(tmp_path, "b1.json", CLASS0)
    code, out, _ = cli(
        iris,
        capsys,
        "compare",
        a_file,
        "POPULATION",
        "--baseline-run",
        iris.baseline,
        "--resamples",
        "100",
    )
    cmp = json.loads(out)
    assert code == 0 and cmp["a"]["slice_id"] == sid and cmp["comparison"]["n_population"] == 38
    code, out, _ = cli(
        iris,
        capsys,
        "compare",
        a_file,
        "REST",
        "--baseline-run",
        iris.baseline,
        "--resamples",
        "100",
        "--method",
        "bca",
    )
    assert code == 0 and json.loads(out)["comparison"]["inference"]["interval"]["method"] == "bca"
    code, out, _ = cli(
        iris,
        capsys,
        "compare",
        a_file,
        b_file,
        "--baseline-run",
        iris.baseline,
        "--resamples",
        "100",
    )
    both = json.loads(out)["comparison"]
    assert (
        code == 0
        and both["measure"] == "accuracy"
        and both["descriptive"]["difference"] is not None
    )


def test_errors_are_reported_not_raised(
    iris: Iris, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    base = iris.baseline
    bad_op = write(
        tmp_path,
        "bad.json",
        {"name": "x", "condition": {"op": "python", "field": "a", "values": ["1+1"]}},
    )
    empty = write(
        tmp_path,
        "e.json",
        {"name": "none", "condition": {"op": "eq", "field": "target", "values": [42]}},
    )
    for args in (
        ("validate", bad_op),
        ("evaluate", bad_op, "--baseline-run", base),
        (
            "analyze",
            write(tmp_path, "n.json", {"baseline_run": "run_" + "0" * 32, "slices": [WIDE]}),
        ),
        ("analyze", write(tmp_path, "m.json", {"baseline_run": base, "slices": []})),
        ("inspect", "san_" + "0" * 32),
        ("compare", empty, "REST", "--baseline-run", base),
        (
            "evaluate",
            write(
                tmp_path,
                "g.json",
                {
                    "name": "ghost",
                    "condition": {"op": "eq", "field": "feature:nope", "values": [1]},
                },
            ),
            "--baseline-run",
            base,
        ),
    ):
        code, _, err = cli(iris, capsys, *args)
        assert code == 2 and err.startswith("error:"), (args, err)
    code, out, _ = cli(iris, capsys, "evaluate", empty, "--baseline-run", base)
    m = json.loads(out)["memberships"][0]
    assert (
        code == 0 and m["status"] == "EMPTY" and m["n_members"] == 0
    )  # a valid slice with no members is a result
    with SqliteRegistry(iris.ws / "registry.sqlite") as reg:
        n = len(reg.find(SliceAnalysis))
    z = write(
        tmp_path,
        "z.json",
        {"baseline_run": base, "slices": [WIDE], "fault_experiments": ["fxp_" + "0" * 32]},
    )
    assert cli(iris, capsys, "analyze", z)[0] == 2
    with SqliteRegistry(iris.ws / "registry.sqlite") as reg:
        assert len(reg.find(SliceAnalysis)) == n  # a refused request leaves no record
