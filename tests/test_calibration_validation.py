"""REAL-DATA engineering validation of the calibration laboratory: (1) the bundled iris dataset with a
scikit-learn logistic regression (real predict_proba); (2) the existing blobs tensor dataset with a
PyTorch linear classifier (logits, so the softmax is an explicit, recorded assumption). Every expected
value is computed directly with NumPy, scikit-learn (`calibration_curve`, `log_loss`,
`brier_score_loss`, `LogisticRegression`, `IsotonicRegression`) or PyTorch (`softmax`), never through
`experionyx.calibration`. These check that the machinery reproduces independent numbers on real
models; they are NOT scientific findings about iris, blobs or about any model's calibration."""

import dataclasses
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("sklearn")

import joblib
from sklearn.calibration import calibration_curve
from sklearn.datasets import load_iris
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

import experionyx.calibration.engine as ce
from calibration_helpers import View
from experionyx.adapters.capabilities import DeviceKind
from experionyx.adapters.registry import default_registries
from experionyx.artifacts import LocalArtifactStore
from experionyx.calibration.entities import CalibrationAnalysis
from experionyx.calibration.registry import CalibrationRegistry
from experionyx.calibration.spec import CalibrationSpec
from experionyx.demos import run_demo
from experionyx.domain import ConfigurationRef, Experiment, ExperimentStatus, Run
from experionyx.evaluation.config import EvaluationConfig
from experionyx.evaluation.engine import PROCEDURE
from experionyx.execution import Executor, resolve_procedure
from experionyx.interactions.samples import _rows
from experionyx.sqlite import SqliteRegistry

FAST = {"resamples": 40, "permutations": 40, "min_samples": 10}


class Lab:
    """A demo workspace with a registry, artifact store and executor (the CLI's own wiring)."""

    def __init__(self, ws: Path, baseline: str) -> None:
        self.ws, self.baseline = ws, baseline
        self.reg = SqliteRegistry(ws / "registry.sqlite")
        self.store = LocalArtifactStore(ws / "experiments")
        self.executor = Executor(self.reg, self.store, source_root=Path.cwd(), adapters=default_registries(), inputs_root=ws, device=DeviceKind.CPU)  # fmt: skip

    def eval_split(self, split: str, seed: int = 0) -> str:
        """Evaluate another split with the same model, dataset and evaluation settings: a NEW run."""
        run = self.reg.get(Run, self.baseline)
        exp0 = self.reg.get(Experiment, run.experiment_id)
        cfg0 = self.reg.get(ConfigurationRef, exp0.configuration_id)
        conf = dataclasses.replace(EvaluationConfig.from_parameters(cfg0.parameters), split=split)
        cfg = ConfigurationRef(conf.to_parameters())
        if not self.reg.exists(ConfigurationRef, cfg.id):
            self.reg.add(cfg)
        exp = Experiment(exp0.investigation_id, f"eval-{split}", "calibration data", exp0.model, exp0.dataset, cfg.id, datetime.now(UTC))  # fmt: skip
        if not self.reg.exists(Experiment, exp.id):
            self.reg.add(exp)
            self.reg.update_status(exp.with_status(ExperimentStatus.READY))
        res = self.executor.execute(exp.id, resolve_procedure(PROCEDURE), seed=seed, procedure_name=PROCEDURE)  # fmt: skip
        assert res.status.value == "COMPLETED"
        return res.run.id

    def analyze(self, source: str, **over: Any) -> View:
        spec = CalibrationSpec.from_dict({"baseline_run": self.baseline, "prediction_source": source, "statistics": FAST, **over})  # fmt: skip
        inv = self.reg.get(
            Experiment, self.reg.get(Run, self.baseline).experiment_id
        ).investigation_id
        out = ce.run_calibration_request(self.reg, self.store, self.executor, inv, spec)
        assert out.analysis_id, out
        return View(self.reg.get(CalibrationAnalysis, out.analysis_id))

    def rows(self, run_id: str) -> list[dict[str, Any]]:
        return list(_rows(self.reg, self.store, run_id))


def top_arrays(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p = np.array([r["scores"] for r in rows])
    y = np.array([r["true"] for r in rows])
    pred = np.array([r["predicted"] for r in rows])
    conf = p[np.arange(len(rows)), pred]
    return p, y, conf, (pred == y).astype(float)


def logit(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 1e-15, 1 - 1e-15)
    out: np.ndarray = np.log(c / (1 - c))
    return out


def numpy_ece(conf: np.ndarray, ok: np.ndarray, bins: int) -> tuple[float, float]:
    idx = np.digitize(conf, [k / bins for k in range(1, bins)])
    ece = mce = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            gap = abs(ok[mask].mean() - conf[mask].mean())
            ece, mce = ece + mask.mean() * gap, max(mce, gap)
    return float(ece), float(mce)


# -- (1) iris + scikit-learn -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def iris(tmp_path_factory: pytest.TempPathFactory) -> Lab:
    ws = tmp_path_factory.mktemp("iris-cal") / "w"
    return Lab(ws, run_demo("sklearn-classification", ws).run.id)


def test_iris_stored_probabilities_are_the_models_own_predict_proba(iris: Lab) -> None:
    order = np.random.RandomState(0).permutation(150)
    test = sorted(int(i) for i in order[150 - round(150 * 0.25) :])
    est = joblib.load(iris.ws / "models" / "iris-logreg.joblib")
    want = est.predict_proba(load_iris().data[test])
    rows = sorted(iris.rows(iris.baseline), key=lambda r: r["index"])
    assert [r["index"] for r in rows] == test
    assert np.array([r["scores"] for r in rows]) == pytest.approx(want, abs=1e-12)


def test_iris_metrics_and_reliability_table_match_scikit_learn(iris: Lab) -> None:
    a = iris.analyze("PREDICT_PROBA", objects=["CLASSWISE", "TOP_LABEL"], binning={"n_bins": 5})
    cr = CalibrationRegistry(iris.reg, iris.store)
    rows = iris.rows(iris.baseline)
    p, y, conf, ok = top_arrays(rows)
    s = a.summary["baseline"]
    ece, mce = numpy_ece(conf, ok, 5)
    assert a.summary["n_usable"] == len(rows) and s["status"] == "COMPUTED"
    assert s["metrics"]["ece"] == pytest.approx(ece, abs=1e-12) and s["metrics"]["mce"] == pytest.approx(mce, abs=1e-12)  # fmt: skip
    assert s["metrics"]["accuracy"] == pytest.approx(ok.mean())
    assert s["metrics"]["brier_top_label"] == pytest.approx(
        brier_score_loss(ok, conf), abs=1e-12
    )  # sklearn
    assert s["metrics"]["log_loss_top_label"] == pytest.approx(log_loss(ok, conf, labels=[0, 1]), rel=1e-6)  # fmt: skip
    assert s["probability_vector"]["nll_multiclass"] == pytest.approx(log_loss(y, p, labels=[0, 1, 2]), rel=1e-6)  # fmt: skip
    onehot = np.eye(3)[y]
    assert s["probability_vector"]["brier_multiclass"] == pytest.approx(((p - onehot) ** 2).sum(axis=1).mean())  # fmt: skip
    # scikit-learn's reliability curve for uniform bins (non-empty bins only)
    prob_true, prob_pred = calibration_curve(ok, conf, n_bins=5, strategy="uniform")
    bins = [b for b in cr.document(a.id, "bins")["baseline"]["top_label"] if b["count"]]
    assert [b["accuracy"] for b in bins] == pytest.approx(prob_true) and [b["mean_confidence"] for b in bins] == pytest.approx(prob_pred)  # fmt: skip
    assert (
        len(cr.document(a.id, "bins")["baseline"]["top_label"]) == 5
    )  # empty bins are still listed
    # classwise: scikit-learn's per-class reliability curve of p_k against 1[y == k]
    cw = cr.document(a.id, "metrics")["contexts"]["baseline"]["metrics"]["classwise"]["per_class"]
    for k in range(3):
        t, pp = calibration_curve((y == k).astype(int), p[:, k], n_bins=5, strategy="uniform")
        kb = [b for b in cr.document(a.id, "bins")["baseline"]["classwise"][str(k)] if b["count"]]
        assert [b["accuracy"] for b in kb] == pytest.approx(t) and [b["mean_confidence"] for b in kb] == pytest.approx(pp)  # fmt: skip
        assert cw[str(k)]["positives"] == int((y == k).sum())
    assert a.summary["baseline"]["metrics"]["ece"] != pytest.approx(cr.document(a.id, "metrics")["contexts"]["baseline"]["metrics"]["classwise"]["mean_ece"])  # different objects  # fmt: skip


def test_iris_platt_and_isotonic_fitted_on_a_separate_split_match_scikit_learn(iris: Lab) -> None:
    calib = iris.eval_split(
        "train"
    )  # a different split of the same dataset: not the evaluation samples
    model_digest = (iris.ws / "models" / "iris-logreg.joblib").read_bytes()
    _, _, c_conf, c_ok = top_arrays(iris.rows(calib))
    _, _, t_conf, t_ok = top_arrays(iris.rows(iris.baseline))
    for method in ("PLATT", "ISOTONIC"):
        a = iris.analyze("PREDICT_PROBA", method=method, fit={"mode": "RUN", "calibration_run": calib}, binning={"n_bins": 5})  # fmt: skip
        cr = CalibrationRegistry(iris.reg, iris.store)
        m = cr.document(a.id, "metrics")["calibration_method"]
        assert m["protocol"]["calibration_data"]["source"] == f"run {calib}" and m["protocol"]["overlap"] == 0  # fmt: skip
        if method == "PLATT":
            lr = LogisticRegression(C=1e6, tol=1e-12, max_iter=100_000).fit(logit(c_conf).reshape(-1, 1), c_ok)  # fmt: skip
            assert m["parameters"]["a"] == pytest.approx(lr.coef_[0][0], rel=1e-3, abs=1e-3)
            assert m["parameters"]["b"] == pytest.approx(lr.intercept_[0], rel=1e-3, abs=1e-3)
            cal = lr.predict_proba(logit(t_conf).reshape(-1, 1))[:, 1]
        else:
            iso = IsotonicRegression(out_of_bounds="clip").fit(c_conf, c_ok)
            cal = iso.predict(t_conf)
        want_ece, _ = numpy_ece(cal, t_ok, 5)
        assert m["after"]["metrics"]["top_label"]["ece"] == pytest.approx(want_ece, abs=2e-3 if method == "PLATT" else 1e-9)  # fmt: skip
        assert m["before"]["metrics"]["top_label"]["ece"] == pytest.approx(numpy_ece(t_conf, t_ok, 5)[0])  # fmt: skip
        assert m["after"]["n_samples"] == len(t_conf)
    assert (
        iris.ws / "models" / "iris-logreg.joblib"
    ).read_bytes() == model_digest  # the model was not touched


def test_iris_calibrating_on_the_evaluated_split_is_refused(iris: Lab) -> None:
    twin = iris.eval_split("test", seed=1)  # the same dataset fingerprint and split
    spec = CalibrationSpec.from_dict({"baseline_run": iris.baseline, "prediction_source": "PREDICT_PROBA", "method": "ISOTONIC", "fit": {"mode": "RUN", "calibration_run": twin}, "statistics": FAST})  # fmt: skip
    with pytest.raises(ce.CalibrationError, match="LEAKAGE"):
        ce.validate(iris.reg, iris.store, spec)


def test_iris_analysis_replays_deterministically_and_the_same_data_gives_the_same_fingerprint(iris: Lab) -> None:  # fmt: skip
    a = iris.analyze("PREDICT_PROBA", method="PLATT", fit={"seed": 2}, objects=["CLASSWISE", "TOP_LABEL"])  # fmt: skip
    out = ce.replay_check(iris.reg, iris.store, iris.executor, a.id)
    assert out["deterministic"] is True and out["differences"] == []
    spec = CalibrationSpec.from_dict(a.spec)
    assert ce.compute(iris.reg, iris.store, spec, None).fingerprint == a.provenance_fingerprint


# -- (2) blobs + PyTorch (logits -> softmax is a recorded assumption) ------------------------------------------


@pytest.fixture(scope="module")
def blobs(tmp_path_factory: pytest.TempPathFactory) -> Lab:
    pytest.importorskip("torch")
    ws = tmp_path_factory.mktemp("blobs-cal") / "w"
    return Lab(ws, run_demo("torch-classification", ws).run.id)


def test_torch_probabilities_are_softmax_of_logits_and_declared_as_such(blobs: Lab) -> None:
    import torch

    from experionyx.adapters.torch_adapter import TorchModelAdapter

    payload = torch.load(next((blobs.ws / "datasets").glob("*.pt")))
    model = TorchModelAdapter.load(next((blobs.ws / "models").glob("*.pt")), version="1", device=DeviceKind.CPU, options={"task": "CLASSIFICATION", "input_shape": [4]})  # fmt: skip
    x = payload["X"][150:200]
    logits = torch.tensor([list(o) for o in model.predict(x).outputs])  # type: ignore[call-overload]
    want = torch.softmax(logits, dim=1).numpy()
    rows = sorted(blobs.rows(blobs.baseline), key=lambda r: r["index"])
    assert np.array([r["scores"] for r in rows]) == pytest.approx(want, abs=1e-6)
    # scores that are not probabilities from the model are never accepted under the wrong name
    with pytest.raises(ce.CalibrationError, match="never treated as another representation"):
        ce.validate(blobs.reg, blobs.store, CalibrationSpec.from_dict({"baseline_run": blobs.baseline, "prediction_source": "PREDICT_PROBA", "statistics": FAST}))  # fmt: skip
    a = blobs.analyze("SOFTMAX_LOGITS", binning={"n_bins": 5}, method="PLATT", fit={"seed": 1})
    doc = CalibrationRegistry(blobs.reg, blobs.store).document(a.id, "spec")
    assert (
        doc["prediction_representation"] == "SOFTMAX_LOGITS"
        and "ASSUMING" in doc["score_assumption"]
    )
    _, _, conf, ok = top_arrays(rows)
    ece, mce = numpy_ece(conf, ok, 5)
    s = a.summary["baseline"]["metrics"]
    assert s["ece"] == pytest.approx(ece, abs=1e-9) and s["mce"] == pytest.approx(mce, abs=1e-9)
    assert s["brier_top_label"] == pytest.approx(brier_score_loss(ok, conf), abs=1e-9)
    assert a.summary["prediction_representation"] == "SOFTMAX_LOGITS"


def test_torch_platt_fitted_on_the_train_split_matches_scikit_learn(blobs: Lab) -> None:
    calib = blobs.eval_split("train")
    a = blobs.analyze(
        "SOFTMAX_LOGITS", method="PLATT", fit={"mode": "RUN", "calibration_run": calib}
    )
    m = CalibrationRegistry(blobs.reg, blobs.store).document(a.id, "metrics")["calibration_method"]
    _, _, c_conf, c_ok = top_arrays(blobs.rows(calib))
    z = np.log(np.clip(c_conf, 1e-15, 1 - 1e-15) / (1 - np.clip(c_conf, 1e-15, 1 - 1e-15)))
    lr = LogisticRegression(C=1e6, tol=1e-12, max_iter=100_000).fit(z.reshape(-1, 1), c_ok)
    assert m["parameters"]["a"] == pytest.approx(lr.coef_[0][0], rel=1e-3, abs=1e-3)
    assert m["parameters"]["b"] == pytest.approx(lr.intercept_[0], rel=1e-3, abs=1e-3)
    assert m["protocol"]["calibration_data"]["split"] == "train" and m["protocol"]["evaluation_data"]["split"] == "test"  # fmt: skip
