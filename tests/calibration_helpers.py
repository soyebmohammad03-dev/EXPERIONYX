"""Calibration validation fixtures (ENGINEERING VALIDATION DATA, not findings about any model).

`cal_world` is the controlled binary linear world of `stress_helpers` (120 samples, sigmoid
probabilities, every 9th label flipped) with two named splits so a separate calibration run exists:
"test" = samples 0..59 (the baseline) and "calib" = samples 60..119. `extra_run` evaluates another
split with the same model in the same registry. Expected numbers in the tests are recomputed from the
stored rows with formulas written independently of `experionyx.calibration`."""

from pathlib import Path
from typing import Any

from eval_helpers import NOW, EvalWorld
from experionyx.adapters.records import RegisteredDataset, RegisteredModel
from experionyx.calibration import engine as ce
from experionyx.calibration.entities import CalibrationAnalysis
from experionyx.calibration.registry import CalibrationRegistry
from experionyx.calibration.spec import CalibrationSpec
from experionyx.domain import (
    ConfigurationRef,
    Experiment,
    ExperimentStatus,
    Investigation,
    to_jsonable,
)
from experionyx.evaluation.config import EvaluationConfig
from experionyx.evaluation.engine import PROCEDURE
from experionyx.execution import resolve_procedure
from experionyx.interactions.samples import _rows
from stress_helpers import stress_world

SPLITS = {"test": list(range(60)), "calib": list(range(60, 120))}
FAST = {"resamples": 60, "permutations": 60, "min_samples": 20}


def cal_world(tmp_path: Path, **over: Any) -> tuple[EvalWorld, str]:
    """(world, baseline run id): the linear model evaluated on the 'test' split."""
    w, _mid, _did = stress_world(tmp_path, splits=over.pop("splits", SPLITS), **over)
    res = w.run()
    assert res.status.value == "COMPLETED"
    return w, res.run.id


def extra_run(w: EvalWorld, split: str, *, seed: int = 0) -> str:
    """Evaluate `split` with the registered model and dataset as a NEW experiment; the run ID."""
    (m,) = [m for m in w.registry.find(RegisteredModel) if m.name == "model"]
    (d,) = [d for d in w.registry.find(RegisteredDataset) if d.name == "data"]
    inv = w.registry.find(Investigation)[0]
    cfg = ConfigurationRef(EvaluationConfig(split=split, batch_size=32).to_parameters())
    if not w.registry.exists(ConfigurationRef, cfg.id):
        w.registry.add(cfg)
    exp = Experiment(inv.id, f"eval-{split}", "baseline", m.ref(), d.ref(), cfg.id, NOW)
    if not w.registry.exists(Experiment, exp.id):
        w.registry.add(exp)
        w.registry.update_status(exp.with_status(ExperimentStatus.READY))
    res = w.executor.execute(
        exp.id, resolve_procedure(PROCEDURE), seed=seed, procedure_name=PROCEDURE
    )
    assert res.status.value == "COMPLETED"
    return res.run.id


def spec_dict(baseline: str, **over: Any) -> dict[str, Any]:
    d: dict[str, Any] = {
        "baseline_run": baseline,
        "prediction_source": "PREDICT_PROBA",
        "statistics": dict(FAST),
    }
    stats = over.pop("statistics", {})
    d.update(over)
    d["statistics"] = {**d["statistics"], **stats}
    return d


def make_spec(baseline: str, **over: Any) -> CalibrationSpec:
    return CalibrationSpec.from_dict(spec_dict(baseline, **over))


class View:
    """A calibration analysis with its frozen mappings as plain dicts, so tests can index them."""

    def __init__(self, entity: CalibrationAnalysis) -> None:
        self.entity = entity
        self.summary: dict[str, Any] = to_jsonable(entity.summary)  # type: ignore[assignment]
        self.spec: dict[str, Any] = to_jsonable(entity.spec)  # type: ignore[assignment]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.entity, name)


def analyze(w: EvalWorld, baseline: str, **over: Any) -> View:
    spec = make_spec(baseline, **over)
    inv = w.registry.find(Investigation)[0]
    out = ce.run_calibration_request(w.registry, w.store, w.executor, inv.id, spec)
    assert out.analysis_id, out
    return View(w.registry.get(CalibrationAnalysis, out.analysis_id))


def registry(w: EvalWorld) -> CalibrationRegistry:
    return CalibrationRegistry(w.registry, w.store)


def stored_rows(w: EvalWorld, run_id: str) -> list[dict[str, Any]]:
    """The run's stored per-sample rows, straight from the artifact."""
    return list(_rows(w.registry, w.store, run_id))
