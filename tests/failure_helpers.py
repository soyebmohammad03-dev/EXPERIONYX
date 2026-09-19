"""Helpers for failure-discovery tests: hand-built signals and a real registry world."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval_helpers import EvalWorld, class_data, eval_world
from experionyx.evaluation.config import EvaluationConfig
from experionyx.failures.config import EXTRACTOR_VERSION, DiscoveryConfig
from experionyx.failures.entities import FailureSignal
from experionyx.failures.taxonomy import Direction, FailureCategory, SignalKind

NOW = datetime(2026, 9, 19, 12, tzinfo=UTC)
CFG = DiscoveryConfig()
FP_A = "sha256:" + "a" * 64
FP_B = "sha256:" + "b" * 64


def rid(prefix: str, n: int) -> str:
    return f"{prefix}_{n:032x}"


def sig(
    n: int = 1, *, run: str | None = None, exp: str | None = None, kind: SignalKind = SignalKind.CLASS_RECALL_DEGRADATION,
    cls: str | None = "1", pred: str | None = None, slice_name: str | None = None, mag: float = 0.2,
    fault: str | None = "gaussian_noise", seed: int = 0, ids: tuple[str, ...] = ("1", "2"), sample_count: int | None = None,
    params: dict[str, object] | None = None, frac: float | None = 0.5, model: str = FP_A, dataset: str = FP_A,
    category: FailureCategory = FailureCategory.CLASS_SPECIFIC_ERROR, direction: Direction = Direction.WORSE,
    cfg_hash: str | None = None, extra: dict[str, object] | None = None,
) -> FailureSignal:  # fmt: skip
    detail: dict[str, object] = {
        "source": "fault" if fault else "evaluation",
        "model_fingerprint": model,
        "dataset_fingerprint": dataset,
        "n_samples": 20,
        "sample_ids_recorded": True,
        **(extra or {}),
    }
    if cls is not None:
        detail["class_label"] = cls
    if pred is not None:
        detail["predicted_label"] = pred
    if slice_name is not None:
        detail["slice"] = slice_name
    if fault:
        detail["fault"] = {
            "type": fault,
            "family_id": rid("flt", n),
            "fault_id": rid("fap", n),
            "seed": seed,
            "target": "INPUT_OR_MIXED",
            "parameters": params if params is not None else {"sigma": 1.0},
            "affected_fraction": frac,
            "fault_experiment_id": rid("fxp", 1),
            "baseline_run_id": rid("run", 999),
        }
    return FailureSignal(
        run_id=run or rid("run", n), experiment_id=exp or rid("exp", n), signal_kind=kind, category=category,
        signature=f"{kind.value}|class={cls}|pred={pred}|slice={slice_name}|fault={fault}|dir={direction.value}|n={n}",
        direction=direction, magnitude=mag, detail=detail, sample_ids=ids, sample_count=len(ids) if sample_count is None else sample_count,
        extractor_version=EXTRACTOR_VERSION, config_hash=cfg_hash or CFG.config_hash, created_at=NOW,
    )  # fmt: skip


def noisy_world(tmp_path: Path) -> EvalWorld:
    """A real, imperfect classifier (some noisy targets) evaluated on a real split."""
    return eval_world(
        tmp_path,
        model={"threshold": 5.0, "proba": True},
        data=class_data(40, noise_every=5),
        config=EvaluationConfig(split="test"),
    )


# -- real fault experiments ------------------------------------------------------------------------

from experionyx.adapters.records import RegisteredDataset, RegisteredModel  # noqa: E402
from experionyx.domain import Experiment, Investigation, Run  # noqa: E402
from experionyx.faults.design import FaultDesign, SweepSpec  # noqa: E402
from experionyx.faults.lab import FaultExperimentResult, run_fault_experiment  # noqa: E402
from experionyx.faults.library import default_fault_registry  # noqa: E402

FR = default_fault_registry()
EVAL = EvaluationConfig(split="test")


def launch(
    w: EvalWorld,
    fault: str,
    params: dict[str, float],
    sweep: tuple[str, tuple[float, ...]] | None,
    seeds: tuple[int, ...],
    **kw: Any,
) -> FaultExperimentResult:
    reg = w.registry
    spec = FR.make(fault, seed=seeds[0], **params)
    design = FaultDesign(
        spec.to_dict(),
        EVAL,
        seeds=seeds,
        sweep=SweepSpec(*sweep) if sweep else None,
        limits=kw.pop("limits", FaultDesign(spec.to_dict(), EVAL).limits),
    )
    result: FaultExperimentResult = run_fault_experiment(
        reg, w.store, w.executor, model_id=reg.find(RegisteredModel)[0].id, dataset_id=reg.find(RegisteredDataset)[0].id,
        base_spec=spec, fault_registry=FR, design=design, name=kw.pop("name", f"{fault} sweep"), source_root=w.workspace.parent, **kw,
    )  # fmt: skip
    return result


def noise_sweep(
    w: EvalWorld,
    seeds: tuple[int, ...] = (1, 2, 3),
    values: tuple[float, ...] = (2.0, 4.0, 8.0),
    **kw: Any,
) -> FaultExperimentResult:
    return launch(w, "gaussian_noise", {"sigma": 1.0}, ("sigma", values), seeds, **kw)


def home_of(w: EvalWorld, res: FaultExperimentResult) -> str:
    run = w.registry.get(Run, res.baseline_run_id)
    return w.registry.get(
        Investigation, w.registry.get(Experiment, run.experiment_id).investigation_id
    ).id
