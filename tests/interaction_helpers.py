"""Helpers: a REAL four-cell (plus reverse-order) design built with the fault laboratory."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eval_helpers import EvalWorld, class_data, eval_world
from experionyx.adapters.records import RegisteredDataset, RegisteredModel
from experionyx.domain import Experiment, Investigation, Run
from experionyx.evaluation.config import EvaluationConfig
from experionyx.faults.design import FaultDesign
from experionyx.faults.entities import FaultTrial, TrialStatus
from experionyx.faults.lab import FaultExperimentResult, run_fault_experiment
from experionyx.faults.library import default_fault_registry
from experionyx.faults.spec import FaultSpec
from experionyx.interactions.config import InteractionConfig, InteractionSpec

FR = default_fault_registry()
EVAL = EvaluationConfig(split="test")


def world(tmp_path: Path, n: int = 80, noise_every: int = 7) -> EvalWorld:
    return eval_world(
        tmp_path,
        model={"threshold": 5.0, "proba": True},
        data=class_data(n, noise_every=noise_every),
        config=EVAL,
    )


def noise(sigma: float = 3.0, seed: int = 0) -> FaultSpec:
    return FR.make("gaussian_noise", seed=seed, sigma=sigma)


def dropout(p: float = 0.3, seed: int = 0) -> FaultSpec:
    return FR.make("feature_dropout", seed=seed, probability=p)


def launch(
    w: EvalWorld, spec: FaultSpec, seeds: tuple[int, ...], **kw: Any
) -> FaultExperimentResult:
    reg = w.registry
    return run_fault_experiment(
        reg, w.store, w.executor, model_id=next(m.id for m in reg.find(RegisteredModel) if m.name == "model"), dataset_id=next(d.id for d in reg.find(RegisteredDataset) if d.name == "data"),
        base_spec=spec, fault_registry=FR, design=FaultDesign(spec.to_dict(), EVAL, seeds=seeds), name=kw.pop("name", spec.type),
        source_root=w.workspace.parent, **kw,
    )  # fmt: skip


def runs_of(w: EvalWorld, res: FaultExperimentResult) -> tuple[str, ...]:
    return tuple(
        t.treatment_run_id
        for t in w.registry.find(FaultTrial, fault_experiment_id=res.fault_experiment.id)
        if t.status is TrialStatus.COMPLETED and t.treatment_run_id
    )


@dataclass
class Design:
    w: EvalWorld
    baseline: str
    a: FaultExperimentResult
    b: FaultExperimentResult
    ab: FaultExperimentResult
    ba: FaultExperimentResult | None

    def spec(self, config: InteractionConfig | None = None, **kw: Any) -> InteractionSpec:
        cfg = config or InteractionConfig(bootstrap_resamples=300)
        ba = runs_of(self.w, self.ba) if self.ba is not None and cfg.order_analysis else ()
        return InteractionSpec(
            control=kw.pop("control", (self.baseline,)),
            a=kw.pop("a", runs_of(self.w, self.a)),
            b=kw.pop("b", runs_of(self.w, self.b)),
            ab=kw.pop("ab", runs_of(self.w, self.ab)),
            ba=kw.pop("ba", ba),
            config=cfg,
            **kw,
        )

    @property
    def investigation(self) -> str:
        reg = self.w.registry
        return reg.get(
            Investigation,
            reg.get(Experiment, reg.get(Run, self.baseline).experiment_id).investigation_id,
        ).id


def four_cell(
    w: EvalWorld,
    seeds: tuple[int, ...] = (1, 2, 3, 4),
    sigma: float = 3.0,
    p: float = 0.3,
    order: bool = True,
) -> Design:
    a = launch(w, noise(sigma), seeds, name="A noise")
    base = a.baseline_run_id
    b = launch(w, dropout(p), seeds, name="B dropout", baseline_run_id=base)
    ab = launch(w, FR.compound(noise(sigma), dropout(p)), seeds, name="AB", baseline_run_id=base)
    ba = (
        launch(w, FR.compound(dropout(p), noise(sigma)), seeds, name="BA", baseline_run_id=base)
        if order
        else None
    )
    return Design(w, base, a, b, ab, ba)
