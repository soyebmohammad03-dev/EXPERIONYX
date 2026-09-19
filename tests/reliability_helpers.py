"""A REAL evidence set for profile tests: baseline, fault experiments, a Phase 6 discovery and a
Phase 7 interaction analysis, all produced by the actual engines."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eval_helpers import NOW, EvalWorld, class_data
from experionyx.adapters.capabilities import DeviceKind
from experionyx.adapters.records import RegisteredDataset, RegisteredModel
from experionyx.failures.engine import DiscoveryRunResult, run_discovery
from experionyx.failures.entities import FailureMode
from experionyx.failures.taxonomy import FailureStatus
from experionyx.faults.design import FaultDesign
from experionyx.faults.lab import FaultExperimentResult, run_fault_experiment
from experionyx.interactions.config import InteractionConfig
from experionyx.interactions.engine import run_interaction
from experionyx.reliability.spec import ProfileSpec
from experionyx.reliability.taxonomy import Scope
from interaction_helpers import EVAL, FR, Design, four_cell, noise, world
from pure_adapters import ListDatasetAdapter, ThresholdClassifierAdapter


@dataclass
class EvidenceSet:
    design: Design
    discovery: DiscoveryRunResult
    interaction_id: str
    mode_ids: tuple[str, ...]

    @property
    def w(self) -> EvalWorld:
        return self.design.w

    @property
    def investigation(self) -> str:
        return self.design.investigation

    @property
    def fault_experiments(self) -> tuple[str, ...]:
        return (
            self.design.a.fault_experiment.id,
            self.design.b.fault_experiment.id,
            self.design.ab.fault_experiment.id,
        )

    def spec(self, scope: Scope = Scope.MODEL_DATASET_EVALUATION, **over: Any) -> ProfileSpec:
        body: dict[str, Any] = {
            "scope": scope,
            "baseline_run": self.design.baseline,
            "fault_experiments": self.fault_experiments,
            "interactions": (self.interaction_id,),
            "failure_modes": self.mode_ids,
        }
        body.update(over)
        return ProfileSpec(**body)


def full_evidence(tmp_path: Path, seeds: tuple[int, ...] = (1, 2, 3, 4)) -> EvidenceSet:
    w = world(tmp_path)
    d = four_cell(w, seeds=seeds, sigma=6.0, p=0.5, order=False)
    ia = run_interaction(
        w.registry,
        w.store,
        w.executor,
        d.investigation,
        d.spec(InteractionConfig(bootstrap_resamples=300)),
    )
    assert ia.analysis_id
    fx = (d.a.fault_experiment.id, d.b.fault_experiment.id, d.ab.fault_experiment.id)
    disc = run_discovery(w.registry, w.store, w.executor, d.investigation, [], list(fx))
    # Prefer modes that are still open to lifecycle changes (DISCOVERED/CANDIDATE) so tests that
    # change a mode's state never depend on which IDs happened to sort first.
    open_states = (FailureStatus.DISCOVERED, FailureStatus.CANDIDATE)
    ranked = sorted(w.registry.find(FailureMode), key=lambda m: (m.status not in open_states, m.id))
    modes = tuple(sorted(m.id for m in ranked[:4]))
    assert sum(m.status in open_states for m in ranked[:4]) >= 3, (
        "the evidence set needs open modes"
    )
    return EvidenceSet(d, disc, ia.analysis_id, modes)


def register_variant(
    w: EvalWorld, *, threshold: float = 5.0, n: int = 80, noise_every: int = 7, tag: str = "v"
) -> tuple[str, str]:
    """Register another model and/or dataset next to the world's own; returns (model_id, dataset_id)."""
    (w.workspace / "m" / f"data_{tag}.json").write_text(
        json.dumps(class_data(n, noise_every=noise_every)), encoding="utf-8"
    )
    (w.workspace / "m" / f"model_{tag}.json").write_text(
        json.dumps({"threshold": threshold, "proba": True}), encoding="utf-8"
    )
    d = ListDatasetAdapter.load(w.workspace / "m" / f"data_{tag}.json", version="1", options={})
    m = ThresholdClassifierAdapter.load(
        w.workspace / "m" / f"model_{tag}.json", version="1", device=DeviceKind.CPU, options={}
    )
    rec_d = RegisteredDataset(f"data_{tag}", d.metadata(), NOW, f"m/data_{tag}.json")
    rec_m = RegisteredModel(f"model_{tag}", m.metadata(), NOW, f"m/model_{tag}.json")
    for r in (rec_d, rec_m):
        if not w.registry.exists(type(r), r.id):
            w.registry.add(r)
    return rec_m.id, rec_d.id


def base_ids(w: EvalWorld) -> tuple[str, str]:
    return (
        next(m.id for m in w.registry.find(RegisteredModel) if m.name == "model"),
        next(d.id for d in w.registry.find(RegisteredDataset) if d.name == "data"),
    )


def launch_on(
    w: EvalWorld,
    model_id: str,
    dataset_id: str,
    *,
    evaluation: Any = EVAL,
    seeds: tuple[int, ...] = (21, 22, 23),
    sigma: float = 6.0,
    name: str = "other noise",
) -> FaultExperimentResult:
    spec = noise(sigma)
    return run_fault_experiment(
        w.registry,
        w.store,
        w.executor,
        model_id=model_id,
        dataset_id=dataset_id,
        base_spec=spec,
        fault_registry=FR,
        design=FaultDesign(spec.to_dict(), evaluation, seeds=seeds),
        name=name,
        source_root=w.workspace.parent,
    )
