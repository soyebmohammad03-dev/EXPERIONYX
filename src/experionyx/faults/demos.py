"""Real, laptop-sized fault experiments (`experionyx fault demo <name>` and `examples/fault_*.py`).

Each demo trains a tiny model on tiny data inside the workspace, runs a baseline evaluation through
the normal engine, then runs actual seeded fault experiments against that baseline. All numbers
are measured during execution."""

from pathlib import Path

from experionyx.adapters.capabilities import DeviceKind, TaskType
from experionyx.adapters.registry import default_registries
from experionyx.artifacts import LocalArtifactStore
from experionyx.demos import _register_and_run, _sklearn_config, run_demo
from experionyx.errors import AdapterUnavailableError, ExperionyxError
from experionyx.evaluation.config import EvaluationConfig, ScoreSource
from experionyx.execution import Executor
from experionyx.faults.design import FaultDesign, FaultLimits, SweepSpec
from experionyx.faults.lab import FaultExperimentResult, run_fault_experiment
from experionyx.faults.library import default_fault_registry
from experionyx.faults.spec import FaultScope, FaultSpec, ScopeKind
from experionyx.provenance import Provenance
from experionyx.sqlite import SqliteRegistry

FAULT_DEMOS = ("tabular", "tensor", "labels")


def _run_sweeps(
    workspace: Path,
    baseline_run_id: str,
    evaluation: EvaluationConfig,
    plans: list[
        tuple[
            str, str, dict[str, float], str, tuple[float, ...], tuple[int, ...], FaultScope | None
        ]
    ],
) -> list[FaultExperimentResult]:
    fr = default_fault_registry()
    reg = SqliteRegistry(workspace / "registry.sqlite")
    try:
        store = LocalArtifactStore(workspace / "experiments")
        executor = Executor(
            reg,
            store,
            source_root=Path.cwd(),
            adapters=default_registries(),
            inputs_root=workspace,
            device=DeviceKind.CPU,
        )
        (prov,) = reg.find(
            Provenance, run_id=baseline_run_id
        )  # the control's own model and dataset
        if prov.inputs is None or prov.inputs.model is None or prov.inputs.dataset is None:
            raise ExperionyxError("the baseline run has no bound model and dataset")
        model_id, data_id = prov.inputs.model.record_id, prov.inputs.dataset.record_id
        out = []
        for name, fault, params, param, values, seeds, scope in plans:
            spec: FaultSpec = fr.make(fault, seed=seeds[0], scope=scope, **params)  # type: ignore[arg-type]
            design = FaultDesign(
                spec.to_dict(),
                evaluation,
                seeds=seeds,
                sweep=SweepSpec(param, values),
                limits=FaultLimits(),
            )
            out.append(
                run_fault_experiment(
                    reg,
                    store,
                    executor,
                    model_id=model_id,
                    dataset_id=data_id,
                    base_spec=spec,
                    fault_registry=fr,
                    design=design,
                    name=name,
                    source_root=Path.cwd(),
                    baseline_run_id=baseline_run_id,
                )
            )
        return out
    finally:
        reg.close()


def _tabular(workspace: Path) -> list[FaultExperimentResult]:
    base = run_demo("sklearn-classification", workspace)
    scale = FaultScope(ScopeKind.ALL)
    return _run_sweeps(
        workspace, base.run.id, _sklearn_config(False),
        [
            ("iris: gaussian noise", "gaussian_noise", {"sigma": 0.0}, "sigma", (0.0, 0.1, 0.2, 0.3, 0.5, 1.0), (1, 2, 3, 4, 5), scale),
            ("iris: feature dropout", "feature_dropout", {"probability": 0.0}, "probability", (0.0, 0.1, 0.3, 0.5), (1, 2, 3), scale),
            ("iris: feature scaling (half of the columns)", "feature_scaling", {"factor": 1.0, "fraction": 0.5}, "factor", (1.0, 1.5, 2.0, 4.0), (1, 2, 3), scale),
        ],
    )  # fmt: skip


def _labels(workspace: Path) -> list[FaultExperimentResult]:
    base = run_demo("sklearn-classification", workspace)
    return _run_sweeps(
        workspace, base.run.id, _sklearn_config(False),
        [("iris: label flip (evaluation-label quality, not inference robustness)", "label_flip", {"rate": 0.0}, "rate", (0.0, 0.1, 0.2, 0.3, 0.5), tuple(range(1, 11)), FaultScope(ScopeKind.ALL))],
    )  # fmt: skip


def _tensor(workspace: Path) -> list[FaultExperimentResult]:
    try:
        import torch

        from experionyx.adapters.torch_adapter import TorchDatasetAdapter, TorchModelAdapter
    except ImportError as exc:
        raise AdapterUnavailableError(
            f"{exc}. Install with: pip install 'experionyx[torch]'"
        ) from exc
    name = "stripes-cnn"
    (workspace / "datasets").mkdir(parents=True, exist_ok=True)
    (workspace / "models").mkdir(parents=True, exist_ok=True)
    data_path, model_path = (
        workspace / "datasets" / f"{name}.pt",
        workspace / "models" / f"{name}.pt",
    )
    if (
        not data_path.exists()
    ):  # 8x8 images: class 0 = bright left half, class 1 = bright right half
        gen = torch.Generator().manual_seed(0)
        n = 240
        y = torch.arange(n) % 2
        x = 0.1 * torch.rand(n, 1, 8, 8, generator=gen)
        x[y == 0, :, :, :4] += 0.7
        x[y == 1, :, :, 4:] += 0.7
        torch.save(
            {
                "X": x.clamp(0, 1),
                "y": y,
                "split_train": torch.arange(0, 160),
                "split_test": torch.arange(160, 240),
            },
            data_path,
        )
    options: dict[str, object] = {"task": TaskType.CLASSIFICATION.value}
    dataset = TorchDatasetAdapter.load(data_path, version="1", options=options)
    if not model_path.exists():
        torch.manual_seed(0)
        net = torch.nn.Sequential(
            torch.nn.Conv2d(1, 4, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(16, 2),
        )
        (train,) = dataset.batches(10**6, "train")
        opt = torch.optim.Adam(net.parameters(), lr=0.05)
        for _ in range(60):
            opt.zero_grad()
            torch.nn.functional.cross_entropy(net(train.inputs), train.target).backward()
            opt.step()
        TorchModelAdapter.save(net, model_path, torch.zeros(1, 1, 8, 8))
    model = TorchModelAdapter.load(
        model_path,
        version="1",
        device=DeviceKind.CPU,
        options={"task": "CLASSIFICATION", "input_shape": [1, 8, 8]},
    )
    config = EvaluationConfig(split="test", batch_size=16, score_source=ScoreSource.SOFTMAX_LOGITS)
    base = _register_and_run(
        workspace,
        name=name,
        model=model,
        model_source=f"models/{name}.pt",
        dataset=dataset,
        dataset_source=f"datasets/{name}.pt",
        dataset_options=options,
        config=config,
        seed=0,
    )
    if base.status.value != "COMPLETED":
        raise ExperionyxError(f"the baseline evaluation failed: {base.error}")
    every = FaultScope(ScopeKind.ALL)
    return _run_sweeps(
        workspace, base.run.id, config,
        [
            ("stripes: gaussian noise", "gaussian_noise", {"sigma": 0.0}, "sigma", (0.0, 0.3, 1.0, 3.0, 6.0), (1, 2, 3), every),
            ("stripes: random occlusion", "random_occlusion", {"fraction": 0.25}, "fraction", (0.25, 0.5, 0.75), (1, 2, 3), every),
            ("stripes: brightness", "brightness", {"delta": 0.0}, "delta", (-0.5, -0.25, 0.0, 0.25), (0,), every),
            ("stripes: contrast", "contrast", {"factor": 1.0}, "factor", (0.25, 0.5, 1.0), (0,), every),
        ],
    )  # fmt: skip


def run_fault_demo(name: str, workspace: Path) -> list[FaultExperimentResult]:
    workspace.mkdir(parents=True, exist_ok=True)
    if name == "tabular":
        return _tabular(workspace)
    if name == "labels":
        return _labels(workspace)
    if name == "tensor":
        return _tensor(workspace)
    raise ExperionyxError(f"unknown fault demo {name!r} (choose from {list(FAULT_DEMOS)})")
