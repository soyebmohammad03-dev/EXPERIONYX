"""Resource validation fixtures (ENGINEERING VALIDATION DATA, not findings about any model or machine).

`Sim` is a deterministic simulated clock with process CPU and memory backends: the model adapter
`SimLinear` advances it by a chosen cost on every inference call (and can fail or grow memory on chosen
calls), so every expected latency, throughput, timeout and memory number in the tests is computed by hand
from the script, never through the engine. The world is the controlled linear classifier of
`stress_helpers` (120 samples, split "test"). Real-clock measurements are tested separately, on real
models, without asserting any timing value."""

import dataclasses
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, ClassVar

from eval_helpers import EvalWorld, pure_registries
from experionyx.adapters.base import InferenceResult, SampleId
from experionyx.adapters.records import RegisteredDataset, RegisteredModel
from experionyx.adapters.registry import AdapterRegistries, AdapterRegistry
from experionyx.domain import Investigation, to_jsonable
from experionyx.execution import Executor
from experionyx.resources import engine as re_
from experionyx.resources.entities import ResourceAnalysis
from experionyx.resources.measure import Probe
from experionyx.resources.spec import ResourceSpec
from pure_adapters import LinearProbaAdapter, ListDatasetAdapter
from stress_helpers import N, stress_world

MB = 1_000_000


class Sim:
    """A scripted clock. `duration` is seconds per inference call (a number or f(call_index, n));
    `fail_on` are 0-based call numbers (counted across every pass) that raise; `rss_step` bytes are
    added to the process peak on every call."""

    def __init__(self, duration: float | Callable[[int, int], float] = 0.01, cpu_per_call: float = 0.008, rss: int = 200 * MB, rss_step: int = 0) -> None:  # fmt: skip
        self.t, self.cpu, self.rss = 0.0, 0.0, rss
        self.duration, self.cpu_per_call, self.rss_step = duration, cpu_per_call, rss_step
        self.calls = 0
        self.fail_on: set[int] = set()
        self.lock = threading.Lock()

    def now(self) -> float:
        return self.t

    def cpu_now(self) -> float:
        return self.cpu

    def rss_now(self) -> int:
        return self.rss

    def on_call(self, n: int) -> None:
        with self.lock:
            i = self.calls
            self.calls += 1
            d = self.duration(i, n) if callable(self.duration) else self.duration
            self.t += d
            self.cpu += self.cpu_per_call
            self.rss += self.rss_step
        if i in self.fail_on:
            raise RuntimeError(f"simulated inference failure on call {i}")

    def probe(self, *, cpu: bool = True, memory: bool = True, resolution: float = 1e-9) -> Probe:
        return Probe("simulated", self.now, resolution, self.cpu_now if cpu else None, "simulated process CPU", self.rss_now if memory else None, "simulated peak RSS")  # fmt: skip


class SimLinear(LinearProbaAdapter):
    """The linear classifier whose inference costs simulated time. It does NOT declare thread-safe
    inference (the default), so a concurrent request must be refused as UNAVAILABLE."""

    SIM: ClassVar[Sim | None] = None

    def _cost(self, inputs: Any) -> None:
        if SimLinear.SIM is not None:
            SimLinear.SIM.on_call(len(inputs))

    def predict(self, inputs: Any, *, sample_ids: Any = None) -> InferenceResult:
        self._cost(inputs)
        return super().predict(inputs, sample_ids=sample_ids)

    def predict_proba(self, inputs: Any, *, sample_ids: Any = None) -> InferenceResult:
        self._cost(inputs)
        return super().predict_proba(inputs, sample_ids=sample_ids)


class SafeSimLinear(SimLinear):
    THREAD_SAFE_INFERENCE: ClassVar[bool] = True


class SleepLinear(LinearProbaAdapter):
    """Real 4 ms sleep per call and a declared thread-safe inference: real threads, real clock."""

    THREAD_SAFE_INFERENCE: ClassVar[bool] = True
    ACTIVE: ClassVar[list[int]] = [0, 0]  # current and maximum concurrent calls
    LOCK: ClassVar[threading.Lock] = threading.Lock()

    def predict(self, inputs: Any, *, sample_ids: Sequence[SampleId] | None = None) -> InferenceResult:  # fmt: skip
        with SleepLinear.LOCK:
            SleepLinear.ACTIVE[0] += 1
            SleepLinear.ACTIVE[1] = max(SleepLinear.ACTIVE)
        try:
            time.sleep(0.004)
            return super().predict(inputs, sample_ids=sample_ids)
        finally:
            with SleepLinear.LOCK:
                SleepLinear.ACTIVE[0] -= 1


def registries(model_cls: type) -> AdapterRegistries:
    models: AdapterRegistry = AdapterRegistry("model")  # type: ignore[type-arg]
    datasets: AdapterRegistry = AdapterRegistry("dataset")  # type: ignore[type-arg]
    models.register("linear", model_cls)
    datasets.register("list", ListDatasetAdapter)
    return AdapterRegistries(models, datasets)


class View:
    """A resource analysis with its frozen mappings as plain dicts, so tests can index them."""

    def __init__(self, entity: ResourceAnalysis) -> None:
        self.entity = entity
        self.summary: dict[str, Any] = to_jsonable(entity.summary)  # type: ignore[assignment]
        self.spec: dict[str, Any] = to_jsonable(entity.spec)  # type: ignore[assignment]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.entity, name)


@dataclasses.dataclass
class RWorld:
    w: EvalWorld
    model_id: str
    dataset_id: str
    investigation: str

    def spec(self, **over: Any) -> ResourceSpec:
        body: dict[str, Any] = {"model_id": self.model_id, "dataset_id": self.dataset_id, "split": "test", "batch_size": 30, "repeats": 5, "warmup_trials": 1}  # fmt: skip
        body.update(over)
        return ResourceSpec.from_dict(body)

    def run(self, spec: ResourceSpec) -> View:
        out = re_.run_resource_request(self.w.registry, self.w.store, self.w.executor, self.investigation, spec)  # fmt: skip
        assert out.analysis_id, out
        return View(self.w.registry.get(ResourceAnalysis, out.analysis_id))


def resource_world(tmp_path: Path, model_cls: type = SimLinear, **over: Any) -> RWorld:
    w, mid, did = stress_world(tmp_path, **over)
    w = dataclasses.replace(w, executor=Executor(w.registry, w.store, source_root=tmp_path, adapters=registries(model_cls), inputs_root=w.workspace))  # fmt: skip
    inv = w.registry.find(Investigation)[0]
    return RWorld(w, mid, did, inv.id)


def model_id_of(w: EvalWorld) -> str:
    return next(m.id for m in w.registry.find(RegisteredModel) if m.name == "model")


def dataset_id_of(w: EvalWorld) -> str:
    return next(d.id for d in w.registry.find(RegisteredDataset) if d.name == "data")


__all__ = ["MB", "N", "RWorld", "SafeSimLinear", "Sim", "SimLinear", "SleepLinear", "View", "pure_registries", "registries", "resource_world"]  # fmt: skip
