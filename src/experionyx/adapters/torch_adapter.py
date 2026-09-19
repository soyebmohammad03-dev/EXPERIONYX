"""PyTorch adapters (optional extra: `pip install 'experionyx[torch]'`). CPU-first.

Models are loaded from TorchScript archives (`torch.jit.save`), which carry architecture and
weights without running user-supplied Python; in-memory `nn.Module`s can be wrapped directly.
Datasets are map-style `torch.utils.data.Dataset`s or tensor files read with
`torch.load(weights_only=True)`. Nothing here enables CUDA/MPS unless explicitly requested, and
results may differ across hardware backends (no cross-device bitwise determinism is claimed).
"""

import hashlib
import pickle
import time
import warnings
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import ClassVar, Self

import torch
from torch.utils.data import Dataset, default_collate

import experionyx.validation as v
from experionyx.adapters.base import (
    BaseDatasetAdapter,
    BaseModelAdapter,
    Batch,
    InferenceResult,
    Inputs,
    Sample,
    SampleId,
)
from experionyx.adapters.batching import batch_slices
from experionyx.adapters.capabilities import (
    DatasetCapability,
    DeviceInfo,
    DeviceKind,
    ModelCapability,
    TaskType,
)
from experionyx.adapters.metadata import DatasetMetadata, ModelMetadata
from experionyx.adapters.schema import Modality, TensorSchema
from experionyx.artifacts import sha256_file
from experionyx.errors import (
    DatasetFingerprintError,
    DatasetLoadError,
    DeviceUnavailableError,
    InferenceError,
    InvalidDatasetError,
    InvalidModelError,
    ModelFingerprintError,
    ModelLoadError,
)
from experionyx.hashing import HASH_PREFIX, canonical_json

ADAPTER_VERSION = "1.0.0"
_SPLIT_CAPABILITIES = {
    "train": DatasetCapability.TRAIN_SPLIT,
    "validation": DatasetCapability.VALIDATION_SPLIT,
    "test": DatasetCapability.TEST_SPLIT,
}


def resolve_device(requested: DeviceKind) -> DeviceInfo:
    """CPU is the default. MPS/CUDA are used only when explicitly requested *and* available;
    otherwise DeviceUnavailableError (no silent fallback that would change what was measured)."""
    if requested is DeviceKind.MPS and not torch.backends.mps.is_available():
        raise DeviceUnavailableError("MPS was requested but is not available")
    if requested is DeviceKind.CUDA and not torch.cuda.is_available():
        raise DeviceUnavailableError("CUDA was requested but is not available")
    return DeviceInfo(requested, requested)


def _torch_device(kind: DeviceKind) -> torch.device:
    return torch.device({"CPU": "cpu", "MPS": "mps", "CUDA": "cuda"}[kind.value])


def _sync(kind: DeviceKind) -> None:
    if kind is DeviceKind.CUDA:
        torch.cuda.synchronize()
    elif kind is DeviceKind.MPS:
        torch.mps.synchronize()


def _tensor_bytes(h: "hashlib._Hash", t: torch.Tensor) -> None:
    """Feed a tensor's raw element bytes (any dtype) into `h` without a full extra copy."""
    flat = t.detach().to("cpu").contiguous().reshape(-1)
    if flat.numel():
        # zero-copy buffer; older numpy stubs do not declare ndarray as a Buffer
        h.update(flat.view(torch.uint8).numpy())  # type: ignore[arg-type]


def _rows(t: torch.Tensor) -> tuple[object, ...]:
    def freeze(x: object) -> object:
        return tuple(freeze(i) for i in x) if isinstance(x, list) else x

    return tuple(freeze(x) for x in t.detach().to("cpu").tolist())


def _dtype_name(t: torch.Tensor) -> str:
    return str(t.dtype).removeprefix("torch.")


class TorchModelAdapter(BaseModelAdapter):
    NAME: ClassVar[str] = "torch"
    VERSION: ClassVar[str] = ADAPTER_VERSION
    FRAMEWORK: ClassVar[str] = "pytorch"
    POSSIBLE_CAPABILITIES: ClassVar[frozenset[ModelCapability]] = frozenset(
        {ModelCapability.PREDICT, ModelCapability.BATCH_PREDICT}
    )
    DESCRIPTION: ClassVar[str] = "TorchScript archives or nn.Modules; CPU by default"

    def __init__(
        self,
        module: torch.nn.Module,
        *,
        version: str,
        device: DeviceKind = DeviceKind.CPU,
        task: TaskType = TaskType.UNKNOWN,
        input_shape: Sequence[int] | None = None,
        size_bytes: int | None = None,
        serialization_format: str | None = None,
        load_seconds: float = 0.0,
    ) -> None:
        """Wrap `module`. NOTE: this puts the module in eval mode and moves it to the device."""
        v.version("version", version)
        if not isinstance(module, torch.nn.Module):
            raise InvalidModelError(f"expected a torch.nn.Module, got {type(module).__name__}")
        self.device = resolve_device(device)
        self._device = _torch_device(device)
        started = time.perf_counter()
        self._module = module.to(self._device).eval()
        self.load_seconds = load_seconds + (time.perf_counter() - started)
        self._version, self._task = version, task
        self._input_shape = tuple(input_shape) if input_shape is not None else None
        self._size_bytes, self._format = size_bytes, serialization_format
        params = next(self._module.parameters(), None)
        self._dtype = params.dtype if params is not None else torch.float32
        self._fingerprint = self._compute_fingerprint()

    @staticmethod
    def save(module: torch.nn.Module, path: str | Path, example_inputs: torch.Tensor) -> Path:
        """Trace `module` with an example input and write a TorchScript archive."""
        target = Path(path)
        with warnings.catch_warnings():
            # torch.jit is deprecated in favour of torch.export; it still works (see docs).
            warnings.simplefilter("ignore", FutureWarning)
            torch.jit.trace(module.eval(), example_inputs).save(str(target))
        return target

    @classmethod
    def load(
        cls,
        source: str | Path,
        *,
        version: str,
        device: DeviceKind,
        options: Mapping[str, object],
    ) -> Self:
        resolve_device(device)
        path = Path(source)
        started = time.perf_counter()
        try:
            digest_size = sha256_file(path)[1]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)  # torch.jit is deprecated
                module = torch.jit.load(str(path), map_location="cpu")
        except OSError as exc:
            raise ModelLoadError(f"cannot read model file {path.name!r}: {exc}") from exc
        except RuntimeError as exc:
            raise ModelLoadError(
                f"{path.name!r} is not a loadable TorchScript archive: {exc}"
            ) from exc
        task = options.get("task", TaskType.UNKNOWN.value)
        shape = options.get("input_shape")
        if shape is not None and not (
            isinstance(shape, list | tuple) and all(isinstance(d, int) for d in shape)
        ):
            raise ModelLoadError("option input_shape must be a list of integers")
        try:
            task_type = TaskType(str(task))
        except ValueError as exc:
            raise ModelLoadError(f"unknown task {task!r}") from exc
        return cls(
            module,
            version=version,
            device=device,
            task=task_type,
            input_shape=shape,  # type: ignore[arg-type]
            size_bytes=digest_size,
            serialization_format="torchscript",
            load_seconds=time.perf_counter() - started,
        )

    @property
    def capabilities(self) -> frozenset[ModelCapability]:
        return self.POSSIBLE_CAPABILITIES

    def _compute_fingerprint(self) -> str:
        """SHA-256 over the module tree (submodule names + class names) and, for every entry of
        the state dict in sorted order, its name, dtype, shape and raw bytes. Independent of the
        file format and of `repr`. Train/eval mode and non-persistent buffers are not included."""
        h = hashlib.sha256()
        try:
            arch = [
                (n, getattr(m, "original_name", type(m).__name__))
                for n, m in self._module.named_modules()
            ]
            h.update(canonical_json({"kind": "torch-module", "modules": arch}).encode())
            for name, t in sorted(self._module.state_dict().items()):
                h.update(canonical_json([name, _dtype_name(t), list(t.shape)]).encode())
                _tensor_bytes(h, t)
        except (RuntimeError, ValueError) as exc:
            raise ModelFingerprintError(f"cannot fingerprint the module: {exc}") from exc
        return HASH_PREFIX + h.hexdigest()

    def fingerprint(self) -> str:
        return self._fingerprint

    def metadata(self) -> ModelMetadata:
        root = self._module
        params = list(root.parameters())
        shape = None if self._input_shape is None else (None, *self._input_shape)
        return ModelMetadata(
            adapter=self.NAME,
            adapter_version=self.VERSION,
            framework=f"{self.FRAMEWORK} {torch.__version__}",
            model_type=str(getattr(root, "original_name", type(root).__name__)),
            version=self._version,
            fingerprint=self._fingerprint,
            task=self._task,
            input_schema=TensorSchema(
                Modality.UNKNOWN, dtype=_dtype_name(params[0]) if params else None, shape=shape
            ),
            output_schema=None,
            parameter_count=sum(p.numel() for p in params),
            trainable_parameter_count=sum(p.numel() for p in params if p.requires_grad),
            serialization_format=self._format,
            size_bytes=self._size_bytes,
            capabilities=tuple(self.capabilities),
        )

    # -- inference ----------------------------------------------------------------------------

    def _tensor(self, inputs: Inputs) -> torch.Tensor:
        try:
            t = inputs if isinstance(inputs, torch.Tensor) else torch.as_tensor(inputs)
        except Exception as exc:
            raise InferenceError(f"inputs cannot be converted to a tensor: {exc}") from exc
        if t.ndim < 1 or t.shape[0] == 0:
            raise InferenceError(f"expected a non-empty batch, got shape {tuple(t.shape)}")
        if self._input_shape is not None and tuple(t.shape[1:]) != self._input_shape:
            raise InferenceError(
                f"expected per-sample shape {self._input_shape}, got {tuple(t.shape[1:])}"
            )
        if t.is_floating_point():
            t = t.to(self._dtype)
        return t.to(self._device)

    def _forward(self, x: torch.Tensor) -> tuple[tuple[object, ...], float]:
        started = time.perf_counter()
        try:
            with torch.no_grad():
                out = self._module(x)
            _sync(self.device.resolved)
        except Exception as exc:
            raise InferenceError(f"forward pass failed: {exc}") from exc
        seconds = time.perf_counter() - started
        if not isinstance(out, torch.Tensor) or out.ndim < 1 or out.shape[0] != x.shape[0]:
            raise InferenceError("the model must return one tensor with one row per sample")
        return _rows(out), seconds

    def _ids(self, ids: Sequence[SampleId] | None, n: int) -> tuple[SampleId, ...] | None:
        if ids is None:
            return None
        if len(ids) != n:
            raise InferenceError(f"got {len(ids)} sample ids for {n} samples")
        return tuple(ids)

    def _result(
        self,
        capability: ModelCapability,
        outputs: tuple[object, ...],
        seconds: Sequence[float],
        batch_size: int | None,
        ids: tuple[SampleId, ...] | None,
    ) -> InferenceResult:
        return InferenceResult(
            outputs=outputs,
            capability=capability,
            sample_count=len(outputs),
            batch_count=len(seconds),
            batch_size=batch_size,
            inference_seconds=float(sum(seconds)),
            batch_seconds=tuple(seconds),
            model_fingerprint=self._fingerprint,
            adapter=self.NAME,
            adapter_version=self.VERSION,
            device=self.device,
            sample_ids=ids,
        )

    def predict(
        self, inputs: Inputs, *, sample_ids: Sequence[SampleId] | None = None
    ) -> InferenceResult:
        x = self._tensor(inputs)
        ids = self._ids(sample_ids, x.shape[0])
        outputs, seconds = self._forward(x)
        return self._result(ModelCapability.PREDICT, outputs, (seconds,), None, ids)

    def batch_predict(
        self, inputs: Inputs, batch_size: int, *, sample_ids: Sequence[SampleId] | None = None
    ) -> InferenceResult:
        x = self._tensor(inputs)
        ids = self._ids(sample_ids, x.shape[0])
        outputs: list[object] = []
        timings: list[float] = []
        for chunk in batch_slices(x.shape[0], batch_size):
            rows, seconds = self._forward(x[chunk])
            outputs.extend(rows)
            timings.append(seconds)
        return self._result(ModelCapability.BATCH_PREDICT, tuple(outputs), timings, batch_size, ids)


def _parts(sample: object) -> tuple[torch.Tensor, ...]:
    items = sample if isinstance(sample, tuple | list) else (sample,)
    out = []
    for item in items:
        if isinstance(item, torch.Tensor):
            out.append(item)
        elif isinstance(item, bool | int | float):
            out.append(torch.as_tensor(item))
        else:
            raise InvalidDatasetError(f"unsupported sample component {type(item).__name__}")
    return tuple(out)


class TorchDatasetAdapter(BaseDatasetAdapter):
    """Wraps a map-style `torch.utils.data.Dataset` whose samples are a tensor or an `(x, y)`
    pair. Nothing is materialized: samples are read on demand."""

    NAME: ClassVar[str] = "torch"
    VERSION: ClassVar[str] = ADAPTER_VERSION
    FRAMEWORK: ClassVar[str] = "pytorch"
    POSSIBLE_CAPABILITIES: ClassVar[frozenset[DatasetCapability]] = frozenset(DatasetCapability)
    DESCRIPTION: ClassVar[str] = "Map-style torch Datasets (tensor or (x, y) samples)"

    def __init__(
        self,
        dataset: "Dataset[object]",
        *,
        version: str,
        task: TaskType = TaskType.UNKNOWN,
        splits: Mapping[str, Sequence[int]] | None = None,
        source: Mapping[str, object] | None = None,
    ) -> None:
        v.version("version", version)
        try:
            self._n = len(dataset)  # type: ignore[arg-type]
            first = dataset[0]
        except (TypeError, IndexError, KeyError) as exc:
            raise InvalidDatasetError(f"a non-empty map-style dataset is required: {exc}") from exc
        self._ds, self._version, self._task = dataset, version, task
        self._source = dict(source or {})
        self._labeled = isinstance(first, tuple | list) and len(first) == 2
        if isinstance(first, tuple | list) and len(first) not in (1, 2):
            raise InvalidDatasetError("samples must be a tensor, (input,) or (input, target)")
        _parts(first)  # validates component types
        self._splits: dict[str, tuple[int, ...]] = {}
        for name, idx in (splits or {}).items():
            items = tuple(int(i) for i in idx)
            if any(i < 0 or i >= self._n for i in items):
                raise InvalidDatasetError(f"split {name!r} has indices outside 0..{self._n - 1}")
            self._splits[v.text("split name", name)] = items
        self._fingerprint: str | None = None
        caps = {
            DatasetCapability.RANDOM_ACCESS,
            DatasetCapability.ITERATION,
            DatasetCapability.BATCHING,
        }
        if self._labeled:
            caps.add(DatasetCapability.LABELS)
        caps |= {c for n, c in _SPLIT_CAPABILITIES.items() if n in self._splits}
        self._caps = frozenset(caps)

    @classmethod
    def load(cls, source: str | Path, *, version: str, options: Mapping[str, object]) -> Self:
        """`source` is a file written by `torch.save({"X": tensor, "y": tensor?}, path)`; it is
        read with weights_only=True so no arbitrary code can run."""
        path = Path(source)
        try:
            file_digest, _ = sha256_file(path)
            data = torch.load(path, weights_only=True, map_location="cpu")
        except (OSError, RuntimeError, ValueError, pickle.UnpicklingError) as exc:
            raise DatasetLoadError(f"cannot load dataset {path.name!r}: {exc}") from exc
        if not isinstance(data, dict) or not isinstance(data.get("X"), torch.Tensor):
            raise InvalidDatasetError("dataset file must hold a dict with a tensor under 'X'")
        y = data.get("y")
        tensors = (data["X"],) if y is None else (data["X"], y)
        try:
            ds: Dataset[object] = torch.utils.data.TensorDataset(*tensors)  # type: ignore[assignment]
        except (RuntimeError, ValueError, AssertionError) as exc:  # torch asserts on size mismatch
            raise InvalidDatasetError(f"inconsistent tensors: {exc}") from exc
        raw_splits = {
            k.removeprefix("split_"): t.tolist() for k, t in data.items() if k.startswith("split_")
        }
        task = TaskType(str(options.get("task", TaskType.UNKNOWN.value)))
        return cls(
            ds, version=version, task=task, splits=raw_splits,
            source={"format": "torch-tensors", "file_sha256": file_digest},
        )  # fmt: skip

    @property
    def capabilities(self) -> frozenset[DatasetCapability]:
        return self._caps

    def splits(self) -> tuple[str, ...]:
        return tuple(sorted(self._splits))

    def _index(self, split: str | None) -> tuple[int, ...] | None:
        if split is None:
            return None
        if split not in self._splits:
            raise InvalidDatasetError(f"unknown split {split!r} (have {list(self.splits())})")
        return self._splits[split]

    def num_samples(self, split: str | None = None) -> int:
        idx = self._index(split)
        return self._n if idx is None else len(idx)

    def fingerprint(self) -> str:
        """SHA-256 over a header (length, labeled, split definitions) and every sample's
        component dtype, shape and raw bytes in dataset order. It reads the whole dataset once
        (one sample at a time, bounded memory) and is cached; order, values, dtypes and splits
        all matter. Samples that are not tensors/numbers cannot be fingerprinted."""
        if self._fingerprint is None:
            h = hashlib.sha256()
            header = {
                "kind": "torch-map",
                "length": self._n,
                "labeled": self._labeled,
                "splits": {k: list(a) for k, a in sorted(self._splits.items())},
            }
            h.update(canonical_json(header).encode())
            try:
                for i in range(self._n):
                    for part in _parts(self._ds[i]):
                        h.update(canonical_json([_dtype_name(part), list(part.shape)]).encode())
                        _tensor_bytes(h, part)
            except (InvalidDatasetError, RuntimeError, TypeError) as exc:
                raise DatasetFingerprintError(f"cannot fingerprint dataset: {exc}") from exc
            self._fingerprint = HASH_PREFIX + h.hexdigest()
        return self._fingerprint

    def metadata(self, *, deep: bool = False) -> DatasetMetadata:
        first = _parts(self._ds[0])
        x = first[0]
        y = first[1] if self._labeled else None
        class_count = missing = None
        if deep:  # scans every sample: opt-in
            labels: set[object] = set()
            nan = 0
            for i in range(self._n):
                parts = _parts(self._ds[i])
                nan += sum(int(torch.isnan(p).sum()) for p in parts if p.is_floating_point())
                if self._labeled and not parts[1].is_floating_point():
                    labels.add(parts[1].item() if parts[1].ndim == 0 else None)
            missing = nan
            if self._task is TaskType.CLASSIFICATION and None not in labels:
                class_count = len(labels)
        return DatasetMetadata(
            adapter=self.NAME,
            adapter_version=self.VERSION,
            family="torch-map-style",
            version=self._version,
            fingerprint=self.fingerprint(),
            task=self._task,
            num_samples=self._n,
            num_features=int(x.numel()) if x.ndim == 1 else None,
            input_schema=TensorSchema(
                Modality.UNKNOWN, dtype=_dtype_name(x), shape=(None, *x.shape)
            ),
            target_schema=None
            if y is None
            else TensorSchema(dtype=_dtype_name(y), shape=(None, *y.shape)),
            class_count=class_count,
            splits={k: len(a) for k, a in self._splits.items()},
            missing_values=missing,
            source=self._source,
            capabilities=tuple(self._caps),
        )

    def sample(self, index: int, split: str | None = None) -> Sample:
        idx = self._index(split)
        n = self.num_samples(split)
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < n:
            raise InvalidDatasetError(f"sample index {index!r} out of range 0..{n - 1}")
        i = index if idx is None else idx[index]
        parts = _parts(self._ds[i])
        return Sample(i, parts[0], parts[1] if self._labeled else None)

    def batches(self, batch_size: int, split: str | None = None) -> Iterator[Batch]:
        idx = self._index(split)
        for chunk in batch_slices(self.num_samples(split), batch_size):
            indices = tuple(range(chunk.start, chunk.stop)) if idx is None else idx[chunk]
            collated = default_collate([self._ds[i] for i in indices])
            if self._labeled:
                yield Batch(indices, collated[0], collated[1])
            else:  # unlabeled: a bare tensor, or a 1-tuple such as TensorDataset(x) yields
                inputs = collated[0] if isinstance(collated, list | tuple) else collated
                yield Batch(indices, inputs, None)
