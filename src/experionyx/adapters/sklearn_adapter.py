"""scikit-learn adapters (optional extra: `pip install 'experionyx[sklearn]'`).

SECURITY: sklearn models are persisted with joblib, which is pickle-based. Loading an untrusted
model file can execute arbitrary code. Only load artifacts you produced or trust.
"""

import copy
import hashlib
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import ClassVar, Self

import joblib
import numpy as np
import sklearn
from sklearn.base import is_classifier, is_outlier_detector, is_regressor
from sklearn.utils.validation import check_is_fitted

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
from experionyx.adapters.schema import Label, Modality, TensorSchema
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
_HASH_CHUNK_ROWS = 65536
_NUMERIC_KINDS = "biuf"


def _rows(values: object) -> tuple[object, ...]:
    """ndarray/list -> plain nested tuples (no numpy objects leak into results)."""
    listed = values.tolist() if isinstance(values, np.ndarray) else list(values)  # type: ignore[call-overload]

    def freeze(x: object) -> object:
        return tuple(freeze(i) for i in x) if isinstance(x, list) else x

    return tuple(freeze(x) for x in listed)


def _labels(values: np.ndarray) -> tuple[Label, ...] | None:
    out: list[Label] = []
    for x in values.tolist():
        if isinstance(x, bool) or not isinstance(x, int | str):
            return None  # e.g. float class labels: not representable, report as unknown
        out.append(x)
    return tuple(out)


class SklearnModelAdapter(BaseModelAdapter):
    NAME: ClassVar[str] = "sklearn"
    VERSION: ClassVar[str] = ADAPTER_VERSION
    FRAMEWORK: ClassVar[str] = "scikit-learn"
    THREAD_SAFE_INFERENCE: ClassVar[bool] = True  # predict on a fitted estimator is read-only
    POSSIBLE_CAPABILITIES: ClassVar[frozenset[ModelCapability]] = frozenset(
        {ModelCapability.PREDICT, ModelCapability.BATCH_PREDICT, ModelCapability.PREDICT_PROBA}
    )
    DESCRIPTION: ClassVar[str] = "Fitted scikit-learn estimators persisted with joblib (CPU only)"

    def __init__(
        self,
        estimator: object,
        *,
        version: str,
        fingerprint: str,
        size_bytes: int | None,
        load_seconds: float,
        device: DeviceInfo,
    ) -> None:
        self._estimator = estimator
        self._version, self._fingerprint = version, fingerprint
        self._size_bytes, self.load_seconds, self.device = size_bytes, load_seconds, device
        caps = {ModelCapability.PREDICT, ModelCapability.BATCH_PREDICT}
        if callable(getattr(estimator, "predict_proba", None)):  # honours available_if
            caps.add(ModelCapability.PREDICT_PROBA)
        self._caps = frozenset(caps)

    @staticmethod
    def save(estimator: object, path: str | Path) -> Path:
        """Persist a fitted estimator in the format `load` expects."""
        target = Path(path)
        joblib.dump(estimator, target)
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
        if device is not DeviceKind.CPU:
            raise DeviceUnavailableError("the sklearn adapter runs on CPU only")
        v.version("version", version)
        path = Path(source)
        started = time.perf_counter()
        try:
            digest, size = sha256_file(path)
        except OSError as exc:
            raise ModelLoadError(f"cannot read model file {path.name!r}: {exc}") from exc
        try:
            estimator = joblib.load(path)
        except Exception as exc:  # unpickling can raise anything
            raise ModelLoadError(f"cannot load {path.name!r} with joblib: {exc}") from exc
        if not (hasattr(estimator, "predict") and hasattr(estimator, "get_params")):
            raise InvalidModelError(f"{type(estimator).__name__} is not a scikit-learn estimator")
        try:
            check_is_fitted(estimator)
        except Exception as exc:
            raise InvalidModelError(f"estimator is not fitted: {exc}") from exc
        return cls(
            estimator,
            version=version,
            fingerprint=digest,
            size_bytes=size,
            load_seconds=time.perf_counter() - started,
            device=DeviceInfo.cpu(),
        )

    @property
    def capabilities(self) -> frozenset[ModelCapability]:
        return self._caps

    def fingerprint(self) -> str:
        """SHA-256 of the persisted artifact's bytes. Identifies *that file*: re-saving the same
        fitted estimator may produce different bytes (pickle embeds library versions and layout),
        so equal models saved separately can have different fingerprints."""
        if not self._fingerprint.startswith(HASH_PREFIX):
            raise ModelFingerprintError("no fingerprint available")
        return self._fingerprint

    def metadata(self) -> ModelMetadata:
        est = self._estimator
        cls = type(est)
        if is_classifier(est):
            task = TaskType.CLASSIFICATION
        elif is_regressor(est):
            task = TaskType.REGRESSION
        elif is_outlier_detector(est):
            task = TaskType.ANOMALY_DETECTION
        else:
            task = TaskType.UNKNOWN
        n_features = getattr(est, "n_features_in_", None)
        names = getattr(est, "feature_names_in_", None)
        classes = getattr(est, "classes_", None)
        return ModelMetadata(
            adapter=self.NAME,
            adapter_version=self.VERSION,
            framework=f"{self.FRAMEWORK} {sklearn.__version__}",
            model_type=f"{cls.__module__}.{cls.__qualname__}",
            version=self._version,
            fingerprint=self.fingerprint(),
            task=task,
            family=cls.__module__.split(".")[1] if cls.__module__.startswith("sklearn.") else None,
            input_schema=TensorSchema(
                Modality.TABULAR,
                shape=(None, int(n_features)) if isinstance(n_features, int | np.integer) else None,
                feature_names=tuple(str(n) for n in names) if names is not None else None,
            ),
            output_schema=TensorSchema(
                shape=(None,) if task in (TaskType.CLASSIFICATION, TaskType.REGRESSION) else None,
                class_labels=_labels(classes)
                if task is TaskType.CLASSIFICATION and isinstance(classes, np.ndarray)
                else None,
            ),
            parameter_count=None,  # not defined for scikit-learn estimators in general
            trainable_parameter_count=None,
            serialization_format="joblib",
            size_bytes=self._size_bytes,
            capabilities=tuple(self._caps),
        )

    # -- optional ParameterAccess (Model Stress Laboratory) -------------------------------------

    SAFE_PARAMETERS: ClassVar[tuple[str, ...]] = ("coef_", "intercept_")

    def parameter_arrays(self) -> dict[str, np.ndarray]:
        """Copies of the fitted linear parameters (`coef_`, `intercept_`) when the estimator has
        them as floating-point arrays; other estimators expose none (no unsafe attribute access)."""
        out: dict[str, np.ndarray] = {}
        for name in self.SAFE_PARAMETERS:
            value = getattr(self._estimator, name, None)
            if isinstance(value, np.ndarray) and value.dtype.kind == "f":
                out[name] = value.copy()
        return out

    def with_parameters(self, arrays: Mapping[str, object]) -> "SklearnModelAdapter":
        current = self.parameter_arrays()
        for name, value in arrays.items():
            if name not in current:
                raise InvalidModelError(
                    f"{name!r} is not a safely accessible parameter of this model"
                )
            arr = np.asarray(value)
            if arr.shape != current[name].shape or arr.dtype.kind != "f":
                raise InvalidModelError(
                    f"replacement for {name!r} must be a float array of shape {current[name].shape}"
                )
        clone = copy.deepcopy(self._estimator)
        for name, value in arrays.items():
            setattr(clone, name, np.asarray(value, dtype=current[name].dtype).copy())
        return type(self)(
            clone,
            version=self._version,
            fingerprint=self._fingerprint,
            size_bytes=self._size_bytes,
            load_seconds=self.load_seconds,
            device=self.device,
        )

    # -- inference ----------------------------------------------------------------------------

    def _matrix(self, inputs: Inputs) -> np.ndarray:
        try:
            x = np.asarray(inputs)
        except Exception as exc:
            raise InferenceError(f"inputs cannot be converted to an array: {exc}") from exc
        if x.ndim != 2 or x.shape[0] == 0:
            raise InferenceError(f"expected a non-empty 2-D array, got shape {x.shape}")
        if x.dtype.kind not in _NUMERIC_KINDS:
            raise InferenceError(f"expected numeric inputs, got dtype {x.dtype}")
        expected = getattr(self._estimator, "n_features_in_", None)
        if isinstance(expected, int | np.integer) and x.shape[1] != expected:
            raise InferenceError(f"model expects {expected} features, got {x.shape[1]}")
        return x

    def _ids(self, ids: Sequence[SampleId] | None, n: int) -> tuple[SampleId, ...] | None:
        if ids is None:
            return None
        if len(ids) != n:
            raise InferenceError(f"got {len(ids)} sample ids for {n} samples")
        return tuple(ids)

    def _run(self, method: str, x: np.ndarray) -> tuple[tuple[object, ...], float]:
        started = time.perf_counter()
        try:
            y = getattr(self._estimator, method)(x)
        except Exception as exc:
            raise InferenceError(f"{method} failed: {exc}") from exc
        return _rows(y), time.perf_counter() - started

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
            model_fingerprint=self.fingerprint(),
            adapter=self.NAME,
            adapter_version=self.VERSION,
            device=self.device,
            sample_ids=ids,
        )

    def predict(
        self, inputs: Inputs, *, sample_ids: Sequence[SampleId] | None = None
    ) -> InferenceResult:
        x = self._matrix(inputs)
        ids = self._ids(sample_ids, x.shape[0])
        outputs, seconds = self._run("predict", x)
        return self._result(ModelCapability.PREDICT, outputs, (seconds,), None, ids)

    def predict_proba(
        self, inputs: Inputs, *, sample_ids: Sequence[SampleId] | None = None
    ) -> InferenceResult:
        self.require(ModelCapability.PREDICT_PROBA)
        x = self._matrix(inputs)
        ids = self._ids(sample_ids, x.shape[0])
        outputs, seconds = self._run("predict_proba", x)
        return self._result(ModelCapability.PREDICT_PROBA, outputs, (seconds,), None, ids)

    def batch_predict(
        self, inputs: Inputs, batch_size: int, *, sample_ids: Sequence[SampleId] | None = None
    ) -> InferenceResult:
        x = self._matrix(inputs)
        ids = self._ids(sample_ids, x.shape[0])
        outputs: list[object] = []
        timings: list[float] = []
        for chunk in batch_slices(x.shape[0], batch_size):
            rows, seconds = self._run("predict", x[chunk])
            outputs.extend(rows)
            timings.append(seconds)
        return self._result(ModelCapability.BATCH_PREDICT, tuple(outputs), timings, batch_size, ids)


_BUILTIN = {
    "iris": ("load_iris", TaskType.CLASSIFICATION),
    "wine": ("load_wine", TaskType.CLASSIFICATION),
    "diabetes": ("load_diabetes", TaskType.REGRESSION),
}
_SPLIT_CAPABILITIES = {
    "train": DatasetCapability.TRAIN_SPLIT,
    "validation": DatasetCapability.VALIDATION_SPLIT,
    "test": DatasetCapability.TEST_SPLIT,
}


def _stream(h: "hashlib._Hash", array: np.ndarray) -> None:
    """Feed an array's little-endian bytes into `h` in row chunks (bounded extra memory)."""
    le = array.astype(array.dtype.newbyteorder("<"), copy=False)
    for start in range(0, len(le), _HASH_CHUNK_ROWS):
        chunk = np.ascontiguousarray(le[start : start + _HASH_CHUNK_ROWS])
        # zero-copy buffer; older numpy stubs do not declare ndarray as a Buffer
        h.update(chunk.reshape(-1).view(np.uint8))  # type: ignore[arg-type]


class SklearnDatasetAdapter(BaseDatasetAdapter):
    """In-memory tabular data: a numeric 2-D feature matrix, an optional 1-D target and optional
    named index splits. Sources: `builtin:iris|wine|diabetes` (bundled with scikit-learn, offline)
    or a `.npz` file with arrays `X`, optional `y`, `feature_names`, and `split_<name>`."""

    NAME: ClassVar[str] = "sklearn"
    VERSION: ClassVar[str] = ADAPTER_VERSION
    FRAMEWORK: ClassVar[str] = "scikit-learn"
    POSSIBLE_CAPABILITIES: ClassVar[frozenset[DatasetCapability]] = frozenset(DatasetCapability)
    DESCRIPTION: ClassVar[str] = "Tabular numpy data with optional target and named splits"

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        *,
        version: str,
        feature_names: Sequence[str] | None = None,
        task: TaskType | None = None,
        splits: Mapping[str, Sequence[int] | np.ndarray] | None = None,
        source: Mapping[str, object] | None = None,
    ) -> None:
        v.version("version", version)
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[0] == 0 or X.dtype.kind not in _NUMERIC_KINDS:
            raise InvalidDatasetError(
                f"X must be a non-empty 2-D numeric array (got shape {X.shape}, dtype {X.dtype})"
            )
        self._X = X
        n = X.shape[0]
        if y is not None:
            y = np.asarray(y)
            if y.ndim != 1 or len(y) != n or y.dtype.kind == "O":
                raise InvalidDatasetError(
                    f"y must be 1-D with {n} entries and a non-object dtype "
                    f"(got {y.shape}, {y.dtype})"
                )
        self._y = y
        if feature_names is not None and len(feature_names) != X.shape[1]:
            raise InvalidDatasetError("feature_names must match the number of columns")
        self._names = tuple(feature_names) if feature_names is not None else None
        self._splits: dict[str, np.ndarray] = {}
        for name, idx in (splits or {}).items():
            arr = np.asarray(idx)
            if arr.ndim != 1 or arr.dtype.kind not in "iu":
                raise InvalidDatasetError(f"split {name!r} must be a 1-D integer index array")
            if len(arr) and (arr.min() < 0 or arr.max() >= n):
                raise InvalidDatasetError(f"split {name!r} has indices outside 0..{n - 1}")
            self._splits[v.text("split name", name)] = arr.astype("<i8")
        self._version = version
        self._source = dict(source or {})
        self._task = task or self._infer_task()
        self._fingerprint: str | None = None
        caps = {
            DatasetCapability.RANDOM_ACCESS,
            DatasetCapability.ITERATION,
            DatasetCapability.BATCHING,
        }
        if y is not None:
            caps.add(DatasetCapability.LABELS)
        caps |= {c for name, c in _SPLIT_CAPABILITIES.items() if name in self._splits}
        self._caps = frozenset(caps)

    def _infer_task(self) -> TaskType:
        """Only what is unambiguous from the target dtype; otherwise UNKNOWN."""
        if self._y is None:
            return TaskType.UNKNOWN
        kind = self._y.dtype.kind
        if kind in "USb":
            return TaskType.CLASSIFICATION
        return TaskType.REGRESSION if kind == "f" else TaskType.UNKNOWN

    @classmethod
    def load(cls, source: str | Path, *, version: str, options: Mapping[str, object]) -> Self:
        text = str(source)
        try:
            if text.startswith("builtin:"):
                return cls._builtin(text.removeprefix("builtin:"), version, options)
            return cls._npz(Path(text), version)
        except InvalidDatasetError:
            raise
        except (OSError, ValueError, KeyError) as exc:
            raise DatasetLoadError(f"cannot load dataset {text!r}: {exc}") from exc

    @classmethod
    def _builtin(cls, name: str, version: str, options: Mapping[str, object]) -> Self:
        import sklearn.datasets as datasets

        if name not in _BUILTIN:
            raise DatasetLoadError(f"unknown built-in dataset {name!r} (have {sorted(_BUILTIN)})")
        test_size = options.get("test_size", 0.25)
        seed = options.get("seed", 0)
        if not isinstance(test_size, float) or not 0 < test_size < 1:
            raise DatasetLoadError("option test_size must be a float in (0, 1)")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise DatasetLoadError("option seed must be an integer")
        loader, task = _BUILTIN[name]
        bunch = getattr(datasets, loader)()
        n = len(bunch.data)
        order = np.random.RandomState(seed).permutation(n)
        cut = n - max(1, round(n * test_size))
        return cls(
            bunch.data,
            bunch.target,
            version=version,
            feature_names=list(bunch.feature_names),
            task=task,
            splits={"train": order[:cut], "test": order[cut:]},
            source={
                "loader": f"sklearn.datasets.{loader}",
                "sklearn_version": sklearn.__version__,
                "test_size": test_size,
                "seed": seed,
            },
        )

    @classmethod
    def _npz(cls, path: Path, version: str) -> Self:
        file_digest, _ = sha256_file(path)
        with np.load(path, allow_pickle=False) as data:  # allow_pickle=False: no code execution
            arrays = {k: data[k] for k in data.files}
        if "X" not in arrays:
            raise InvalidDatasetError("npz dataset must contain an array named 'X'")
        names = arrays.get("feature_names")
        return cls(
            arrays["X"],
            arrays.get("y"),
            version=version,
            feature_names=[str(n) for n in names] if names is not None else None,
            splits={
                k.removeprefix("split_"): a for k, a in arrays.items() if k.startswith("split_")
            },
            source={"format": "npz", "file_sha256": file_digest},
        )

    # -- protocol -----------------------------------------------------------------------------

    @property
    def capabilities(self) -> frozenset[DatasetCapability]:
        return self._caps

    def splits(self) -> tuple[str, ...]:
        return tuple(sorted(self._splits))

    def _index(self, split: str | None) -> np.ndarray | None:
        if split is None:
            return None
        if split not in self._splits:
            raise InvalidDatasetError(f"unknown split {split!r} (have {list(self.splits())})")
        return self._splits[split]

    def num_samples(self, split: str | None = None) -> int:
        idx = self._index(split)
        return int(self._X.shape[0] if idx is None else len(idx))

    def fingerprint(self) -> str:
        """SHA-256 over: canonical header (X/y dtype+shape, feature names, split names and sizes),
        then X bytes, y bytes, and each split's index bytes (sorted by name). Row order, column
        names, dtypes, targets and splits all matter; nothing is sorted or normalized (bytes are
        little-endian). It identifies this in-memory representation, not the file it came from."""
        if self._fingerprint is None:
            h = hashlib.sha256()
            try:
                header = {
                    "kind": "tabular",
                    "X": [str(self._X.dtype), list(self._X.shape)],
                    "y": None if self._y is None else [str(self._y.dtype), list(self._y.shape)],
                    "feature_names": list(self._names) if self._names is not None else None,
                    "splits": {k: len(a) for k, a in sorted(self._splits.items())},
                }
                h.update(canonical_json(header).encode("utf-8"))
                _stream(h, self._X)
                if self._y is not None:
                    _stream(h, self._y)
                for _, arr in sorted(self._splits.items()):
                    _stream(h, arr)
            except (ValueError, TypeError) as exc:
                raise DatasetFingerprintError(f"cannot fingerprint dataset: {exc}") from exc
            self._fingerprint = HASH_PREFIX + h.hexdigest()
        return self._fingerprint

    def metadata(self, *, deep: bool = False) -> DatasetMetadata:
        y = self._y
        classification = self._task is TaskType.CLASSIFICATION and y is not None
        classes = np.unique(y) if classification and y is not None else None
        missing = None
        if deep:
            missing = sum(
                int(np.isnan(a).sum())
                for a in (self._X, y)
                if a is not None and a.dtype.kind == "f"
            )
        return DatasetMetadata(
            adapter=self.NAME,
            adapter_version=self.VERSION,
            family="tabular",
            version=self._version,
            fingerprint=self.fingerprint(),
            task=self._task,
            num_samples=int(self._X.shape[0]),
            num_features=int(self._X.shape[1]),
            input_schema=TensorSchema(
                Modality.TABULAR,
                dtype=str(self._X.dtype),
                shape=(None, int(self._X.shape[1])),
                feature_names=self._names,
            ),
            target_schema=None
            if y is None
            else TensorSchema(
                dtype=str(y.dtype),
                shape=(None,),
                class_labels=_labels(classes) if classes is not None else None,
            ),
            class_count=len(classes) if classes is not None else None,
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
        i = int(index if idx is None else idx[index])
        return Sample(i, self._X[i], None if self._y is None else self._y[i])

    def batches(self, batch_size: int, split: str | None = None) -> Iterator[Batch]:
        idx = self._index(split)
        for chunk in batch_slices(self.num_samples(split), batch_size):
            if idx is None:
                yield Batch(
                    tuple(range(chunk.start, chunk.stop)),
                    self._X[chunk],
                    None if self._y is None else self._y[chunk],
                )
            else:
                sel = idx[chunk]
                yield Batch(
                    tuple(int(i) for i in sel),
                    self._X[sel],
                    None if self._y is None else self._y[sel],
                )
