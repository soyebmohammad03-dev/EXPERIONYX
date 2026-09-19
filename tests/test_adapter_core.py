"""Adapter registries, capabilities, metadata and the pure-Python reference adapters."""

import importlib.metadata
import json
from pathlib import Path
from typing import ClassVar

import pytest

import pure_adapters
from contracts import DatasetAdapterContract, ModelAdapterContract
from experionyx.adapters.base import AdapterInfo
from experionyx.adapters.batching import batch_slices
from experionyx.adapters.capabilities import (
    DatasetCapability,
    DeviceInfo,
    DeviceKind,
    ModelCapability,
    TaskType,
)
from experionyx.adapters.metadata import DatasetMetadata, ModelMetadata
from experionyx.adapters.registry import AdapterRegistry, default_registries
from experionyx.adapters.schema import FeatureKind, Modality, TensorSchema
from experionyx.domain import to_jsonable
from experionyx.errors import (
    AdapterNotFoundError,
    AdapterUnavailableError,
    DuplicateAdapterError,
    InferenceError,
    ValidationError,
)
from pure_adapters import ConstantModelAdapter, ListDatasetAdapter

DIGEST = "sha256:" + "ab" * 32


# --- registry ---------------------------------------------------------------------------------


def test_register_resolve_list_and_inspect() -> None:
    reg: AdapterRegistry[pure_adapters.ConstantModelAdapter] = AdapterRegistry("model")
    reg.register("constant", ConstantModelAdapter)
    assert reg.names() == ("constant",)
    assert reg.resolve("constant") is ConstantModelAdapter
    (status,) = reg.status()
    assert status.available
    assert status.info == AdapterInfo(
        "model", "constant", "1.0.0", "pure-python", ("BATCH_PREDICT", "PREDICT"), ""
    )


def test_duplicate_and_unknown_and_invalid_names() -> None:
    reg: AdapterRegistry[pure_adapters.ConstantModelAdapter] = AdapterRegistry("model")
    reg.register("constant", ConstantModelAdapter)
    with pytest.raises(DuplicateAdapterError):
        reg.register("constant", ConstantModelAdapter)
    with pytest.raises(AdapterNotFoundError, match="registered"):
        reg.resolve("nope")
    with pytest.raises(ValidationError):
        reg.register(" bad ", ConstantModelAdapter)


def test_lazy_adapters_report_unavailability_with_an_install_hint() -> None:
    reg: AdapterRegistry[pure_adapters.ConstantModelAdapter] = AdapterRegistry(
        "model", lambda name: f"pip install 'experionyx[{name}]'"
    )
    reg.register_lazy("ghost", "no_such_framework_module:Adapter")
    reg.register("constant", ConstantModelAdapter)
    with pytest.raises(AdapterUnavailableError, match=r"experionyx\[ghost\]"):
        reg.resolve("ghost")
    by_name = {s.name: s for s in reg.status()}
    assert not by_name["ghost"].available
    assert by_name["ghost"].info is None
    assert by_name["constant"].available


def test_third_party_adapters_are_discovered_via_entry_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eps = [
        importlib.metadata.EntryPoint(
            "constant", "pure_adapters:ConstantModelAdapter", "experionyx.model_adapters"
        )
    ]
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda group: eps)
    regs = default_registries(entry_points=True)
    assert "constant" in regs.models.names()
    assert regs.models.resolve("constant") is ConstantModelAdapter


def test_default_registries_are_independent_fresh_objects() -> None:
    a, b = default_registries(), default_registries()
    a.models.register("extra", ConstantModelAdapter)
    assert "extra" not in b.models.names()
    assert {"sklearn", "torch"} <= set(b.models.names())
    assert {"sklearn", "torch"} <= set(b.datasets.names())


# --- capabilities -----------------------------------------------------------------------------


def test_capability_checks_before_use(tmp_path: Path) -> None:
    (tmp_path / "m.json").write_text('{"value": 2}')
    m = ConstantModelAdapter.load(
        tmp_path / "m.json", version="1", device=DeviceKind.CPU, options={}
    )
    assert m.supports(ModelCapability.PREDICT)
    assert not m.supports(ModelCapability.PREDICT_PROBA)
    from experionyx.errors import UnsupportedCapabilityError

    with pytest.raises(UnsupportedCapabilityError, match="PREDICT_PROBA"):
        m.require(ModelCapability.PREDICT_PROBA)
    with pytest.raises(UnsupportedCapabilityError):
        m.predict_proba([[1.0]])


def test_device_info_forbids_silent_fallback() -> None:
    assert DeviceInfo.cpu() == DeviceInfo(DeviceKind.CPU, DeviceKind.CPU)
    with pytest.raises(ValidationError):
        DeviceInfo(DeviceKind.MPS, DeviceKind.CPU)


def test_task_taxonomy_has_explicit_unknown_and_custom() -> None:
    assert {"CLASSIFICATION", "REGRESSION", "CUSTOM", "UNKNOWN"} <= {t.value for t in TaskType}


# --- metadata and schemas ---------------------------------------------------------------------


def test_metadata_rejects_fabricated_or_inconsistent_values() -> None:
    base = {
        "adapter": "a",
        "adapter_version": "1",
        "framework": "f",
        "model_type": "M",
        "version": "1",
        "fingerprint": DIGEST,
    }

    def make(**over: object) -> ModelMetadata:
        return ModelMetadata(**{**base, **over})  # type: ignore[arg-type]

    make()
    with pytest.raises(ValidationError):
        make(parameter_count=10, trainable_parameter_count=11)
    with pytest.raises(ValidationError):
        make(parameter_count=-1)
    with pytest.raises(ValidationError):
        make(fingerprint="abc")
    assert make().parameter_count is None  # unavailable, not 0


def test_metadata_round_trips_with_schemas() -> None:
    schema = TensorSchema(
        Modality.TABULAR,
        "float64",
        (None, 2),
        ("a", "b"),
        (FeatureKind.NUMERIC, FeatureKind.UNKNOWN),
        (0, 1, "x"),
    )
    meta = ModelMetadata(
        "a", "1", "f", "M", "1", DIGEST, TaskType.CLASSIFICATION, "fam", schema, schema,
        5, 3, "fmt", 10, (ModelCapability.PREDICT, ModelCapability.BATCH_PREDICT),
    )  # fmt: skip
    again = ModelMetadata.from_dict(json.loads(json.dumps(to_jsonable(meta))))
    assert again == meta
    dmeta = DatasetMetadata(
        "a", "1", "tabular", "1", DIGEST, TaskType.REGRESSION, 10, 2, schema, None, None,
        {"train": 8, "test": 2}, None, {"loader": "x"}, (DatasetCapability.BATCHING,),
    )  # fmt: skip
    assert DatasetMetadata.from_dict(json.loads(json.dumps(to_jsonable(dmeta)))) == dmeta


def test_schema_validation() -> None:
    with pytest.raises(ValidationError):
        TensorSchema(shape=(-1,))
    with pytest.raises(ValidationError):
        TensorSchema(feature_names=("a",), feature_kinds=(FeatureKind.NUMERIC, FeatureKind.NUMERIC))
    with pytest.raises(ValidationError):
        TensorSchema(class_labels=(1.5,))  # type: ignore[arg-type]
    assert TensorSchema().shape is None  # unknown stays unknown


def test_batch_slices() -> None:
    assert list(batch_slices(5, 2)) == [slice(0, 2), slice(2, 4), slice(4, 5)]
    assert list(batch_slices(0, 3)) == []
    with pytest.raises(InferenceError):
        list(batch_slices(5, 0))


# --- the contract suites, run against the pure-Python reference adapters ----------------------


def _write(path: Path, payload: object) -> str:
    path.write_text(json.dumps(payload))
    return str(path)


class TestConstantModelContract(ModelAdapterContract):
    expected_capabilities = ConstantModelAdapter.POSSIBLE_CAPABILITIES

    @pytest.fixture
    def model_source(self, tmp_path: Path) -> str:
        return _write(tmp_path / "a.json", {"value": 1.5})

    @pytest.fixture
    def adapter(self, model_source: str) -> ConstantModelAdapter:
        return ConstantModelAdapter.load(
            model_source, version="1", device=DeviceKind.CPU, options={}
        )

    @pytest.fixture
    def reloaded(self, model_source: str) -> ConstantModelAdapter:
        return ConstantModelAdapter.load(
            model_source, version="1", device=DeviceKind.CPU, options={}
        )

    @pytest.fixture
    def different(self, tmp_path: Path) -> ConstantModelAdapter:
        return ConstantModelAdapter.load(
            _write(tmp_path / "b.json", {"value": 2.5}),
            version="1",
            device=DeviceKind.CPU,
            options={},
        )

    @pytest.fixture
    def valid_inputs(self) -> list[list[float]]:
        return [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 1.0]]

    @pytest.fixture
    def invalid_inputs(self) -> list[object]:
        return [[], "text", 5, [1, 2]]

    @pytest.fixture
    def missing_source(self, tmp_path: Path) -> str:
        return str(tmp_path / "missing.json")

    @pytest.fixture
    def unavailable_device(self) -> DeviceKind:
        return DeviceKind.CUDA


class TestListDatasetContract(DatasetAdapterContract):
    expected_dataset_capabilities = frozenset(
        {
            DatasetCapability.RANDOM_ACCESS,
            DatasetCapability.ITERATION,
            DatasetCapability.BATCHING,
            DatasetCapability.LABELS,
            DatasetCapability.TEST_SPLIT,
        }
    )
    payload: ClassVar[dict[str, object]] = {
        "rows": [[float(i), float(i * 2)] for i in range(23)],
        "targets": list(range(23)),
        "splits": {"test": [20, 21, 22, 3, 5]},
    }

    @pytest.fixture
    def source(self, tmp_path: Path) -> str:
        return _write(tmp_path / "d.json", self.payload)

    @pytest.fixture
    def dataset(self, source: str) -> ListDatasetAdapter:
        return ListDatasetAdapter.load(source, version="1", options={})

    @pytest.fixture
    def reloaded_dataset(self, source: str) -> ListDatasetAdapter:
        return ListDatasetAdapter.load(source, version="1", options={})

    @pytest.fixture
    def different_dataset(self, tmp_path: Path) -> ListDatasetAdapter:
        other = {**self.payload, "targets": [*range(22), 99]}
        return ListDatasetAdapter.load(_write(tmp_path / "e.json", other), version="1", options={})

    @pytest.fixture
    def missing_dataset_source(self, tmp_path: Path) -> str:
        return str(tmp_path / "missing.json")
