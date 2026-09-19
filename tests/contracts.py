"""Reusable adapter contract suites. A concrete test class inherits `ModelAdapterContract` or
`DatasetAdapterContract`, supplies the fixtures below, and gets the whole quality bar.

Fixtures for models:   adapter, reloaded, different, valid_inputs, invalid_inputs,
                       missing_source, unavailable_device; class attr expected_capabilities.
Fixtures for datasets: dataset, reloaded_dataset, different_dataset, missing_dataset_source;
                       class attr expected_dataset_capabilities.
"""

import math
from collections.abc import Callable, Sequence

import pytest

from experionyx.adapters.base import DatasetAdapter, InferenceResult, ModelAdapter
from experionyx.adapters.capabilities import DatasetCapability, DeviceKind, ModelCapability
from experionyx.adapters.metadata import DatasetMetadata, ModelMetadata
from experionyx.domain import to_jsonable
from experionyx.errors import (
    DatasetLoadError,
    DeviceUnavailableError,
    InferenceError,
    InvalidDatasetError,
    ModelLoadError,
    UnsupportedCapabilityError,
)

_OPERATIONS: dict[ModelCapability, Callable[[ModelAdapter, object], InferenceResult]] = {
    ModelCapability.PREDICT: lambda a, x: a.predict(x),
    ModelCapability.BATCH_PREDICT: lambda a, x: a.batch_predict(x, 2),
    ModelCapability.PREDICT_PROBA: lambda a, x: a.predict_proba(x),
}


def outputs_match(a: object, b: object) -> bool:
    """Exact for ints/strings/structure; floats within rounding. Batch size may legitimately
    change the last bits of a float result (BLAS blocking differs by platform), so this contract
    demands agreement to ~1e-9, not bit equality."""
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    if isinstance(a, tuple) and isinstance(b, tuple):
        return len(a) == len(b) and all(outputs_match(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def _n(inputs: object) -> int:
    return len(inputs)  # type: ignore[arg-type]


class ModelAdapterContract:
    expected_capabilities: frozenset[ModelCapability]

    # -- identity and metadata ----------------------------------------------------------------

    def test_identity_is_declared(self, adapter: ModelAdapter) -> None:
        cls = type(adapter)
        assert cls.NAME
        assert cls.VERSION
        assert cls.FRAMEWORK
        info = cls.info()
        assert (info.kind, info.name, info.version) == ("model", cls.NAME, cls.VERSION)
        assert set(info.capabilities) >= {c.value for c in adapter.capabilities}

    def test_metadata_is_consistent_and_serializable(self, adapter: ModelAdapter) -> None:
        meta = adapter.metadata()
        assert (meta.adapter, meta.adapter_version) == (type(adapter).NAME, type(adapter).VERSION)
        assert meta.fingerprint == adapter.fingerprint()
        assert set(meta.capabilities) == set(adapter.capabilities)
        assert ModelMetadata.from_dict(to_jsonable(meta)) == meta  # type: ignore[arg-type]

    def test_optional_metadata_is_unavailable_not_zero(self, adapter: ModelAdapter) -> None:
        meta = adapter.metadata()
        for value in (meta.parameter_count, meta.trainable_parameter_count, meta.size_bytes):
            assert value is None or value > 0

    # -- capabilities -------------------------------------------------------------------------

    def test_capabilities_match_expectation(self, adapter: ModelAdapter) -> None:
        assert adapter.capabilities == self.expected_capabilities
        assert adapter.capabilities <= type(adapter).POSSIBLE_CAPABILITIES  # type: ignore[attr-defined]

    def test_unsupported_operations_raise_typed_errors(
        self, adapter: ModelAdapter, valid_inputs: object
    ) -> None:
        for capability, call in _OPERATIONS.items():
            if capability not in adapter.capabilities:
                with pytest.raises(UnsupportedCapabilityError):
                    call(adapter, valid_inputs)

    # -- inference ----------------------------------------------------------------------------

    def test_predict_returns_a_complete_result(
        self, adapter: ModelAdapter, valid_inputs: object
    ) -> None:
        ids = list(range(100, 100 + _n(valid_inputs)))
        result = adapter.predict(valid_inputs, sample_ids=ids)
        assert result.capability is ModelCapability.PREDICT
        assert result.sample_count == len(result.outputs) == _n(valid_inputs)
        assert result.sample_ids == tuple(ids)
        assert result.model_fingerprint == adapter.fingerprint()
        assert (result.adapter, result.adapter_version) == (
            type(adapter).NAME,
            type(adapter).VERSION,
        )
        assert result.device == adapter.device
        assert result.batch_count == 1
        assert result.batch_size is None
        assert math.isfinite(result.inference_seconds)
        assert result.inference_seconds >= 0
        assert adapter.load_seconds >= 0

    def test_repeated_predictions_are_identical(
        self, adapter: ModelAdapter, valid_inputs: object
    ) -> None:
        assert adapter.predict(valid_inputs).outputs == adapter.predict(valid_inputs).outputs

    @pytest.mark.parametrize("batch_size", [1, 2, 3, 1000])
    def test_batching_covers_all_samples_including_the_partial_batch(
        self, adapter: ModelAdapter, valid_inputs: object, batch_size: int
    ) -> None:
        n = _n(valid_inputs)
        result = adapter.batch_predict(valid_inputs, batch_size)
        assert result.capability is ModelCapability.BATCH_PREDICT
        assert result.batch_size == batch_size
        assert result.batch_count == math.ceil(n / batch_size)
        assert len(result.batch_seconds) == result.batch_count
        assert outputs_match(result.outputs, adapter.predict(valid_inputs).outputs)
        assert result.sample_count == n

    @pytest.mark.parametrize("bad", [0, -1, True, 1.5])
    def test_invalid_batch_sizes_are_rejected(
        self, adapter: ModelAdapter, valid_inputs: object, bad: object
    ) -> None:
        with pytest.raises(InferenceError):
            adapter.batch_predict(valid_inputs, bad)  # type: ignore[arg-type]

    def test_invalid_inputs_raise_inference_errors(
        self, adapter: ModelAdapter, invalid_inputs: Sequence[object]
    ) -> None:
        for bad in invalid_inputs:
            with pytest.raises(InferenceError):
                adapter.predict(bad)

    def test_sample_id_count_must_match(self, adapter: ModelAdapter, valid_inputs: object) -> None:
        with pytest.raises(InferenceError):
            adapter.predict(valid_inputs, sample_ids=[1])

    # -- fingerprints -------------------------------------------------------------------------

    def test_fingerprint_is_stable_across_calls_and_reloads(
        self, adapter: ModelAdapter, reloaded: ModelAdapter
    ) -> None:
        assert adapter.fingerprint() == adapter.fingerprint()
        assert adapter.fingerprint() == reloaded.fingerprint()
        assert adapter.fingerprint().startswith("sha256:")

    def test_different_models_have_different_fingerprints(
        self, adapter: ModelAdapter, different: ModelAdapter
    ) -> None:
        assert adapter.fingerprint() != different.fingerprint()

    # -- loading and devices ------------------------------------------------------------------

    def test_loading_a_missing_artifact_fails_with_a_typed_error(
        self, adapter: ModelAdapter, missing_source: str
    ) -> None:
        with pytest.raises(ModelLoadError):
            type(adapter).load(missing_source, version="1", device=DeviceKind.CPU, options={})

    def test_default_device_is_cpu(self, adapter: ModelAdapter) -> None:
        assert adapter.device.requested is DeviceKind.CPU
        assert adapter.device.resolved is DeviceKind.CPU

    def test_unavailable_devices_are_refused_not_substituted(
        self, adapter: ModelAdapter, unavailable_device: DeviceKind | None, model_source: str
    ) -> None:
        if unavailable_device is None:
            pytest.skip("every device is available here")
        with pytest.raises(DeviceUnavailableError):
            type(adapter).load(model_source, version="1", device=unavailable_device, options={})


class DatasetAdapterContract:
    expected_dataset_capabilities: frozenset[DatasetCapability]
    split_name: str = "test"

    def test_identity_is_declared(self, dataset: DatasetAdapter) -> None:
        cls = type(dataset)
        info = cls.info()
        assert (info.kind, info.name, info.version) == ("dataset", cls.NAME, cls.VERSION)

    def test_metadata_is_consistent_and_serializable(self, dataset: DatasetAdapter) -> None:
        meta = dataset.metadata()
        assert meta.num_samples == dataset.num_samples()
        assert meta.fingerprint == dataset.fingerprint()
        assert (meta.adapter, meta.adapter_version) == (type(dataset).NAME, type(dataset).VERSION)
        assert set(meta.capabilities) == set(dataset.capabilities)
        assert DatasetMetadata.from_dict(to_jsonable(meta)) == meta  # type: ignore[arg-type]
        assert dict(meta.splits) == {s: dataset.num_samples(s) for s in dataset.splits()}

    def test_expensive_statistics_are_opt_in(self, dataset: DatasetAdapter) -> None:
        assert dataset.metadata().missing_values is None
        deep = dataset.metadata(deep=True).missing_values
        assert deep is not None
        assert deep >= 0

    def test_capabilities_match_expectation(self, dataset: DatasetAdapter) -> None:
        assert dataset.capabilities == self.expected_dataset_capabilities

    @pytest.mark.parametrize("batch_size", [1, 4, 7, 10_000])
    def test_batches_cover_every_sample_once_in_order(
        self, dataset: DatasetAdapter, batch_size: int
    ) -> None:
        n = dataset.num_samples()
        seen: list[int] = []
        sizes: list[int] = []
        for batch in dataset.batches(batch_size):
            seen.extend(batch.indices)
            sizes.append(len(batch.indices))
        assert seen == list(range(n))
        assert all(s == batch_size for s in sizes[:-1])
        assert 1 <= sizes[-1] <= batch_size
        assert len(sizes) == math.ceil(n / batch_size)

    def test_split_batches_stay_inside_the_split(self, dataset: DatasetAdapter) -> None:
        if self.split_name not in dataset.splits():
            pytest.skip("no such split")
        indices = [i for b in dataset.batches(3, self.split_name) for i in b.indices]
        assert len(indices) == dataset.num_samples(self.split_name)
        assert len(set(indices)) == len(indices)

    def test_sample_access_agrees_with_batches(self, dataset: DatasetAdapter) -> None:
        first = next(iter(dataset.batches(1)))
        sample = dataset.sample(0)
        assert sample.index == first.indices[0]

    @pytest.mark.parametrize("bad", [-1, 10**9, True, "0"])
    def test_out_of_range_samples_raise(self, dataset: DatasetAdapter, bad: object) -> None:
        with pytest.raises(InvalidDatasetError):
            dataset.sample(bad)  # type: ignore[arg-type]

    def test_unknown_split_raises(self, dataset: DatasetAdapter) -> None:
        with pytest.raises(InvalidDatasetError):
            dataset.num_samples("no-such-split")
        with pytest.raises(InvalidDatasetError):
            next(iter(dataset.batches(2, "no-such-split")))

    @pytest.mark.parametrize("bad", [0, -3])
    def test_invalid_batch_size_raises(self, dataset: DatasetAdapter, bad: int) -> None:
        with pytest.raises(InferenceError):
            next(iter(dataset.batches(bad)))

    def test_fingerprint_is_stable_and_discriminating(
        self,
        dataset: DatasetAdapter,
        reloaded_dataset: DatasetAdapter,
        different_dataset: DatasetAdapter,
    ) -> None:
        assert dataset.fingerprint() == dataset.fingerprint()
        assert dataset.fingerprint() == reloaded_dataset.fingerprint()
        assert dataset.fingerprint() != different_dataset.fingerprint()
        assert dataset.fingerprint().startswith("sha256:")

    def test_loading_a_missing_source_fails_with_a_typed_error(
        self, dataset: DatasetAdapter, missing_dataset_source: str
    ) -> None:
        with pytest.raises(DatasetLoadError):
            type(dataset).load(missing_dataset_source, version="1", options={})
