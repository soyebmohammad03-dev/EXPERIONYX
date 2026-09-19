import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("numpy")
import numpy as np
import torch

from contracts import DatasetAdapterContract, ModelAdapterContract
from experionyx.adapters.capabilities import (
    DatasetCapability,
    DeviceKind,
    ModelCapability,
    TaskType,
)
from experionyx.adapters.torch_adapter import (
    TorchDatasetAdapter,
    TorchModelAdapter,
    resolve_device,
)
from experionyx.errors import (
    DatasetLoadError,
    DeviceUnavailableError,
    InferenceError,
    InvalidDatasetError,
    InvalidModelError,
    ModelLoadError,
    UnsupportedCapabilityError,
)


class Weird:
    """Module-level so torch.save can pickle it; weights_only loading must refuse it."""


def net(seed: int = 0, dropout: bool = False) -> torch.nn.Module:
    torch.manual_seed(seed)
    layers: list[torch.nn.Module] = [torch.nn.Linear(4, 3)]
    if dropout:
        layers.append(torch.nn.Dropout(0.5))
    layers += [torch.nn.ReLU(), torch.nn.Linear(3, 2)]
    return torch.nn.Sequential(*layers)


def save(module: torch.nn.Module, path: Path) -> Path:
    return TorchModelAdapter.save(module, path, torch.zeros(1, 4))


def load(path: Path | str, **options: object) -> TorchModelAdapter:
    return TorchModelAdapter.load(path, version="1", device=DeviceKind.CPU, options=options)


def unavailable_device() -> DeviceKind | None:
    if not torch.cuda.is_available():
        return DeviceKind.CUDA
    if not torch.backends.mps.is_available():
        return DeviceKind.MPS
    return None


class TestTorchModelContract(ModelAdapterContract):
    expected_capabilities = frozenset({ModelCapability.PREDICT, ModelCapability.BATCH_PREDICT})

    @pytest.fixture
    def model_source(self, tmp_path: Path) -> str:
        return str(save(net(0), tmp_path / "a.pt"))

    @pytest.fixture
    def adapter(self, model_source: str) -> TorchModelAdapter:
        return load(model_source, input_shape=[4])

    @pytest.fixture
    def reloaded(self, model_source: str) -> TorchModelAdapter:
        return load(model_source, input_shape=[4])

    @pytest.fixture
    def different(self, tmp_path: Path) -> TorchModelAdapter:
        return load(save(net(1), tmp_path / "b.pt"), input_shape=[4])

    @pytest.fixture
    def valid_inputs(self) -> object:
        return torch.linspace(-1, 1, 28).reshape(7, 4)

    @pytest.fixture
    def invalid_inputs(self) -> list[object]:
        return [torch.zeros(0, 4), torch.zeros(4), torch.zeros(3, 5), "text", [[1, 2, 3]], object()]

    @pytest.fixture
    def missing_source(self, tmp_path: Path) -> str:
        return str(tmp_path / "missing.pt")

    @pytest.fixture
    def unavailable_device(self) -> DeviceKind | None:
        return unavailable_device()


# --- model behaviour --------------------------------------------------------------------------


def test_outputs_match_a_direct_forward_pass_and_are_plain_python(tmp_path: Path) -> None:
    module = net(0)
    adapter = load(save(module, tmp_path / "m.pt"))
    x = torch.randn(9, 4, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        expected = module(x)
    result = adapter.predict(x)
    assert np.allclose(np.array(result.outputs), expected.numpy(), atol=1e-6)
    assert all(type(v) is float for row in result.outputs for v in row)  # type: ignore[attr-defined]
    assert adapter.batch_predict(x, 4).outputs == result.outputs  # batching does not change values
    assert result.device.resolved is DeviceKind.CPU
    with pytest.raises(UnsupportedCapabilityError):
        adapter.predict_proba(x)  # softmax is not assumed


def test_inference_runs_in_eval_mode_without_gradients(tmp_path: Path) -> None:
    adapter = load(save(net(0, dropout=True), tmp_path / "d.pt"))
    x = torch.ones(5, 4)
    first, second = adapter.predict(x).outputs, adapter.predict(x).outputs
    assert first == second  # dropout is inactive: the module is in eval mode
    torch.set_grad_enabled(True)
    adapter.predict(x.requires_grad_())
    assert torch.is_grad_enabled()  # no_grad scope did not leak


def test_in_memory_modules_are_wrapped_and_put_in_eval_mode() -> None:
    module = net(0, dropout=True).train()
    adapter = TorchModelAdapter(module, version="1")
    assert not module.training
    assert adapter.metadata().serialization_format is None
    assert adapter.metadata().size_bytes is None


def test_input_conversion_and_shape_validation(tmp_path: Path) -> None:
    adapter = load(save(net(0), tmp_path / "m.pt"), input_shape=[4])
    as_float64 = np.random.default_rng(0).normal(size=(3, 4))
    assert adapter.predict(as_float64).sample_count == 3  # float64 numpy is cast to the model dtype
    assert adapter.predict([[0.1, 0.2, 0.3, 0.4]]).sample_count == 1
    with pytest.raises(InferenceError, match="per-sample shape"):
        adapter.predict(torch.zeros(2, 5))
    with pytest.raises(InferenceError) as exc:
        load(save(net(0), tmp_path / "n.pt")).predict(torch.zeros(2, 5))
    assert exc.value.__cause__ is not None  # the torch error is kept as the cause


def test_metadata_counts_parameters_only_when_measurable(tmp_path: Path) -> None:
    module = net(0)
    adapter = load(save(module, tmp_path / "m.pt"), task="CLASSIFICATION", input_shape=[4])
    meta = adapter.metadata()
    assert meta.parameter_count == 4 * 3 + 3 + 3 * 2 + 2
    assert meta.trainable_parameter_count == meta.parameter_count
    assert meta.task is TaskType.CLASSIFICATION
    assert meta.serialization_format == "torchscript"
    assert meta.size_bytes == (tmp_path / "m.pt").stat().st_size
    assert meta.input_schema is not None
    assert meta.input_schema.shape == (None, 4)
    assert meta.input_schema.dtype == "float32"
    assert meta.output_schema is None  # not inferable, not invented
    assert meta.framework == f"pytorch {torch.__version__}"

    frozen = net(0)
    for p in frozen[0].parameters():
        p.requires_grad_(False)
    counted = TorchModelAdapter(frozen, version="1").metadata()
    assert counted.parameter_count == 23
    assert counted.trainable_parameter_count == 23 - (4 * 3 + 3)


def test_default_task_is_unknown_and_bad_options_fail(tmp_path: Path) -> None:
    path = save(net(0), tmp_path / "m.pt")
    assert load(path).metadata().task is TaskType.UNKNOWN
    with pytest.raises(ModelLoadError, match="unknown task"):
        load(path, task="TELEPATHY")
    with pytest.raises(ModelLoadError, match="input_shape"):
        load(path, input_shape="four")


# --- fingerprints -----------------------------------------------------------------------------


def independent_fingerprint(module: torch.nn.Module) -> str:
    """The documented algorithm, re-derived with numpy instead of the adapter's helpers."""

    def canon(x: object) -> bytes:
        return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()

    h = hashlib.sha256()
    arch = [[n, getattr(m, "original_name", type(m).__name__)] for n, m in module.named_modules()]
    h.update(canon({"kind": "torch-module", "modules": arch}))
    for name, t in sorted(module.state_dict().items()):
        h.update(canon([name, str(t.dtype).removeprefix("torch."), list(t.shape)]))
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return "sha256:" + h.hexdigest()


def test_fingerprint_matches_an_independent_computation(tmp_path: Path) -> None:
    module = net(0)
    assert load(save(module, tmp_path / "m.pt")).fingerprint() == independent_fingerprint(
        load(tmp_path / "m.pt")._module  # the loaded TorchScript module
    )
    in_memory = TorchModelAdapter(net(3), version="1")
    assert in_memory.fingerprint() == independent_fingerprint(net(3))


def test_fingerprint_depends_on_weights_not_on_file_bytes(tmp_path: Path) -> None:
    module = net(0)
    a = load(save(module, tmp_path / "a.pt"))
    b = load(save(module, tmp_path / "b.pt"))  # saved again: file bytes may differ
    assert a.fingerprint() == b.fingerprint()
    assert a.fingerprint() == TorchModelAdapter(module, version="9").fingerprint()  # even in memory


def test_fingerprint_changes_with_a_one_ulp_weight_change_and_with_architecture() -> None:
    base = net(0)
    original = TorchModelAdapter(net(0), version="1").fingerprint()
    perturbed = net(0)
    with torch.no_grad():
        w = perturbed[0].weight
        w[0, 0] = torch.nextafter(w[0, 0], torch.tensor(10.0))
    assert TorchModelAdapter(perturbed, version="1").fingerprint() != original
    wrapped = torch.nn.Sequential(base, torch.nn.Identity())  # same weights, different structure
    assert TorchModelAdapter(wrapped, version="1").fingerprint() != original
    assert TorchModelAdapter(net(1), version="1").fingerprint() != original


def test_fingerprint_covers_buffers_and_dtypes() -> None:
    with_bn = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.BatchNorm1d(3)).eval()
    before = TorchModelAdapter(with_bn, version="1").fingerprint()
    with_bn[1].running_mean.add_(1.0)  # a buffer, not a parameter
    assert TorchModelAdapter(with_bn, version="1").fingerprint() != before
    half = net(0).double()
    assert (
        TorchModelAdapter(half, version="1").fingerprint()
        != TorchModelAdapter(net(0), version="1").fingerprint()
    )


# --- errors and devices -----------------------------------------------------------------------


def test_bad_artifacts_and_types_fail_with_typed_errors(tmp_path: Path) -> None:
    garbage = tmp_path / "garbage.pt"
    garbage.write_bytes(b"not a torchscript archive")
    with pytest.raises(ModelLoadError):
        load(garbage)
    with pytest.raises(InvalidModelError):
        TorchModelAdapter("not a module", version="1")  # type: ignore[arg-type]


def test_cpu_is_the_default_and_devices_resolve_without_fallback() -> None:
    assert resolve_device(DeviceKind.CPU).resolved is DeviceKind.CPU
    if not torch.cuda.is_available():
        with pytest.raises(DeviceUnavailableError, match="CUDA"):
            resolve_device(DeviceKind.CUDA)
    if not torch.backends.mps.is_available():
        with pytest.raises(DeviceUnavailableError, match="MPS"):
            resolve_device(DeviceKind.MPS)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS is optional and not required"
)
def test_mps_works_only_when_explicitly_requested_and_available(tmp_path: Path) -> None:
    path = save(net(0), tmp_path / "m.pt")
    adapter = TorchModelAdapter.load(path, version="1", device=DeviceKind.MPS, options={})
    assert adapter.device.resolved is DeviceKind.MPS
    assert adapter.predict(torch.ones(2, 4)).device.resolved is DeviceKind.MPS
    assert load(path).device.resolved is DeviceKind.CPU  # the default never picks an accelerator


# --- dataset adapter --------------------------------------------------------------------------


def write_tensors(
    path: Path, *, n: int = 23, seed: int = 0, labeled: bool = True, dtype: object = torch.float32
) -> Path:
    gen = torch.Generator().manual_seed(seed)
    payload: dict[str, object] = {
        "X": torch.randn(n, 4, generator=gen).to(dtype),
        "split_test": torch.tensor([20, 21, 22, 3, 5]),
    }
    if labeled:
        payload["y"] = torch.arange(n) % 3
    torch.save(payload, path)
    return path


class TestTorchDatasetContract(DatasetAdapterContract):
    expected_dataset_capabilities = frozenset(
        {
            DatasetCapability.RANDOM_ACCESS,
            DatasetCapability.ITERATION,
            DatasetCapability.BATCHING,
            DatasetCapability.LABELS,
            DatasetCapability.TEST_SPLIT,
        }
    )

    @pytest.fixture
    def source(self, tmp_path: Path) -> str:
        return str(write_tensors(tmp_path / "d.pt"))

    @pytest.fixture
    def dataset(self, source: str) -> TorchDatasetAdapter:
        return TorchDatasetAdapter.load(source, version="1", options={})

    @pytest.fixture
    def reloaded_dataset(self, source: str) -> TorchDatasetAdapter:
        return TorchDatasetAdapter.load(source, version="1", options={})

    @pytest.fixture
    def different_dataset(self, tmp_path: Path) -> TorchDatasetAdapter:
        return TorchDatasetAdapter.load(
            write_tensors(tmp_path / "e.pt", seed=1), version="1", options={}
        )

    @pytest.fixture
    def missing_dataset_source(self, tmp_path: Path) -> str:
        return str(tmp_path / "missing.pt")


def test_dataset_metadata_and_lazy_access(tmp_path: Path) -> None:
    d = TorchDatasetAdapter.load(
        write_tensors(tmp_path / "d.pt"), version="1", options={"task": "CLASSIFICATION"}
    )
    meta = d.metadata()
    assert (meta.num_samples, meta.num_features) == (23, 4)
    assert meta.family == "torch-map-style"
    assert meta.task is TaskType.CLASSIFICATION
    assert meta.input_schema is not None
    assert meta.input_schema.dtype == "float32"
    assert meta.input_schema.shape == (None, 4)
    assert meta.target_schema is not None
    assert meta.target_schema.dtype == "int64"
    assert meta.class_count is None  # needs a full scan: opt-in only
    deep = d.metadata(deep=True)
    assert deep.class_count == 3
    assert deep.missing_values == 0
    assert meta.source["format"] == "torch-tensors"


def test_batches_are_stacked_tensors_in_order(tmp_path: Path) -> None:
    d = TorchDatasetAdapter.load(write_tensors(tmp_path / "d.pt"), version="1", options={})
    batches = list(d.batches(10))
    assert [len(b.indices) for b in batches] == [10, 10, 3]
    assert batches[0].inputs.shape == (10, 4)  # type: ignore[attr-defined]
    assert batches[0].target.tolist() == [i % 3 for i in range(10)]  # type: ignore[attr-defined]
    sample = d.sample(2)
    assert sample.index == 2
    assert torch.equal(sample.inputs, batches[0].inputs[2])  # type: ignore[index]
    test_batches = list(d.batches(2, "test"))
    assert [i for b in test_batches for i in b.indices] == [20, 21, 22, 3, 5]  # split order kept


def test_unlabeled_and_wrapped_datasets(tmp_path: Path) -> None:
    d = TorchDatasetAdapter.load(
        write_tensors(tmp_path / "u.pt", labeled=False), version="1", options={}
    )
    assert DatasetCapability.LABELS not in d.capabilities
    assert next(iter(d.batches(4))).target is None
    wrapped = TorchDatasetAdapter(
        torch.utils.data.TensorDataset(torch.arange(6.0).reshape(3, 2)), version="1"
    )
    assert wrapped.num_samples() == 3
    assert wrapped.metadata().num_features == 2


def test_dataset_fingerprint_matches_an_independent_computation(tmp_path: Path) -> None:
    path = write_tensors(tmp_path / "d.pt")
    data = torch.load(path, weights_only=True)
    header = {
        "kind": "torch-map",
        "length": 23,
        "labeled": True,
        "splits": {"test": [20, 21, 22, 3, 5]},
    }

    def canon(x: object) -> bytes:
        return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()

    h = hashlib.sha256(canon(header))
    for i in range(23):
        for part in (data["X"][i], data["y"][i]):
            h.update(canon([str(part.dtype).removeprefix("torch."), list(part.shape)]))
            h.update(part.numpy().tobytes())
    d = TorchDatasetAdapter.load(path, version="1", options={})
    assert d.fingerprint() == "sha256:" + h.hexdigest()


def test_dataset_fingerprint_is_sensitive_to_values_order_dtype_and_splits(tmp_path: Path) -> None:
    def fp(**kw: object) -> str:
        return TorchDatasetAdapter.load(
            write_tensors(tmp_path / "x.pt", **kw), version="1", options={}
        ).fingerprint()  # type: ignore[arg-type]

    base = fp()
    assert fp() == base
    assert fp(seed=1) != base
    assert fp(dtype=torch.float64) != base
    assert fp(n=24) != base
    assert fp(labeled=False) != base
    x = torch.randn(6, 2)
    same = TorchDatasetAdapter(torch.utils.data.TensorDataset(x), version="1").fingerprint()
    assert (
        TorchDatasetAdapter(torch.utils.data.TensorDataset(x.flip(0)), version="1").fingerprint()
        != same
    )
    assert (
        TorchDatasetAdapter(
            torch.utils.data.TensorDataset(x), version="1", splits={"test": [0, 1]}
        ).fingerprint()
        != same
    )


def test_invalid_and_unsafe_datasets_are_rejected(tmp_path: Path) -> None:
    unsafe = tmp_path / "unsafe.pt"
    torch.save({"X": torch.zeros(2, 2), "obj": Weird()}, unsafe)
    with pytest.raises(DatasetLoadError):  # weights_only=True refuses arbitrary objects
        TorchDatasetAdapter.load(unsafe, version="1", options={})
    no_x = tmp_path / "nox.pt"
    torch.save({"y": torch.zeros(2)}, no_x)
    with pytest.raises(InvalidDatasetError, match="'X'"):
        TorchDatasetAdapter.load(no_x, version="1", options={})
    mismatch = tmp_path / "mismatch.pt"
    torch.save({"X": torch.zeros(3, 2), "y": torch.zeros(4)}, mismatch)
    with pytest.raises(InvalidDatasetError):
        TorchDatasetAdapter.load(mismatch, version="1", options={})
    with pytest.raises(InvalidDatasetError):
        TorchDatasetAdapter([], version="1")  # type: ignore[arg-type]
    with pytest.raises(InvalidDatasetError):
        TorchDatasetAdapter(
            torch.utils.data.TensorDataset(torch.zeros(3, 2)), version="1", splits={"test": [5]}
        )
