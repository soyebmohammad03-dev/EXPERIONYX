import hashlib
import json
import shutil
from collections.abc import Mapping
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("sklearn")
import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from contracts import DatasetAdapterContract, ModelAdapterContract
from experionyx.adapters.capabilities import (
    DatasetCapability,
    DeviceKind,
    ModelCapability,
    TaskType,
)
from experionyx.adapters.sklearn_adapter import (
    SklearnDatasetAdapter,
    SklearnModelAdapter,
)
from experionyx.errors import (
    DatasetLoadError,
    InferenceError,
    InvalidDatasetError,
    InvalidModelError,
    ModelLoadError,
)

IRIS = load_iris()
X, Y = IRIS.data, IRIS.target


def load(path: Path | str) -> SklearnModelAdapter:
    return SklearnModelAdapter.load(path, version="1", device=DeviceKind.CPU, options={})


def saved(estimator: object, path: Path) -> Path:
    return SklearnModelAdapter.save(estimator, path)


class ClassifierFixtures:
    @pytest.fixture
    def model_source(self, tmp_path: Path) -> str:
        return str(saved(LogisticRegression(max_iter=300).fit(X, Y), tmp_path / "clf.joblib"))

    @pytest.fixture
    def adapter(self, model_source: str) -> SklearnModelAdapter:
        return load(model_source)

    @pytest.fixture
    def reloaded(self, model_source: str) -> SklearnModelAdapter:
        return load(model_source)

    @pytest.fixture
    def different(self, tmp_path: Path) -> SklearnModelAdapter:
        other = LogisticRegression(max_iter=300, C=0.01).fit(X, Y)
        return load(saved(other, tmp_path / "other.joblib"))

    @pytest.fixture
    def valid_inputs(self) -> object:
        return X[:7]

    @pytest.fixture
    def invalid_inputs(self) -> list[object]:
        return [
            np.zeros((0, 4)),
            np.zeros(4),
            np.zeros((3, 5)),
            np.array([["a", "b", "c", "d"]]),
            "text",
            np.zeros((2, 2, 4)),
        ]

    @pytest.fixture
    def missing_source(self, tmp_path: Path) -> str:
        return str(tmp_path / "missing.joblib")

    @pytest.fixture
    def unavailable_device(self) -> DeviceKind:
        return DeviceKind.MPS  # the sklearn adapter is CPU only


class TestSklearnClassifierContract(ClassifierFixtures, ModelAdapterContract):
    expected_capabilities = frozenset(
        {ModelCapability.PREDICT, ModelCapability.BATCH_PREDICT, ModelCapability.PREDICT_PROBA}
    )


class TestSklearnRegressorContract(ClassifierFixtures, ModelAdapterContract):
    expected_capabilities = frozenset({ModelCapability.PREDICT, ModelCapability.BATCH_PREDICT})

    @pytest.fixture
    def model_source(self, tmp_path: Path) -> str:
        return str(saved(Ridge().fit(X, X[:, 0] * 2.0), tmp_path / "reg.joblib"))

    @pytest.fixture
    def different(self, tmp_path: Path) -> SklearnModelAdapter:
        return load(saved(Ridge(alpha=50.0).fit(X, X[:, 0] * 2.0), tmp_path / "reg2.joblib"))


# --- sklearn-specific behaviour ---------------------------------------------------------------


def test_predictions_match_the_estimator_and_are_plain_python(tmp_path: Path) -> None:
    est = LogisticRegression(max_iter=300).fit(X, Y)
    adapter = load(saved(est, tmp_path / "m.joblib"))
    result = adapter.predict(X[:20])
    assert list(result.outputs) == est.predict(X[:20]).tolist()
    assert all(type(o) is int for o in result.outputs)  # no numpy scalars leak out
    proba = adapter.predict_proba(X[:20])
    assert np.allclose(np.array(proba.outputs), est.predict_proba(X[:20]))
    assert np.allclose(np.array(proba.outputs).sum(axis=1), 1.0)
    assert proba.capability is ModelCapability.PREDICT_PROBA
    assert adapter.batch_predict(X[:20], 6).outputs == result.outputs  # integer labels: exact


def test_capabilities_reflect_what_the_estimator_really_supports(tmp_path: Path) -> None:
    no_proba = load(saved(SVC().fit(X, Y), tmp_path / "a.joblib"))
    with_proba = load(saved(GaussianNB().fit(X, Y), tmp_path / "b.joblib"))
    assert ModelCapability.PREDICT_PROBA not in no_proba.capabilities
    assert ModelCapability.PREDICT_PROBA in with_proba.capabilities
    from experionyx.errors import UnsupportedCapabilityError

    with pytest.raises(UnsupportedCapabilityError):
        no_proba.predict_proba(X[:2])
    with_proba.predict_proba(X[:2])


def test_classifier_and_regressor_metadata_are_measured_not_invented(tmp_path: Path) -> None:
    path = saved(LogisticRegression(max_iter=300).fit(X, Y), tmp_path / "c.joblib")
    meta = load(path).metadata()
    assert meta.task is TaskType.CLASSIFICATION
    assert meta.model_type == "sklearn.linear_model._logistic.LogisticRegression"
    assert meta.family == "linear_model"
    assert meta.input_schema is not None
    assert meta.input_schema.shape == (None, 4)
    assert meta.output_schema is not None
    assert meta.output_schema.class_labels == (0, 1, 2)
    assert meta.serialization_format == "joblib"
    assert meta.size_bytes == path.stat().st_size
    assert meta.parameter_count is None
    assert meta.trainable_parameter_count is None
    assert meta.framework.startswith("scikit-learn ")

    reg = load(saved(Ridge().fit(X, Y.astype(float)), tmp_path / "r.joblib")).metadata()
    assert reg.task is TaskType.REGRESSION
    assert reg.output_schema is not None
    assert reg.output_schema.class_labels is None
    assert reg.capabilities == (ModelCapability.BATCH_PREDICT, ModelCapability.PREDICT)


def test_feature_names_are_reported_when_the_estimator_knows_them(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")
    est = LogisticRegression(max_iter=300).fit(pd.DataFrame(X, columns=list("abcd")), Y)
    schema = load(saved(est, tmp_path / "n.joblib")).metadata().input_schema
    assert schema is not None
    assert schema.feature_names == ("a", "b", "c", "d")


def test_pipelines_are_supported(tmp_path: Path) -> None:
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300)).fit(X, Y)
    adapter = load(saved(pipe, tmp_path / "p.joblib"))
    assert list(adapter.predict(X[:10]).outputs) == pipe.predict(X[:10]).tolist()
    assert adapter.metadata().input_schema.shape == (None, 4)  # type: ignore[union-attr]


def test_fingerprint_is_the_sha256_of_the_file_bytes_and_tracks_content(tmp_path: Path) -> None:
    path = saved(LogisticRegression(max_iter=300).fit(X, Y), tmp_path / "m.joblib")
    assert load(path).fingerprint() == "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    copy = tmp_path / "copy.joblib"
    shutil.copy(path, copy)
    assert load(copy).fingerprint() == load(path).fingerprint()  # same bytes, other path
    retrained = saved(LogisticRegression(max_iter=300, C=5.0).fit(X, Y), tmp_path / "m2.joblib")
    assert load(retrained).fingerprint() != load(path).fingerprint()
    original = load(path).fingerprint()
    path.write_bytes(
        path.read_bytes() + b"\0"
    )  # trailing bytes: still loadable, different artifact
    assert load(path).fingerprint() != original
    assert load(path).fingerprint() == "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_bad_artifacts_fail_with_typed_errors(tmp_path: Path) -> None:
    garbage = tmp_path / "garbage.joblib"
    garbage.write_bytes(b"this is not a joblib file")
    with pytest.raises(ModelLoadError):
        load(garbage)
    not_estimator = tmp_path / "dict.joblib"
    joblib.dump({"a": 1}, not_estimator)
    with pytest.raises(InvalidModelError, match="not a scikit-learn estimator"):
        load(not_estimator)
    unfitted = tmp_path / "unfitted.joblib"
    joblib.dump(LogisticRegression(), unfitted)
    with pytest.raises(InvalidModelError, match="not fitted"):
        load(unfitted)
    with pytest.raises(ValueError, match="version"):
        load_bad_version(garbage)


def load_bad_version(path: Path) -> SklearnModelAdapter:
    return SklearnModelAdapter.load(
        path, version="not a version", device=DeviceKind.CPU, options={}
    )


def test_inference_errors_wrap_framework_failures(tmp_path: Path) -> None:
    adapter = load(saved(LogisticRegression(max_iter=300).fit(X, Y), tmp_path / "m.joblib"))
    with pytest.raises(InferenceError, match="4 features"):
        adapter.predict(np.zeros((2, 3)))
    bad = np.array([[np.nan] * 4])
    with pytest.raises(InferenceError) as exc:
        adapter.predict(bad)
    assert exc.value.__cause__ is not None  # the sklearn error is preserved as the cause


def test_load_and_inference_times_are_reported_separately(tmp_path: Path) -> None:
    adapter = load(saved(LogisticRegression(max_iter=300).fit(X, Y), tmp_path / "m.joblib"))
    assert adapter.load_seconds > 0
    result = adapter.predict(X[:5])
    assert result.inference_seconds >= 0
    assert result.batch_seconds == (result.inference_seconds,)


# --- dataset adapter --------------------------------------------------------------------------


class TestSklearnDatasetContract(DatasetAdapterContract):
    expected_dataset_capabilities = frozenset(
        {
            DatasetCapability.RANDOM_ACCESS,
            DatasetCapability.ITERATION,
            DatasetCapability.BATCHING,
            DatasetCapability.LABELS,
            DatasetCapability.TRAIN_SPLIT,
            DatasetCapability.TEST_SPLIT,
        }
    )

    @pytest.fixture
    def dataset(self) -> SklearnDatasetAdapter:
        return SklearnDatasetAdapter.load("builtin:iris", version="1", options={})

    @pytest.fixture
    def reloaded_dataset(self) -> SklearnDatasetAdapter:
        return SklearnDatasetAdapter.load("builtin:iris", version="1", options={})

    @pytest.fixture
    def different_dataset(self) -> SklearnDatasetAdapter:
        return SklearnDatasetAdapter.load("builtin:iris", version="1", options={"seed": 1})

    @pytest.fixture
    def missing_dataset_source(self, tmp_path: Path) -> str:
        return str(tmp_path / "missing.npz")


def ds(x: object = X, y: object = Y, **kw: object) -> SklearnDatasetAdapter:
    return SklearnDatasetAdapter(x, y, version="1", **kw)  # type: ignore[arg-type]


def test_builtin_iris_metadata() -> None:
    d = SklearnDatasetAdapter.load(
        "builtin:iris", version="1", options={"seed": 0, "test_size": 0.25}
    )
    meta = d.metadata()
    assert (meta.num_samples, meta.num_features, meta.class_count) == (150, 4, 3)
    assert meta.task is TaskType.CLASSIFICATION
    assert sum(int(str(n)) for n in meta.splits.values()) == 150
    assert meta.input_schema is not None
    assert meta.input_schema.feature_names == tuple(IRIS.feature_names)
    assert meta.input_schema.dtype == "float64"
    assert meta.target_schema is not None
    assert meta.target_schema.class_labels == (0, 1, 2)
    assert meta.source["loader"] == "sklearn.datasets.load_iris"
    assert meta.missing_values is None  # not inspected by default
    assert d.metadata(deep=True).missing_values == 0
    assert d.load_split_disjoint() if hasattr(d, "load_split_disjoint") else True
    train = {i for b in d.batches(50, "train") for i in b.indices}
    test = {i for b in d.batches(50, "test") for i in b.indices}
    assert not train & test
    assert train | test == set(range(150))


def test_the_split_is_deterministic_per_seed() -> None:
    a = SklearnDatasetAdapter.load("builtin:iris", version="1", options={"seed": 3})
    b = SklearnDatasetAdapter.load("builtin:iris", version="1", options={"seed": 3})
    c = SklearnDatasetAdapter.load("builtin:iris", version="1", options={"seed": 4})
    idx = lambda d: [i for bt in d.batches(1000, "test") for i in bt.indices]  # noqa: E731
    assert idx(a) == idx(b)
    assert idx(a) != idx(c)


def _independent_fingerprint(
    x: object, y: object, names: object, splits: Mapping[str, object]
) -> str:
    """Re-derive the documented algorithm without using the adapter's code path."""
    x, y = np.asarray(x), np.asarray(y)  # type: ignore[assignment]
    header = {
        "kind": "tabular",
        "X": [str(x.dtype), list(x.shape)],  # type: ignore[attr-defined]
        "y": [str(y.dtype), list(y.shape)],  # type: ignore[attr-defined]
        "feature_names": list(names) if names is not None else None,  # type: ignore[call-overload]
        "splits": {k: len(v) for k, v in sorted(splits.items())},  # type: ignore[arg-type]
    }
    blob = json.dumps(header, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    blob += (
        x.astype(x.dtype.newbyteorder("<")).tobytes()
        + y.astype(y.dtype.newbyteorder("<")).tobytes()
    )  # type: ignore[attr-defined]
    for _, idx in sorted(splits.items()):
        blob += np.asarray(idx).astype("<i8").tobytes()
    return "sha256:" + hashlib.sha256(blob).hexdigest()


def test_fingerprint_matches_an_independent_computation() -> None:
    splits = {"train": np.arange(100), "test": np.arange(100, 150)}
    d = ds(splits=splits, feature_names=list(IRIS.feature_names))
    assert d.fingerprint() == _independent_fingerprint(X, Y, IRIS.feature_names, splits)
    no_names = ds(splits=splits)
    assert no_names.fingerprint() == _independent_fingerprint(X, Y, None, splits)


def test_fingerprint_is_order_dtype_name_target_and_split_sensitive() -> None:
    base = ds(splits={"test": np.arange(10)}, feature_names=list("abcd")).fingerprint()
    assert ds(splits={"test": np.arange(10)}, feature_names=list("abcd")).fingerprint() == base
    assert (
        ds(
            X[::-1], Y[::-1], splits={"test": np.arange(10)}, feature_names=list("abcd")
        ).fingerprint()
        != base
    )  # row order matters
    assert (
        ds(splits={"test": np.arange(10)}, feature_names=list("abce")).fingerprint() != base
    )  # feature names
    assert (
        ds(
            X.astype("float32"), splits={"test": np.arange(10)}, feature_names=list("abcd")
        ).fingerprint()
        != base
    )  # dtype
    y2 = Y.copy()
    y2[0] = 2
    assert (
        ds(y=y2, splits={"test": np.arange(10)}, feature_names=list("abcd")).fingerprint() != base
    )  # target
    assert (
        ds(splits={"test": np.arange(1, 11)}, feature_names=list("abcd")).fingerprint() != base
    )  # split
    x2 = X.copy()
    x2[70, 2] += 1e-9
    assert (
        ds(x2, splits={"test": np.arange(10)}, feature_names=list("abcd")).fingerprint() != base
    )  # one cell


def test_changed_dataset_is_detected_after_a_fresh_load(tmp_path: Path) -> None:
    path = tmp_path / "d.npz"
    np.savez(path, X=X, y=Y)
    before = SklearnDatasetAdapter.load(path, version="1", options={}).fingerprint()
    x2 = X.copy()
    x2[0, 0] += 1.0
    np.savez(path, X=x2, y=Y)
    assert SklearnDatasetAdapter.load(path, version="1", options={}).fingerprint() != before


def test_npz_round_trip_and_safety(tmp_path: Path) -> None:
    path = tmp_path / "d.npz"
    np.savez(path, X=X, y=Y, feature_names=np.array(IRIS.feature_names), split_test=np.arange(20))
    d = SklearnDatasetAdapter.load(path, version="2", options={})
    assert d.splits() == ("test",)
    assert d.metadata().source["format"] == "npz"
    assert d.fingerprint() == _independent_fingerprint(
        X, Y, IRIS.feature_names, {"test": np.arange(20)}
    )
    pickled = tmp_path / "obj.npz"
    np.savez(pickled, X=np.array([object(), object()], dtype=object))
    with pytest.raises(DatasetLoadError):
        SklearnDatasetAdapter.load(pickled, version="1", options={})
    no_x = tmp_path / "nox.npz"
    np.savez(no_x, y=Y)
    with pytest.raises(InvalidDatasetError, match="'X'"):
        SklearnDatasetAdapter.load(no_x, version="1", options={})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"x": X[0]},  # 1-D
        {"x": np.zeros((0, 4))},
        {"x": np.array([["a", "b"]])},
        {"y": Y[:10]},  # length mismatch
        {"y": np.array([object()] * 150, dtype=object)},
        {"splits": {"test": [0, 150]}},
        {"splits": {"test": [-1]}},
        {"splits": {"test": [0.5]}},
        {"feature_names": ["a"]},
    ],
)
def test_invalid_datasets_are_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(InvalidDatasetError):
        ds(**kwargs)


def test_bad_builtin_options_and_names() -> None:
    for source, options in (
        ("builtin:nope", {}),
        ("builtin:iris", {"test_size": 2.0}),
        ("builtin:iris", {"seed": "x"}),
    ):
        with pytest.raises(DatasetLoadError):
            SklearnDatasetAdapter.load(source, version="1", options=options)


def test_unlabeled_datasets_and_task_inference() -> None:
    d = ds(y=None)
    assert DatasetCapability.LABELS not in d.capabilities
    assert d.metadata().target_schema is None
    assert d.metadata().task is TaskType.UNKNOWN
    assert next(iter(d.batches(10))).target is None
    assert ds(y=Y.astype(float)).metadata().task is TaskType.REGRESSION
    assert ds(y=np.array(["a", "b"] * 75)).metadata().task is TaskType.CLASSIFICATION
    assert ds(y=Y).metadata().task is TaskType.UNKNOWN  # integer targets are ambiguous
    assert ds(y=Y, task=TaskType.CLASSIFICATION).metadata().class_count == 3


def test_missing_values_are_counted_only_on_deep_inspection() -> None:
    x = X.copy()
    x[3, 1] = np.nan
    d = ds(x)
    assert d.metadata().missing_values is None
    assert d.metadata(deep=True).missing_values == 1


def test_unsplit_batches_are_views_not_copies() -> None:
    d = ds()
    first = next(iter(d.batches(10)))
    assert np.shares_memory(first.inputs, X) or np.shares_memory(first.inputs, d._X)
