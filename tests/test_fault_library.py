"""Numerical validation of every built-in fault, determinism, scopes, mutation and edge cases."""

import numpy as np
import pytest

from experionyx.errors import FaultCompatibilityError, FaultError
from experionyx.faults.apply import FaultApplier, apply_to_arrays
from experionyx.faults.library import default_fault_registry
from experionyx.faults.spec import FaultScope, ScopeKind

FR = default_fault_registry()
RNG = np.random.default_rng(123)
X = RNG.normal(size=(4000, 6))
IMG = RNG.random(size=(300, 1, 8, 8)).astype(np.float32)
Y = (np.arange(4000) % 3).astype(np.int64)


def run(name: str, x=X, y=None, seed=1, scope=None, **params):  # type: ignore[no-untyped-def]
    spec = FR.make(name, seed=seed, scope=scope, **params)
    return apply_to_arrays(spec, FR, x, y, classes=(0, 1, 2))


# --- noise: statistical, not exact -------------------------------------------------------------


def test_gaussian_noise_has_the_requested_mean_and_std() -> None:
    out, _, rep = run("gaussian_noise", sigma=0.5)
    noise = out - X
    assert noise.mean() == pytest.approx(0.0, abs=0.01)
    assert noise.std() == pytest.approx(0.5, rel=0.02)
    assert rep[0].affected_samples == 4000
    assert np.allclose(run("gaussian_noise", sigma=0.0)[0], X)  # sigma=0 is the identity


def test_uniform_noise_stays_in_bounds_with_the_right_variance() -> None:
    noise = run("uniform_noise", half_width=0.3)[0] - X
    assert np.abs(noise).max() <= 0.3 + 1e-12
    assert noise.var() == pytest.approx(0.3**2 / 3, rel=0.03)


def test_feature_dropout_and_missing_values_hit_the_expected_share() -> None:
    dropped, _, _ = run("feature_dropout", probability=0.3, value=-7.0)
    assert (dropped == -7.0).mean() == pytest.approx(0.3, abs=0.01)
    assert np.array_equal(dropped[dropped != -7.0], X[dropped != -7.0])  # only dropped cells change
    missing, _, _ = run("missing_values", probability=0.2)
    assert np.isnan(missing).mean() == pytest.approx(0.2, abs=0.01)
    filled, _, _ = run("missing_values", probability=0.2, fill_value=99.0)
    assert (filled == 99.0).mean() == pytest.approx(0.2, abs=0.01)
    assert not np.isnan(filled).any()
    assert np.array_equal(run("feature_dropout", probability=0.0)[0], X)
    assert (run("feature_dropout", probability=1.0, value=0.0)[0] == 0).all()


def test_feature_mask_masks_a_fixed_column_subset_exactly() -> None:
    out, _, _ = run("feature_mask", fraction=0.5, value=9.0)
    masked_cols = np.flatnonzero((out == 9.0).all(axis=0))
    assert len(masked_cols) == 3  # round(0.5 * 6)
    other = np.setdiff1d(np.arange(6), masked_cols)
    assert np.array_equal(out[:, other], X[:, other])
    assert np.array_equal(run("feature_mask", fraction=0.0)[0], X)


def test_scaling_and_offset_are_exact_deterministic_transformations() -> None:
    scaled = run("feature_scaling", factor=2.5)[0]
    assert np.allclose(scaled, X * 2.5, atol=1e-12)
    shifted = run("feature_offset", offset=-1.5)[0]
    assert np.allclose(shifted, X - 1.5, atol=1e-12)
    half = run("feature_scaling", factor=3.0, fraction=0.5)[0]
    ratio_cols = np.flatnonzero(~np.isclose(half, X).all(axis=0))
    assert len(ratio_cols) == 3
    assert np.allclose(half[:, ratio_cols], X[:, ratio_cols] * 3.0)


def test_feature_permutation_cyclically_moves_selected_columns() -> None:
    out = run("feature_permutation", fraction=1.0)[0]
    assert np.array_equal(out, np.roll(X, 1, axis=1))
    assert np.array_equal(np.sort(out, axis=1), np.sort(X, axis=1))  # values are only moved
    with pytest.raises(FaultCompatibilityError, match="at least 2"):
        run("feature_permutation", fraction=0.1)


def test_outlier_injection_rate_and_magnitude() -> None:
    out = run("outlier_injection", probability=0.1, magnitude=5.0)[0]
    diff = out - X
    hit = np.abs(diff) > 1e-9
    assert hit.mean() == pytest.approx(0.1, abs=0.01)
    assert np.allclose(np.abs(diff[hit]), 5.0)
    assert set(np.sign(diff[hit])) == {-1.0, 1.0}


# --- image / tensor faults ---------------------------------------------------------------------


def test_salt_and_pepper_values_and_rate() -> None:
    out = run("salt_and_pepper", x=IMG, probability=0.2)[0]
    changed = out != IMG
    assert changed.mean() == pytest.approx(0.2, abs=0.02)
    assert set(np.unique(out[changed])) <= {0.0, 1.0}
    assert out.dtype == np.float32


def test_brightness_and_contrast_are_exact_and_clipped() -> None:
    bright = run("brightness", x=IMG, delta=0.25)[0]
    assert np.allclose(bright, np.clip(IMG + 0.25, 0, 1), atol=1e-6)
    assert bright.max() <= 1.0
    con = run("contrast", x=IMG, factor=0.5)[0]
    mean = IMG.mean(axis=(1, 2, 3), keepdims=True)
    assert np.allclose(con, np.clip((IMG - mean) * 0.5 + mean, 0, 1), atol=1e-6)
    assert np.allclose(run("contrast", x=IMG, factor=1.0)[0], IMG, atol=1e-6)  # identity


def test_random_occlusion_overwrites_exactly_one_patch_per_sample() -> None:
    out = run("random_occlusion", x=IMG + 1.0, fraction=0.5, value=-1.0)[0]
    per_sample = (out == -1.0).reshape(len(out), -1).sum(axis=1)
    assert (per_sample == 16).all()  # a 4x4 patch on 8x8 images
    assert (run("random_occlusion", x=IMG + 1.0, fraction=1.0, value=-1.0)[0] == -1.0).all()


def test_box_blur_preserves_constant_images_and_smooths_others() -> None:
    flat = np.full((5, 1, 8, 8), 0.4, dtype=np.float32)
    assert np.allclose(run("box_blur", x=flat, radius=2)[0], 0.4, atol=1e-6)
    blurred = run("box_blur", x=IMG, radius=1)[0]
    assert blurred.std() < IMG.std()
    assert blurred.shape == IMG.shape


# --- label faults ------------------------------------------------------------------------------


def test_label_flip_rate_is_controlled_and_always_changes_the_label() -> None:
    _, out, rep = run("label_flip", x=X, y=Y, rate=0.3)
    changed = out != Y
    assert changed.mean() == pytest.approx(0.3, abs=0.02)
    assert rep[0].changed_labels == int(changed.sum())
    assert set(np.unique(out)) <= {0, 1, 2}
    assert np.array_equal(run("label_flip", x=X, y=Y, rate=0.0)[1], Y)
    assert (run("label_flip", x=X, y=Y, rate=1.0)[1] != Y).all()


def test_label_randomize_may_keep_the_label_so_the_effective_rate_is_lower() -> None:
    _, out, _ = run("label_randomize", x=X, y=Y, rate=0.6)
    assert (out != Y).mean() == pytest.approx(0.6 * 2 / 3, abs=0.02)


def test_label_faults_leave_inputs_untouched_and_need_classes() -> None:
    x2, _, _ = run("label_flip", x=X, y=Y, rate=0.5)
    assert x2 is X
    spec = FR.make("label_flip", seed=0, rate=0.5)
    with pytest.raises(FaultCompatibilityError, match="class labels"):
        apply_to_arrays(spec, FR, X, Y, classes=None)
    out = apply_to_arrays(FR.make("label_flip", seed=0, rate=1.0, classes=(0, 1, 2)), FR, X, Y)[1]
    assert (out != Y).all()  # explicit classes work without dataset metadata


def test_string_labels_are_supported_without_truncation() -> None:
    y = np.array(["a", "b", "a", "b"] * 50)
    spec = FR.make("label_flip", seed=0, rate=1.0, classes=("a", "b", "longer-label"))
    out = apply_to_arrays(spec, FR, np.zeros((200, 2)), y)[1]
    assert (out != y).all()
    assert "longer-label" in set(out)  # not truncated to the original one-character width
    assert set(out[y == "a"]) <= {"b", "longer-label"}


# --- determinism, seeds, mutation, batch invariance --------------------------------------------


def test_same_seed_reproduces_and_a_different_seed_differs() -> None:
    a, b = run("gaussian_noise", sigma=0.3, seed=5)[0], run("gaussian_noise", sigma=0.3, seed=5)[0]
    c = run("gaussian_noise", sigma=0.3, seed=6)[0]
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_perturbation_is_independent_of_batch_size() -> None:
    spec = FR.make(
        "feature_dropout",
        seed=3,
        probability=0.2,
        scope=FaultScope(ScopeKind.RANDOM_SUBSET, fraction=0.5),
    )
    whole = apply_to_arrays(spec, FR, X)[0]
    parts = []
    applier = FaultApplier(spec, FR, n_total=len(X), classes=None)
    for start in range(0, len(X), 333):  # uneven batches, final partial batch
        chunk = X[start : start + 333]
        parts.append(applier.apply(chunk, None, np.arange(start, start + len(chunk)))[0])
    assert np.array_equal(np.vstack(parts), whole)


def test_inputs_are_never_mutated() -> None:
    x, y, img = X.copy(), Y.copy(), IMG.copy()
    for name, params, data in (
        ("gaussian_noise", {"sigma": 1.0}, x),
        ("feature_mask", {"fraction": 0.5}, x),
        ("feature_scaling", {"factor": 2.0}, x),
        ("random_occlusion", {"fraction": 0.5}, img),
        ("brightness", {"delta": 0.2}, img),
        ("box_blur", {"radius": 1}, img),
    ):
        before = data.copy()
        run(name, x=data, **params)
        assert np.array_equal(data, before), name
    run("label_flip", x=x, y=y, rate=0.5)
    assert np.array_equal(y, Y)
    assert np.array_equal(x, X)


def test_dtype_and_shape_are_preserved() -> None:
    x32 = X.astype(np.float32)
    for name, params in (
        ("gaussian_noise", {"sigma": 0.2}),
        ("uniform_noise", {"half_width": 0.2}),
        ("feature_dropout", {"probability": 0.2}),
        ("feature_scaling", {"factor": 2.0}),
        ("feature_offset", {"offset": 1.0}),
        ("outlier_injection", {"probability": 0.1, "magnitude": 2.0}),
    ):
        out = run(name, x=x32, **params)[0]
        assert out.dtype == np.float32
        assert out.shape == x32.shape
    assert run("brightness", x=IMG.astype(np.float64), delta=0.1)[0].dtype == np.float64


# --- scopes ------------------------------------------------------------------------------------


def test_random_subset_scope_affects_an_exact_deterministic_share() -> None:
    scope = FaultScope(ScopeKind.RANDOM_SUBSET, fraction=0.1)
    out, _, rep = run("feature_offset", scope=scope, offset=100.0)
    affected = np.flatnonzero((out != X).any(axis=1))
    assert len(affected) == 400 == rep[0].affected_samples
    assert np.array_equal(
        out[np.setdiff1d(np.arange(4000), affected)], X[np.setdiff1d(np.arange(4000), affected)]
    )
    again = run("feature_offset", scope=scope, offset=100.0)[0]
    assert np.array_equal(out, again)
    other_seed = run("feature_offset", scope=scope, offset=100.0, seed=2)[0]
    assert not np.array_equal(np.flatnonzero((other_seed != X).any(axis=1)), affected)


def test_class_scope_affects_only_the_targeted_class() -> None:
    out, _, rep = run(
        "feature_offset", y=Y, scope=FaultScope(ScopeKind.CLASS, label=1), offset=10.0
    )
    changed = (out != X).any(axis=1)
    assert np.array_equal(changed, Y == 1)
    assert rep[0].affected_samples == int((Y == 1).sum())
    with pytest.raises(FaultCompatibilityError, match="true labels"):
        run("feature_offset", scope=FaultScope(ScopeKind.CLASS, label=1), offset=10.0)


def test_scope_share_changes_the_experiment_not_just_the_label() -> None:
    full, part = (
        run("gaussian_noise", sigma=1.0)[0],
        run("gaussian_noise", sigma=1.0, scope=FaultScope(ScopeKind.RANDOM_SUBSET, fraction=0.1))[
            0
        ],
    )
    assert (full != X).any(axis=1).mean() == 1.0
    assert (part != X).any(axis=1).mean() == pytest.approx(0.1, abs=1e-9)


# --- compound faults ---------------------------------------------------------------------------


def test_compound_order_matters_and_reports_every_component() -> None:
    noise, scale = (
        FR.make("gaussian_noise", seed=1, sigma=0.5),
        FR.make("feature_scaling", seed=2, factor=3.0),
    )
    ab = apply_to_arrays(FR.compound(noise, scale), FR, X)
    ba = apply_to_arrays(FR.compound(scale, noise), FR, X)
    assert not np.allclose(ab[0], ba[0])  # noise-then-scale != scale-then-noise
    assert (ab[0] - X * 3.0).std() == pytest.approx(1.5, rel=0.05)  # noise scaled by 3
    assert (ba[0] - X * 3.0).std() == pytest.approx(0.5, rel=0.05)
    assert [r.fault_type for r in ab[2]] == ["gaussian_noise", "feature_scaling"]
    assert [r.seed for r in ba[2]] == [2, 1]
    mixed = apply_to_arrays(
        FR.compound(
            FR.make("feature_offset", seed=0, offset=1.0), FR.make("label_flip", seed=3, rate=0.5)
        ),
        FR,
        X,
        Y,
        classes=(0, 1, 2),
    )
    assert np.allclose(mixed[0], X + 1.0)
    assert (mixed[1] != Y).mean() == pytest.approx(0.5, abs=0.03)
    assert [r.target for r in mixed[2]] == ["INPUT", "LABEL"]


# --- edge cases --------------------------------------------------------------------------------


def test_empty_and_single_sample_inputs() -> None:
    empty = run("gaussian_noise", x=np.zeros((0, 6)), sigma=1.0)[0]
    assert empty.shape == (0, 6)
    one = run("gaussian_noise", x=X[:1], sigma=1.0)[0]
    assert one.shape == (1, 6)
    assert not np.array_equal(one, X[:1])
    one_label = run("label_flip", x=X[:1], y=Y[:1], rate=1.0)[1]
    assert one_label[0] != Y[0]


def test_non_finite_inputs_propagate_and_are_not_silently_replaced() -> None:
    x = X[:100].copy()
    x[3, 2] = np.nan
    x[4, 1] = np.inf
    out = run("feature_scaling", x=x, factor=2.0)[0]
    assert np.isnan(out[3, 2])
    assert np.isinf(out[4, 1])
    noisy = run("gaussian_noise", x=x, sigma=0.1)[0]
    assert np.isnan(noisy[3, 2])
    scope = FaultScope(ScopeKind.RANDOM_SUBSET, fraction=0.5)
    partial = run("feature_offset", x=x, scope=scope, offset=1.0)[0]
    assert np.isnan(partial[3, 2])


def test_shape_and_dtype_mismatches_are_rejected_explicitly() -> None:
    spec = FR.make("gaussian_noise", seed=0, sigma=0.1)
    with pytest.raises(FaultError, match="shape mismatch"):
        FaultApplier(spec, FR, n_total=10, classes=None).apply(X[:5], None, np.arange(4))
    with pytest.raises(FaultError, match="shape mismatch"):
        FaultApplier(FR.make("label_flip", seed=0, rate=0.5), FR, n_total=10, classes=(0, 1)).apply(
            X[:5], Y[:3], np.arange(5)
        )
    with pytest.raises(FaultCompatibilityError, match="floating-point"):
        run("gaussian_noise", x=np.arange(12).reshape(6, 2), sigma=0.1)
    with pytest.raises(FaultCompatibilityError, match="2-D"):
        run("feature_mask", x=IMG, fraction=0.5)
    with pytest.raises(FaultCompatibilityError, match="image-shaped"):
        run("brightness", x=X, delta=0.1)
    with pytest.raises(FaultCompatibilityError, match="not implemented"):
        run("covariate_shift")
