"""Fault specifications: registry, identity, versioning, parameter validation, scopes, entities."""

import dataclasses
from datetime import UTC, datetime

import pytest

from experionyx.domain import ExperimentStatus
from experionyx.errors import (
    FaultCompatibilityError,
    FaultLimitError,
    FaultNotFoundError,
    ValidationError,
)
from experionyx.faults.design import FaultDesign, FaultLimits, SweepSpec
from experionyx.faults.entities import FaultExperiment, FaultTrial, TrialStatus
from experionyx.faults.library import (
    GaussianNoiseParams,
    default_fault_registry,
)
from experionyx.faults.spec import (
    FaultCategory,
    FaultParameters,
    FaultRegistry,
    FaultScope,
    FaultTarget,
    FaultType,
    Requirement,
    ScopeKind,
    check_compatibility,
)

FR = default_fault_registry()
T0 = datetime(2026, 9, 19, 12, tzinfo=UTC)


# --- registry ---------------------------------------------------------------------------------


def test_registry_lists_the_documented_faults_with_metadata() -> None:
    names = set(FR.names())
    assert {"gaussian_noise", "uniform_noise", "feature_dropout", "feature_mask", "missing_values",
            "feature_permutation", "feature_scaling", "feature_offset", "outlier_injection",
            "salt_and_pepper", "brightness", "contrast", "random_occlusion", "box_blur",
            "label_flip", "label_randomize"} <= names  # fmt: skip
    gn = FR.resolve("gaussian_noise")
    assert (gn.category, gn.target, gn.version, gn.stochastic) == (
        FaultCategory.NOISE,
        FaultTarget.INPUT,
        "1",
        True,
    )
    assert Requirement.FLOAT_ARRAY in gn.requires
    assert gn.sweepable() == ("sigma",)
    assert FR.resolve("brightness").stochastic is False
    assert FR.resolve("label_flip").target is FaultTarget.LABEL
    assert FR.resolve("label_flip").category is FaultCategory.LABEL_CORRUPTION
    assert all(ft.description for ft in FR.all())


def test_future_capabilities_are_registered_but_explicitly_not_runnable() -> None:
    for name, cat in (
        ("covariate_shift", FaultCategory.DISTRIBUTION_SHIFT),
        ("temporal_delay", FaultCategory.TEMPORAL_CORRUPTION),
        ("resource_throttle", FaultCategory.RESOURCE_FAULT),
    ):
        ft = FR.resolve(name)
        assert ft.category is cat
        assert not ft.implemented
    spec = FR.make("temporal_delay", seed=0)
    with pytest.raises(FaultCompatibilityError, match="not implemented"):
        check_compatibility(spec, FR, ndim=2, dtype_kind="f", has_classes=False)


def test_the_taxonomy_contains_every_documented_category() -> None:
    assert {c.value for c in FaultCategory} == {
        "INPUT_CORRUPTION",
        "LABEL_CORRUPTION",
        "MISSINGNESS",
        "NOISE",
        "TRANSFORMATION",
        "DISTRIBUTION_SHIFT",
        "TEMPORAL_CORRUPTION",
        "RESOURCE_FAULT",
        "COMPOUND",
    }


def test_registry_rejects_duplicates_and_reports_unknown_faults() -> None:
    with pytest.raises(ValidationError, match="already registered"):
        FR.register(FR.resolve("gaussian_noise"))
    with pytest.raises(FaultNotFoundError, match="registered"):
        FR.resolve("nope")
    with pytest.raises(FaultNotFoundError, match="no version"):
        FR.resolve("gaussian_noise", "99")
    assert "nope" not in FR.names()


def test_registries_are_independent_and_new_faults_plug_in_without_engine_changes() -> None:
    @dataclasses.dataclass(frozen=True)
    class ShiftParams(FaultParameters):
        amount: float = 1.0

    reg = default_fault_registry()
    reg.register(
        FaultType(
            "my_fault",
            "1",
            FaultCategory.TRANSFORMATION,
            FaultTarget.INPUT,
            ShiftParams,
            "test",
            apply=lambda x, m, p, c: x,
        )
    )
    assert "my_fault" in reg.names()
    assert "my_fault" not in default_fault_registry().names()
    assert reg.make("my_fault", seed=1, amount=2.0).parameters == ShiftParams(2.0)


def test_versioning_resolves_newest_by_default_and_exact_versions_on_replay() -> None:
    reg = default_fault_registry()
    old = reg.resolve("gaussian_noise")
    reg.register(dataclasses.replace(old, version="2"))
    reg.register(dataclasses.replace(old, version="10"))
    assert reg.versions("gaussian_noise") == ("1", "2", "10")
    assert reg.resolve("gaussian_noise").version == "10"  # numeric, not lexicographic, ordering
    assert reg.resolve("gaussian_noise", "1").version == "1"
    stored = FR.make("gaussian_noise", seed=1, sigma=0.5).to_dict()
    assert FR.from_dict(stored).version == "1"
    stored_missing = {**stored, "version": "7"}
    with pytest.raises(FaultNotFoundError, match="reproducible only with the implementation"):
        FR.from_dict(stored_missing)


# --- identity ---------------------------------------------------------------------------------


def test_identity_is_deterministic_and_depends_on_every_meaningful_field() -> None:
    a = FR.make("gaussian_noise", seed=1, sigma=0.5)
    assert a.id == FR.make("gaussian_noise", seed=1, sigma=0.5).id
    assert a.id.startswith("fap_")
    assert a.family_id.startswith("flt_")
    variants = [
        FR.make("gaussian_noise", seed=2, sigma=0.5),
        FR.make("gaussian_noise", seed=1, sigma=0.6),
        FR.make(
            "gaussian_noise",
            seed=1,
            sigma=0.5,
            scope=FaultScope(ScopeKind.RANDOM_SUBSET, fraction=0.5),
        ),
        FR.make("uniform_noise", seed=1, half_width=0.5),
    ]
    assert len({a.id, *(v.id for v in variants)}) == 5
    assert dataclasses.replace(a, version="2").id != a.id  # the implementation version is identity


def test_family_identity_ignores_the_seed_but_not_the_rest() -> None:
    a, b = (
        FR.make("gaussian_noise", seed=1, sigma=0.5),
        FR.make("gaussian_noise", seed=2, sigma=0.5),
    )
    assert a.family_id == b.family_id
    assert a.family_id != FR.make("gaussian_noise", seed=1, sigma=0.7).family_id
    assert (
        a.family_id
        != FR.make(
            "gaussian_noise", seed=1, sigma=0.5, scope=FaultScope(ScopeKind.CLASS, label=1)
        ).family_id
    )


def test_equal_parameters_written_differently_get_the_same_identity() -> None:
    assert (
        FR.make("gaussian_noise", seed=1, sigma=1).id
        == FR.make("gaussian_noise", seed=1, sigma=1.0).id
    )


def test_identity_contains_no_timestamps_or_runtime_state() -> None:
    a = FR.make("gaussian_noise", seed=1, sigma=0.5)
    assert "time" not in str(a.to_dict()).lower()
    assert a.to_dict() == FR.make("gaussian_noise", seed=1, sigma=0.5).to_dict()


def test_spec_round_trips_through_its_canonical_dict() -> None:
    spec = FR.make(
        "feature_dropout",
        seed=3,
        probability=0.2,
        value=1.0,
        scope=FaultScope(ScopeKind.RANDOM_SUBSET, fraction=0.25),
    )
    again = FR.from_dict(spec.to_dict())
    assert again == spec
    assert again.id == spec.id


def test_from_dict_is_strict() -> None:
    good = FR.make("gaussian_noise", seed=1, sigma=0.5).to_dict()
    for bad in (
        {**good, "surprise": 1},
        {k: v for k, v in good.items() if k != "seed"},
        {**good, "parameters": {"sigma": 0.5, "extra": 1}},
        {**good, "parameters": {"sigma": "big"}},
        {**good, "schema_version": 9},
        {**good, "seed": -1},
    ):
        with pytest.raises(ValidationError):
            FR.from_dict(bad)


# --- compound faults --------------------------------------------------------------------------


def test_compound_preserves_component_order_parameters_seeds_and_scope() -> None:
    a = FR.make("gaussian_noise", seed=1, sigma=0.1)
    b = FR.make(
        "feature_dropout",
        seed=2,
        probability=0.2,
        scope=FaultScope(ScopeKind.RANDOM_SUBSET, fraction=0.5),
    )
    ab, ba = FR.compound(a, b), FR.compound(b, a)
    assert ab.id != ba.id  # GaussianNoise -> Dropout is not Dropout -> GaussianNoise
    assert ab.family_id != ba.family_id
    assert [c.type for c in ab.flatten()] == ["gaussian_noise", "feature_dropout"]
    stored = ab.to_dict()
    assert [c["seed"] for c in stored["components"]] == [1, 2]  # type: ignore[index]
    assert FR.from_dict(stored) == ab
    assert FR.from_dict(stored).flatten()[1].scope.fraction == 0.5
    with pytest.raises(ValidationError):
        FR.compound(a)  # needs at least two components
    with pytest.raises(ValidationError, match="compound"):
        ab.with_parameter("sigma", 0.2)


# --- parameter validation ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "params"),
    [
        ("gaussian_noise", {"sigma": -0.1}),
        ("gaussian_noise", {"sigma": float("nan")}),
        ("gaussian_noise", {"sigma": float("inf")}),
        ("gaussian_noise", {"sigma": "0.1"}),
        ("gaussian_noise", {"sigma": True}),
        ("gaussian_noise", {}),
        ("gaussian_noise", {"sigma": 0.1, "bogus": 1}),
        ("uniform_noise", {"half_width": -1.0}),
        ("feature_dropout", {"probability": 1.1}),
        ("feature_dropout", {"probability": -0.1}),
        ("missing_values", {"probability": 2.0}),
        ("missing_values", {"probability": 0.5, "fill_value": float("nan")}),
        ("feature_mask", {"fraction": 1.5}),
        ("feature_scaling", {"factor": 0.0}),
        ("feature_scaling", {"factor": -2.0}),
        ("feature_scaling", {"factor": 2.0, "fraction": 1.2}),
        ("feature_offset", {"offset": float("inf")}),
        ("outlier_injection", {"probability": 0.1, "magnitude": 0.0}),
        ("salt_and_pepper", {"probability": 0.1, "low": 1.0, "high": 0.0}),
        ("brightness", {"delta": float("nan")}),
        ("brightness", {"delta": 0.1, "value_min": 1.0, "value_max": 0.0}),
        ("contrast", {"factor": 0.0}),
        ("random_occlusion", {"fraction": 0.0}),
        ("random_occlusion", {"fraction": 1.5}),
        ("box_blur", {"radius": 0}),
        ("box_blur", {"radius": 1.5}),
        ("label_flip", {"rate": 1.5}),
        ("label_flip", {"rate": 0.1, "classes": (1,)}),
    ],
)
def test_invalid_parameters_fail_at_construction_before_any_experiment(
    name: str, params: dict[str, object]
) -> None:
    with pytest.raises(ValidationError):
        FR.make(name, seed=0, **params)


def test_boundary_parameters_are_accepted() -> None:
    for name, params in (
        ("gaussian_noise", {"sigma": 0.0}),
        ("feature_dropout", {"probability": 0.0}),
        ("feature_dropout", {"probability": 1.0}),
        ("label_flip", {"rate": 1.0}),
        ("random_occlusion", {"fraction": 1.0}),
        ("feature_scaling", {"factor": 1e-9}),
    ):
        FR.make(name, seed=0, **params)


def test_seed_must_be_a_non_negative_integer() -> None:
    for seed in (-1, 1.5, "1", True):
        with pytest.raises(ValidationError):
            FR.make("gaussian_noise", seed=seed, sigma=0.1)  # type: ignore[arg-type]


def test_parameters_are_typed_dataclasses_not_dicts() -> None:
    spec = FR.make("gaussian_noise", seed=1, sigma=0.5)
    assert isinstance(spec.parameters, GaussianNoiseParams)
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.parameters.sigma = 1.0  # type: ignore[misc]


# --- scope ------------------------------------------------------------------------------------


def test_scope_validation_and_reserved_kinds() -> None:
    FaultScope()
    FaultScope(ScopeKind.RANDOM_SUBSET, fraction=0.1)
    FaultScope(ScopeKind.CLASS, label="a")
    for bad in (
        lambda: FaultScope(ScopeKind.ALL, fraction=0.5),
        lambda: FaultScope(ScopeKind.RANDOM_SUBSET),
        lambda: FaultScope(ScopeKind.RANDOM_SUBSET, fraction=0.0),
        lambda: FaultScope(ScopeKind.RANDOM_SUBSET, fraction=1.5),
        lambda: FaultScope(ScopeKind.RANDOM_SUBSET, fraction=float("nan")),
        lambda: FaultScope(ScopeKind.CLASS),
        lambda: FaultScope(ScopeKind.CLASS, label=1, fraction=0.5),
    ):
        with pytest.raises(ValidationError):
            bad()
    for reserved in (ScopeKind.SLICE, ScopeKind.FEATURE, ScopeKind.REGION, ScopeKind.TIME_WINDOW):
        with pytest.raises(ValidationError, match="reserved"):
            FaultScope(reserved)


# --- compatibility pre-flight -----------------------------------------------------------------


def test_compatibility_preflight_uses_dataset_metadata() -> None:
    noise = FR.make("gaussian_noise", seed=0, sigma=0.1)
    check_compatibility(noise, FR, ndim=2, dtype_kind="f", has_classes=False)
    check_compatibility(
        noise, FR, ndim=None, dtype_kind=None, has_classes=False
    )  # unknown: allowed
    with pytest.raises(FaultCompatibilityError, match="FLOAT_ARRAY"):
        check_compatibility(noise, FR, ndim=2, dtype_kind="i", has_classes=False)
    with pytest.raises(FaultCompatibilityError, match="IMAGE"):
        check_compatibility(
            FR.make("brightness", seed=0, delta=0.1), FR, ndim=2, dtype_kind="f", has_classes=False
        )
    with pytest.raises(FaultCompatibilityError, match="TABULAR"):
        check_compatibility(
            FR.make("feature_mask", seed=0, fraction=0.5),
            FR,
            ndim=4,
            dtype_kind="f",
            has_classes=False,
        )
    with pytest.raises(FaultCompatibilityError, match="CLASS_LABELS"):
        check_compatibility(
            FR.make("label_flip", seed=0, rate=0.1), FR, ndim=2, dtype_kind="f", has_classes=False
        )
    with pytest.raises(FaultCompatibilityError, match="class-targeted"):
        check_compatibility(
            FR.make(
                "gaussian_noise", seed=0, sigma=0.1, scope=FaultScope(ScopeKind.CLASS, label=1)
            ),
            FR,
            ndim=2,
            dtype_kind="f",
            has_classes=False,
        )
    compound = FR.compound(noise, FR.make("brightness", seed=0, delta=0.1))
    with pytest.raises(FaultCompatibilityError, match="IMAGE"):  # every leaf is checked
        check_compatibility(compound, FR, ndim=2, dtype_kind="f", has_classes=False)


# --- design and limits ------------------------------------------------------------------------


def base_design(**over: object) -> FaultDesign:
    from experionyx.evaluation.config import EvaluationConfig

    spec = FR.make("gaussian_noise", seed=0, sigma=0.0)
    return FaultDesign(spec.to_dict(), over.pop("evaluation", EvaluationConfig()), **over)  # type: ignore[arg-type]


def test_design_validation_and_round_trip() -> None:
    d = base_design(
        seeds=(1, 2, 3), sweep=SweepSpec("sigma", (0.0, 0.1, 0.2)), primary_metric="accuracy"
    )
    assert (d.points, d.total_runs) == (3, 9)
    assert FaultDesign.from_dict(d.to_dict()) == d
    for bad in (
        lambda: base_design(seeds=()),
        lambda: base_design(seeds=(1, 1)),
        lambda: base_design(seeds=(-1,)),
        lambda: base_design(aggregation_confidence=1.0),
        lambda: SweepSpec("sigma", ()),
        lambda: SweepSpec("sigma", (0.1, 0.1)),
        lambda: SweepSpec("sigma", (float("nan"),)),
        lambda: FaultLimits(max_points=0),
    ):
        with pytest.raises(ValidationError):
            bad()
    with pytest.raises(ValidationError, match="unknown field"):
        FaultDesign.from_dict({**d.to_dict(), "typo": 1})


def test_safety_limits_refuse_experiments_explicitly_instead_of_truncating() -> None:
    with pytest.raises(FaultLimitError, match="max_points"):
        base_design(sweep=SweepSpec("sigma", tuple(float(i) for i in range(21))))
    with pytest.raises(FaultLimitError, match="max_repetitions"):
        base_design(seeds=tuple(range(51)))
    with pytest.raises(FaultLimitError, match="max_total_runs"):
        base_design(
            seeds=tuple(range(20)), sweep=SweepSpec("sigma", tuple(float(i) for i in range(11)))
        )
    with pytest.raises(FaultLimitError, match="max_bootstrap_resamples"):
        base_design(aggregation_resamples=10_000)
    with pytest.raises(FaultLimitError, match="max_total_sample_evaluations"):
        base_design(seeds=(1, 2, 3)).check_sample_budget(10**8)
    relaxed = base_design(
        seeds=tuple(range(60)), limits=FaultLimits(max_repetitions=100, max_total_runs=1000)
    )
    assert (
        relaxed.limits.max_repetitions == 100
    )  # limits are configuration and are stored with the design


# --- registry entities ------------------------------------------------------------------------


def test_fault_experiment_entity_lifecycle_and_identity() -> None:
    from factories import configuration, environment, experiment, investigation, run

    inv, cfg = investigation(), configuration()
    exp = experiment(inv, cfg)
    r = run(exp, environment())
    fx = FaultExperiment(inv.id, "noise", exp.id, r.id, base_design().to_dict(), T0)
    assert fx.id.startswith("fxp_")
    assert fx.status is ExperimentStatus.DRAFT
    assert fx.with_status(ExperimentStatus.READY).id == fx.id
    with pytest.raises(ValidationError):
        fx.with_status(ExperimentStatus.COMPLETED)
    other = FaultExperiment(inv.id, "noise", exp.id, r.id, base_design(seeds=(5,)).to_dict(), T0)
    assert other.id != fx.id  # a different design is a different experiment
    assert FaultExperiment.from_dict(fx.to_dict()) == fx


def test_fault_trial_validation_and_round_trip() -> None:
    fid, family = "fxp_" + "1" * 32, FR.make("gaussian_noise", seed=1, sigma=0.1)
    common = {
        "fault_experiment_id": fid,
        "point_index": 0,
        "repeat_index": 0,
        "seed": 1,
        "fault_id": family.id,
        "family_id": family.family_id,
        "baseline_run_id": "run_" + "2" * 32,
        "created_at": T0,
    }
    done = FaultTrial(status=TrialStatus.COMPLETED, treatment_run_id="run_" + "3" * 32, **common)  # type: ignore[arg-type]
    assert FaultTrial.from_dict(done.to_dict()) == done
    with pytest.raises(ValidationError, match="treatment run"):
        FaultTrial(status=TrialStatus.COMPLETED, **common)  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="reason"):
        FaultTrial(status=TrialStatus.FAILED, **common)  # type: ignore[arg-type]
    FaultTrial(status=TrialStatus.SKIPPED, reason="early termination", **common)  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="go together"):
        FaultTrial(status=TrialStatus.SKIPPED, reason="x", parameter_name="sigma", **common)  # type: ignore[arg-type]


def test_empty_registry_behaviour() -> None:
    empty = FaultRegistry()
    assert empty.names() == ()
    with pytest.raises(FaultNotFoundError):
        empty.make("gaussian_noise", seed=0, sigma=1.0)
