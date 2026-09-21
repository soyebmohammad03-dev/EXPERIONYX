"""Profile vocabulary, spec identity, entities and comparison rules (no engines needed)."""

import json
from typing import Any

import pytest

from experionyx.errors import ProfileRefusal, ValidationError
from experionyx.failures.taxonomy import Predicate
from experionyx.reliability.compare import compare_profiles
from experionyx.reliability.entities import ReliabilityProfile, ReliabilityReference
from experionyx.reliability.spec import ProfileSpec
from experionyx.reliability.taxonomy import Dimension, DimensionStatus, RefKind, Scope
from failure_helpers import NOW, rid

FP = "sha256:" + "a" * 64
FP2 = "sha256:" + "b" * 64


def spec(**kw):  # type: ignore[no-untyped-def]
    base = {
        "scope": Scope.MODEL_DATASET,
        "baseline_run": rid("run", 1),
        "fault_experiments": (rid("fxp", 1), rid("fxp", 2)),
    }
    base.update(kw)
    return ProfileSpec(**base)


def test_vocabulary_has_the_required_dimensions_statuses_and_scopes_and_no_cross_dataset_scope() -> (
    None
):
    assert {d.value for d in Dimension} == {
        "BASELINE_PERFORMANCE",
        "FAULT_SENSITIVITY",
        "FAILURE_PREVALENCE",
        "FAILURE_SEVERITY",
        "INTERACTION_SENSITIVITY",
        "SLICE_SENSITIVITY",
        "DISTRIBUTION_SHIFT",
        "MODEL_STRESS",
        "CALIBRATION_ANALYSIS",
        "RESOURCE_SYSTEM",
        "REPRODUCIBILITY",
        "UNCERTAINTY",
        "LATENCY",
        "CALIBRATION",
    }
    assert {s.value for s in DimensionStatus} == {
        "OBSERVED",
        "DERIVED",
        "UNAVAILABLE",
        "INSUFFICIENT_EVIDENCE",
    }
    assert {s.value for s in Scope} == {
        "MODEL_DATASET",
        "MODEL_DATASET_SPLIT",
        "MODEL_DATASET_EVALUATION",
    }  # cross-dataset aggregation is deliberately absent
    assert "CAUSES" not in {p.value for p in Predicate}


def test_spec_identity_is_deterministic_order_insensitive_and_sensitive() -> None:
    s = spec()
    assert s.spec_id == spec().spec_id == ProfileSpec.from_dict(s.to_dict()).spec_id
    assert (
        spec(fault_experiments=(rid("fxp", 2), rid("fxp", 1))).spec_id == s.spec_id
    )  # source order is not identity
    for changed in (
        spec(scope=Scope.MODEL_DATASET_SPLIT),
        spec(baseline_run=rid("run", 2)),
        spec(fault_experiments=(rid("fxp", 1),)),
        spec(interactions=(rid("ian", 1),)),
        spec(failure_modes=(rid("fmd", 1),)),
    ):
        assert changed.spec_id != s.spec_id


def test_spec_validation_is_strict() -> None:
    with pytest.raises(ValidationError):
        spec(baseline_run="run_x")
    with pytest.raises(ValidationError):
        spec(fault_experiments=(rid("run", 1),))  # wrong kind of ID
    with pytest.raises(ValidationError):
        ProfileSpec.from_dict({**spec().to_dict(), "surprise": 1})
    with pytest.raises(ValidationError, match="unknown scope"):
        ProfileSpec.from_dict({**spec().to_dict(), "scope": "CROSS_DATASET"})
    with pytest.raises(ValidationError):
        ProfileSpec.from_dict({"scope": "MODEL_DATASET"})  # no baseline


def profile(**kw):  # type: ignore[no-untyped-def]
    base = {
        "investigation_id": rid("inv", 1),
        "run_id": rid("run", 9),
        "spec_id": spec().spec_id,
        "spec": spec().to_dict(),
        "scope": Scope.MODEL_DATASET,
        "model_fingerprint": FP,
        "dataset_fingerprint": FP,
        "split": "*",
        "evaluation_config_hash": FP,
        "provenance_fingerprint": FP,
        "dimension_status": {d.value: "OBSERVED" for d in Dimension},
        "summary": {},
        "created_at": NOW,
    }
    base.update(kw)
    return ReliabilityProfile(**base)


def test_profile_entity_identity_roundtrip_and_validation() -> None:
    p = profile()
    assert ReliabilityProfile.from_dict(p.to_dict()) == p
    assert (
        profile(created_at=NOW.replace(year=2030), provenance_fingerprint=FP2).id == p.id
    )  # identity is investigation + spec only
    assert profile(spec_id="rsp_other").id != p.id
    with pytest.raises(ValidationError):
        profile(dimension_status={"BASELINE_PERFORMANCE": "GOOD"})  # a made-up status
    with pytest.raises(ValueError, match="NOT_A_DIMENSION"):
        profile(dimension_status={"NOT_A_DIMENSION": "OBSERVED"})
    r = ReliabilityReference(
        p.id, Dimension.FAULT_SENSITIVITY, RefKind.FAULT_EXPERIMENT, rid("fxp", 1), "n", NOW
    )
    assert ReliabilityReference.from_dict(r.to_dict()) == r


def doc(
    model: str = FP,
    dataset: str = FP,
    scope: str = "MODEL_DATASET_EVALUATION",
    split: str = "test",
    cfg: str = FP,
    acc: float = 0.9,
    det: float = 0.1,
    contrast: float = 0.05,
) -> dict[str, Any]:
    dims: dict[str, dict[str, Any]] = {
        d.value: {"status": "OBSERVED", "observations": []} for d in Dimension
    }
    dims["BASELINE_PERFORMANCE"]["observations"] = [
        {
            "metrics": [
                {"metric_id": "accuracy", "value": acc, "higher_is_better": True, "interval": None},
                {
                    "metric_id": "log_loss",
                    "value": 0.3,
                    "higher_is_better": False,
                    "interval": None,
                },
            ]
        }
    ]
    dims["FAULT_SENSITIVITY"]["observations"] = [
        {
            "fault": {
                "type": "gaussian_noise",
                "version": "1",
                "parameters": {"sigma": 1.0},
                "scope": {},
                "components": [],
            },
            "primary_metric": "accuracy",
            "baseline_value": acc,
            "points": [
                {
                    "parameter": None,
                    "value": None,
                    "completed": 3,
                    "deterioration": {"mean": det, "ci_lower": det - 0.01, "ci_upper": det + 0.01},
                    "effect_classification": "MEASURED_DEGRADATION",
                }
            ],
        }
    ]
    dims["FAILURE_PREVALENCE"]["observations"] = [
        {
            "source": {"id": rid("fmd", 1)},
            "category": "INPUT_SENSITIVITY",
            "classes": [],
            "slices": [],
            "fault_types": ["gaussian_noise"],
            "lifecycle_state": "CANDIDATE",
            "prevalence": {"fraction_of_runs": 0.5},
        }
    ]
    dims["FAILURE_SEVERITY"]["observations"] = [
        {"source": {"id": rid("fmd", 1)}, "severity": {"magnitude_mean": 0.2}}
    ]
    dims["INTERACTION_SENSITIVITY"]["observations"] = [
        {
            "source": {"id": rid("ian", 1)},
            "fault_a": {"type": "a"},
            "fault_b": {"type": "b"},
            "components": ["a", "b"],
            "metric": "accuracy",
            "interaction_contrast": contrast,
            "interval": None,
            "label": "POSSIBLE_INTERACTION",
            "lifecycle_state": "DISCOVERED",
            "order_dependent": False,
        }
    ]
    dims["SLICE_SENSITIVITY"]["observations"] = [
        {
            "classes": [{"class": "0", "recall": 0.9, "support": 10}],
            "slices": {"status": "UNAVAILABLE"},
        }
    ]
    dims["CALIBRATION"]["observations"] = [{"source": {}, "ece": 0.1}]
    dims["LATENCY"]["observations"] = [{"source": {}, "median_batch_seconds": 0.01}]
    dims["REPRODUCIBILITY"]["observations"] = [{"unresolved_issues": []}]
    dims["UNCERTAINTY"]["observations"] = [{"intervals": [], "caveats": []}]
    return {
        "spec_id": rid("run", 1),
        "scope": scope,
        "provenance_fingerprint": FP,
        "context": {
            "model_fingerprint": model,
            "dataset_fingerprint": dataset,
            "split": split,
            "evaluation_config_hash": cfg,
        },
        "dimensions": dims,
        "dimension_status": {d: v["status"] for d, v in dims.items()},
    }


def test_comparison_reports_raw_differences_and_no_winner() -> None:
    a, b = (
        doc(model=FP, acc=0.9, det=0.1, contrast=0.05),
        doc(model=FP2, acc=0.8, det=0.3, contrast=0.2),
    )
    out = compare_profiles(a, b)
    assert out["same_model"] is False
    m = {x["metric_id"]: x for x in out["baseline"]["metrics"]}
    assert (
        m["accuracy"]["difference"] == pytest.approx(-0.1)
        and m["log_loss"]["difference"] == 0.0
        and m["accuracy"]["higher_is_better"] is True
    )
    (f,) = out["fault_responses"]["matched"]
    assert f["points"][0]["deterioration_mean"]["difference"] == pytest.approx(0.2)
    (i,) = out["interactions"]["matched"]
    assert i["contrast"]["difference"] == pytest.approx(0.15)
    (fm,) = out["failure_modes"]["matched"]
    assert fm["prevalence_fraction"]["difference"] == 0.0
    text = (
        json.dumps(out).lower().replace("higher_is_better", "")
    )  # metric-direction metadata is not a verdict
    disclaimer = "no overall ranking, winner or verdict is computed"
    text = text.replace(disclaimer, "")
    for banned in ("winner", "ranking", "better", "worse", "best", "worst", "score", "verdict"):
        assert banned not in text  # only the disclaimer may mention them
    assert out["note"].startswith("raw differences")
    same = compare_profiles(a, a)
    assert (
        all(x["difference"] == 0 for x in same["baseline"]["metrics"])
        and same["same_model"] is True
    )


def test_unmatched_faults_and_modes_are_listed_not_forced_together() -> None:
    a, b = doc(), doc()
    b["dimensions"]["FAULT_SENSITIVITY"]["observations"][0]["fault"]["parameters"] = {
        "sigma": 2.0
    }  # a different parameter is a different fault
    b["dimensions"]["FAILURE_PREVALENCE"]["observations"][0]["category"] = "LABEL_ERROR"
    out = compare_profiles(a, b)
    assert (
        out["fault_responses"]["matched"] == []
        and len(out["fault_responses"]["only_in_a"]) == 1
        and len(out["fault_responses"]["only_in_b"]) == 1
    )
    assert (
        out["failure_modes"]["matched"] == []
        and out["failure_modes"]["only_in_a"]
        and out["failure_modes"]["only_in_b"]
    )


@pytest.mark.parametrize(
    ("b_kw", "code"),
    [
        ({"dataset": FP2}, "DATASET_MISMATCH"),
        ({"scope": "MODEL_DATASET"}, "SCOPE_MISMATCH"),
        ({"split": "val"}, "SPLIT_MISMATCH"),
        ({"cfg": FP2}, "EVALUATION_CONFIG_MISMATCH"),
    ],
)
def test_incompatible_profiles_are_refused_with_the_reason(b_kw, code) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ProfileRefusal) as exc:
        compare_profiles(doc(), doc(**b_kw))
    assert code in {i.code for i in exc.value.issues}  # type: ignore[attr-defined]


def test_a_looser_scope_does_not_require_the_split_or_config_to_match() -> None:
    assert compare_profiles(
        doc(scope="MODEL_DATASET"), replace_ctx(doc(scope="MODEL_DATASET"), split="val", cfg=FP2)
    )["note"]


def replace_ctx(d, **kw):  # type: ignore[no-untyped-def]
    d["context"] = {
        **d["context"],
        **({"split": kw["split"]} if "split" in kw else {}),
        **({"evaluation_config_hash": kw["cfg"]} if "cfg" in kw else {}),
    }
    return d
