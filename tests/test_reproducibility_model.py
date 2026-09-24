"""Reproducibility domain model: spec identity, entity validation/round-trip, and the generic
tolerance-aware document comparator -- no registry, no execution."""

from datetime import UTC, datetime

import pytest

from experionyx.errors import ValidationError
from experionyx.reproducibility.compare import compare_documents, diff_mappings
from experionyx.reproducibility.entities import ReproductionAttempt
from experionyx.reproducibility.spec import ReproductionSpec
from experionyx.reproducibility.taxonomy import ComparisonOutcome, ReproductionMode, TargetKind

RUN_ID = "run_" + "1" * 32
INV_ID = "inv_" + "1" * 32
T0 = datetime(2026, 9, 24, tzinfo=UTC)


# --- ReproductionSpec -----------------------------------------------------------------------


def test_spec_identity_is_deterministic_and_content_addressed() -> None:
    a = ReproductionSpec(TargetKind.RUN, RUN_ID, ReproductionMode.EXACT)
    b = ReproductionSpec(TargetKind.RUN, RUN_ID, ReproductionMode.EXACT)
    assert a.spec_id == b.spec_id
    assert a.spec_id.startswith("rsc_")


def test_spec_identity_changes_with_mode() -> None:
    a = ReproductionSpec(TargetKind.RUN, RUN_ID, ReproductionMode.EXACT)
    b = ReproductionSpec(TargetKind.RUN, RUN_ID, ReproductionMode.DETERMINISTIC)
    assert a.spec_id != b.spec_id


def test_spec_identity_changes_with_tolerance() -> None:
    a = ReproductionSpec(TargetKind.RUN, RUN_ID, ReproductionMode.NUMERIC_TOLERANCE, relative_tolerance=1e-6)  # fmt: skip
    b = ReproductionSpec(TargetKind.RUN, RUN_ID, ReproductionMode.NUMERIC_TOLERANCE, relative_tolerance=1e-3)  # fmt: skip
    assert a.spec_id != b.spec_id


def test_spec_rejects_a_target_id_with_the_wrong_prefix() -> None:
    with pytest.raises(ValidationError):
        ReproductionSpec(TargetKind.RUN, "brs_" + "1" * 32, ReproductionMode.EXACT)


def test_spec_rejects_negative_tolerance() -> None:
    with pytest.raises(ValidationError):
        ReproductionSpec(TargetKind.RUN, RUN_ID, ReproductionMode.NUMERIC_TOLERANCE, relative_tolerance=-1.0)  # fmt: skip


def test_spec_round_trips_through_dict() -> None:
    spec = ReproductionSpec(TargetKind.GRAPH_SNAPSHOT, "gsn_" + "2" * 32, ReproductionMode.STATISTICAL, relative_tolerance=0.01)  # fmt: skip
    again = ReproductionSpec.from_dict(spec.to_dict())
    assert again.spec_id == spec.spec_id


def test_spec_from_dict_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        ReproductionSpec.from_dict({"target_kind": "RUN", "target_id": RUN_ID, "mode": "EXACT", "bogus": 1})  # fmt: skip


# --- ReproductionAttempt ---------------------------------------------------------------------


def _attempt(**over: object) -> ReproductionAttempt:
    base: dict[str, object] = dict(
        target_kind=TargetKind.RUN,
        target_id=RUN_ID,
        spec_id="rsc_" + "1" * 32,
        spec={},
        attempt=0,
        investigation_id=INV_ID,
        run_id=RUN_ID,
        replay_run_id=None,
        replay_status=None,
        secondary_ref=None,
        outcome=ComparisonOutcome.EQUAL,
        sources_changed=False,
        differences=(),
        field_differences=(),
        environment_diff={},
        artifact_verification={},
        note=None,
        engine_version="1.0.0",
        created_at=T0,
    )
    base.update(over)
    return ReproductionAttempt(**base)  # type: ignore[arg-type]


def test_attempt_identity_is_target_and_attempt_number() -> None:
    a = _attempt(attempt=0)
    b = _attempt(attempt=1)
    assert a.id != b.id
    assert _attempt(attempt=0).id == a.id  # same target+attempt -> same id regardless of outcome


def test_attempt_round_trips_through_dict() -> None:
    a = _attempt(outcome=ComparisonOutcome.DIFFERENT, differences=("results",))
    again = ReproductionAttempt.from_dict(a.to_dict())
    assert again.id == a.id
    assert again.differences == ("results",)


def test_attempt_run_id_may_be_none_for_targets_without_one() -> None:
    a = _attempt(run_id=None, replay_run_id=None)
    assert a.run_id is None


def test_attempt_rejects_non_boolean_sources_changed() -> None:
    with pytest.raises(ValidationError):
        _attempt(sources_changed=1)  # type: ignore[arg-type]


def test_attempt_rejects_a_run_id_with_the_wrong_prefix() -> None:
    with pytest.raises(ValidationError):
        _attempt(run_id="brs_" + "1" * 32)


# --- generic comparator ----------------------------------------------------------------------


def test_compare_documents_identical() -> None:
    outcome, diffs = compare_documents({"a": 1}, {"a": 1}, relative_tolerance=1e-6, absolute_tolerance=1e-9)  # fmt: skip
    assert outcome is ComparisonOutcome.EQUAL and diffs == ()


def test_compare_documents_within_tolerance_is_approximately_equal() -> None:
    outcome, diffs = compare_documents(
        {"acc": 0.900000001}, {"acc": 0.900000002}, relative_tolerance=1e-6, absolute_tolerance=1e-9
    )
    assert outcome is ComparisonOutcome.APPROXIMATELY_EQUAL
    assert diffs[0].path == "$.acc"


def test_compare_documents_outside_tolerance_is_different() -> None:
    outcome, _ = compare_documents({"acc": 0.9}, {"acc": 0.5}, relative_tolerance=1e-6, absolute_tolerance=1e-9)  # fmt: skip
    assert outcome is ComparisonOutcome.DIFFERENT


def test_compare_documents_type_mismatch_is_incomparable() -> None:
    outcome, diffs = compare_documents({"a": 1}, {"a": "1"}, relative_tolerance=1e-6, absolute_tolerance=1e-9)  # fmt: skip
    assert outcome is ComparisonOutcome.INCOMPARABLE
    assert diffs[0].outcome is ComparisonOutcome.INCOMPARABLE


def test_compare_documents_nested_lists_and_dicts() -> None:
    a = {"rows": [{"x": 1}, {"x": 2}]}
    b = {"rows": [{"x": 1}, {"x": 3}]}
    outcome, diffs = compare_documents(a, b, relative_tolerance=1e-6, absolute_tolerance=1e-9)
    assert outcome is ComparisonOutcome.DIFFERENT
    assert diffs[0].path == "$.rows[1].x"


def test_compare_documents_missing_key_is_different() -> None:
    outcome, diffs = compare_documents({"a": 1, "b": 2}, {"a": 1}, relative_tolerance=1e-6, absolute_tolerance=1e-9)  # fmt: skip
    assert outcome is ComparisonOutcome.DIFFERENT
    assert diffs[0].path == "$.b"


def test_diff_mappings_reports_changed_added_removed() -> None:
    out = diff_mappings({"x": 1, "y": 2}, {"x": 9, "z": 3})
    assert out["changed"] == {"x": {"original": 1, "reproduced": 9}}
    assert out["only_in_original"] == ["y"]
    assert out["only_in_reproduced"] == ["z"]
