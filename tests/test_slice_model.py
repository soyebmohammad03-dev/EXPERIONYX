"""The slice model: identity, the safe condition language, three-valued membership, and the pure
slice statistics. Synthetic, hand-checkable tables; no registry."""

import math
import random
from dataclasses import replace
from typing import Any

import pytest

from experionyx.adapters.capabilities import TaskType
from experionyx.errors import ValidationError
from experionyx.evaluation.analysis import slice_indices
from experionyx.evaluation.config import SliceCondition, SliceKind
from experionyx.evaluation.config import SliceSpec as EvalSliceSpec
from experionyx.slices import analysis as an
from experionyx.slices.data import Baseline
from experionyx.slices.entities import Slice
from experionyx.slices.evaluate import (
    DuplicateSampleError,
    MembershipStatus,
    build_table,
    evaluate,
)
from experionyx.slices.spec import (
    Condition,
    SliceSpec,
    all_of,
    any_of,
    between,
    eq,
    is_true,
    isin,
    label,
    negate,
)

NOW = __import__("datetime").datetime(2026, 1, 1, tzinfo=__import__("datetime").UTC)


def spec(cond: Condition, name: str = "s") -> SliceSpec:
    return SliceSpec(name, cond)


# -- identity ------------------------------------------------------------------------------------------------


def test_equivalent_spellings_share_one_identity_and_name_is_not_identity() -> None:
    a = all_of(eq("feature:x", 1.0), isin("target", [2, 1]), between("feature:y", 0, 5))
    b = all_of(
        between("feature:y", 0.0, 5.0),
        all_of(isin("target", [1, 2]), eq("feature:x", 1)),
        eq("feature:x", 1),
    )  # reordered, nested, duplicated, int vs float
    assert a == b and spec(a, "one").slice_id == spec(b, "another").slice_id
    assert isin("f", ["a"]) == eq("f", "a")  # a one-value IN is an EQ
    assert negate(negate(eq("f", 1))) == eq("f", 1)
    assert any_of(eq("f", 1)) == eq("f", 1) and label(3) == eq("target", 3)
    assert (
        SliceSpec("n", a, "custom text").slice_id == spec(a).slice_id
    )  # descriptions are labels too
    assert spec(a).slice_id.startswith("sls_") and len(spec(a).slice_id) == 36


def test_changing_any_part_of_the_condition_changes_the_identity() -> None:
    base = spec(all_of(eq("feature:x", 1), between("feature:y", 0, 5))).slice_id
    variants = [
        all_of(eq("feature:x", 2), between("feature:y", 0, 5)),
        all_of(eq("feature:x", 1), between("feature:y", 0, 6)),
        all_of(eq("feature:x", 1), between("feature:y", 0, 5, high_inclusive=True)),
        any_of(eq("feature:x", 1), between("feature:y", 0, 5)),
        all_of(eq("feature:z", 1), between("feature:y", 0, 5)),
        negate(all_of(eq("feature:x", 1), between("feature:y", 0, 5))),
    ]
    ids = {spec(v).slice_id for v in variants}
    assert base not in ids and len(ids) == len(variants)
    assert spec(eq("f", True)).slice_id != spec(eq("f", 1)).slice_id  # bool is not the number 1
    assert spec(eq("f", "1")).slice_id != spec(eq("f", 1)).slice_id


def test_registered_slice_id_equals_the_specs_identity_and_round_trips() -> None:
    s = spec(all_of(label(1), negate(between("feature:x", 2, None))), "hard")
    rec = Slice.of(s, NOW)
    assert rec.id == s.slice_id and Slice.from_dict(rec.to_dict()) == rec
    assert SliceSpec.from_dict(s.to_dict()) == s
    again = Slice.of(spec(all_of(negate(between("feature:x", 2, None)), label(1)), "renamed"), NOW)
    assert again.id == rec.id  # equivalent definition: one logical record
    assert s.human == "(NOT (2 <= feature:x)) AND (target == 1)" and s.fields == (
        "feature:x",
        "target",
    )


def test_the_condition_language_is_closed_and_typed() -> None:
    for bad in (
        {"op": "eval", "field": "x", "values": ["1+1"]},
        {"op": "eq", "field": "x", "values": [1], "extra": 1},
        {"op": "eq", "field": "x", "values": [1, 2]},
        {"op": "eq", "field": " x", "values": [1]},
        {"op": "in", "field": "x", "values": []},
        {"op": "in", "field": "x", "values": "abc"},
        {"op": "range", "field": "x"},
        {"op": "range", "field": "x", "low": 5, "high": 1},
        {"op": "range", "field": "x", "low": 1, "high": 1},
        {"op": "range", "field": "x", "low": math.nan},
        {"op": "eq", "field": "x", "values": [math.inf]},
        {"op": "eq", "field": "x", "values": [[1]]},
        {"op": "and", "args": []},
        {"op": "and", "args": ["x"]},
        {"op": "not", "args": [{"op": "is_true", "field": "a"}, {"op": "is_true", "field": "b"}]},
        {"op": "is_true"},
        "x > 1",
    ):
        with pytest.raises((ValidationError, AttributeError)):
            Condition.from_dict(bad)  # type: ignore[arg-type]
    deep: dict[str, Any] = {"op": "is_true", "field": "f"}
    for _ in range(40):
        deep = {"op": "and", "args": [deep, {"op": "is_true", "field": "g"}]}
    with pytest.raises(ValidationError, match="nest"):
        Condition.from_dict(deep)
    # a string that looks like code is just a category value
    c = Condition.from_dict(
        {"op": "eq", "field": "name", "values": ["__import__('os').system('x')"]}
    )
    assert c.values == ("__import__('os').system('x')",)
    with pytest.raises(ValidationError):
        SliceSpec("  ", eq("f", 1))
    with pytest.raises(ValidationError, match="schema"):
        SliceSpec.from_dict({**spec(eq("f", 1)).to_dict(), "slice_schema": 99})


def test_outcome_fields_make_a_slice_non_static() -> None:
    assert spec(all_of(label(1), eq("feature:x", 1))).static
    assert not spec(eq("correct", False)).static and not spec(between("confidence", 0, 0.5)).static
    assert not spec(eq("predicted", 1)).static


# -- membership ---------------------------------------------------------------------------------------------


def table() -> dict[Any, dict[str, Any]]:
    rows = {
        1: {"target": 0, "feature:x": 1.0, "feature:g": "a", "flag": True},
        2: {"target": 1, "feature:x": 2.0, "feature:g": "b", "flag": False},
        3: {"target": 1, "feature:x": 5.0, "feature:g": "a", "flag": True},
        4: {"target": 2, "feature:x": 9.0, "feature:g": "c", "flag": False},
        5: {"target": 1, "feature:x": 5.5, "feature:g": "b", "flag": True},
    }
    return build_table(rows.items())  # type: ignore[return-value]


def ids_of(cond: Condition) -> tuple[Any, ...]:
    return evaluate(spec(cond), table()).sample_ids


def test_membership_ground_truth_for_every_operator() -> None:
    assert ids_of(eq("feature:g", "a")) == (1, 3)
    assert ids_of(isin("feature:g", ["b", "c"])) == (2, 4, 5)
    assert ids_of(is_true("flag")) == (1, 3, 5)
    assert ids_of(label(1)) == (2, 3, 5)
    assert ids_of(between("feature:x", 2, 5.5)) == (2, 3)  # half-open [2, 5.5)
    assert ids_of(between("feature:x", 2, 5.5, high_inclusive=True)) == (2, 3, 5)
    assert ids_of(between("feature:x", 2, 5.5, low_inclusive=False)) == (3,)
    assert ids_of(between("feature:x", None, 2)) == (1,) and ids_of(
        between("feature:x", 9, None)
    ) == (4,)
    assert ids_of(all_of(label(1), eq("feature:g", "b"))) == (2, 5)
    assert ids_of(any_of(label(0), label(2))) == (1, 4)
    assert ids_of(negate(label(1))) == (1, 4)
    assert ids_of(
        all_of(negate(eq("feature:g", "a")), any_of(is_true("flag"), between("feature:x", 8, None)))
    ) == (4, 5)
    m = evaluate(spec(label(1)), table())
    assert (m.n_members, m.n_total, m.prevalence) == (
        3,
        5,
        0.6,
    ) and m.status is MembershipStatus.COMPUTED


def test_missing_metadata_is_never_a_match_and_is_reported() -> None:
    t = build_table(
        [
            (1, {"target": 1, "x": 1.0}),
            (2, {"target": 1}),  # no x
            (3, {"target": 1, "x": None}),
            (4, {"target": 1, "x": math.nan}),
            (5, {"target": 1, "x": math.inf}),
            (6, {"target": 1, "x": "high"}),  # wrong kind for a numeric range
            (7, {"target": 0, "x": 8.0}),
        ]
    )
    m = evaluate(spec(between("x", 0, 5)), t)
    assert m.sample_ids == (1,) and m.n_unknown == 5 and m.unknown_sample_ids == (2, 3, 4, 5, 6)
    assert any("unknown" in w for w in m.warnings) and m.status is MembershipStatus.COMPUTED
    assert evaluate(spec(negate(between("x", 0, 5))), t).sample_ids == (
        7,
    )  # NOT unknown is unknown
    assert (
        evaluate(spec(all_of(label(0), between("x", 0, 5))), t).sample_ids == ()
    )  # false dominates unknown
    dec = evaluate(spec(all_of(label(1), between("x", 0, 5))), t)
    assert dec.n_unknown == 5  # target says maybe, x unknown
    assert evaluate(spec(any_of(label(0), between("x", 0, 5))), t).sample_ids == (
        1,
        7,
    )  # true dominates
    tri = evaluate(spec(is_true("x")), t)
    assert tri.sample_ids == () and tri.n_unknown == 7  # no boolean anywhere: nothing matches


def test_a_field_absent_from_every_sample_is_undefined_not_empty() -> None:
    m = evaluate(spec(eq("feature:nope", 1)), table())
    assert (
        m.status is MembershipStatus.MISSING_FIELD
        and m.sample_ids == ()
        and m.missing_fields == ("feature:nope",)
    )
    assert m.n_unknown == 5 and "undefined, not empty" in str(m.reason)
    assert evaluate(spec(label(1)), build_table([])).status is MembershipStatus.NO_SAMPLES


def test_empty_slices_and_unknown_categories() -> None:
    m = evaluate(spec(all_of(label(0), label(1))), table())
    assert m.status is MembershipStatus.EMPTY and m.n_members == 0 and m.prevalence == 0.0
    u = evaluate(spec(eq("feature:g", "zzz")), table())
    assert u.status is MembershipStatus.EMPTY and u.unseen_values == {"feature:g": ("zzz",)}
    assert any("never observed" in w for w in u.warnings)
    assert evaluate(spec(isin("feature:g", ["a", "zzz"])), table()).unseen_values == {
        "feature:g": ("zzz",)
    }
    # kinds are strict: 1 is not True, "1" is not 1
    t = build_table([(1, {"f": True}), (2, {"f": 1}), (3, {"f": "1"})])
    assert evaluate(spec(eq("f", 1)), t).sample_ids == (2,) and evaluate(
        spec(eq("f", True)), t
    ).sample_ids == (1,)
    assert (
        evaluate(spec(eq("f", "1")), t).sample_ids == (3,)
        and evaluate(spec(eq("f", 1)), t).n_unknown == 2
    )


def test_duplicate_and_invalid_sample_ids_are_refused() -> None:
    with pytest.raises(DuplicateSampleError, match="duplicate"):
        build_table([(1, {"a": 1}), (2, {"a": 2}), (1, {"a": 3})])
    for bad in (1.5, None, True, (1, 2)):
        with pytest.raises(ValidationError):
            build_table([(bad, {"a": 1})])  # type: ignore[list-item]


def test_membership_does_not_depend_on_row_order() -> None:
    rows = [
        (i, {"target": i % 3, "feature:x": float(i % 7), "feature:g": "ab"[i % 2]})
        for i in range(60)
    ]
    s = spec(
        all_of(any_of(label(1), label(2)), negate(eq("feature:g", "a")), between("feature:x", 1, 6))
    )
    want = evaluate(s, build_table(rows))
    for seed in range(5):
        shuffled = rows[:]
        random.Random(seed).shuffle(shuffled)
        got = evaluate(s, build_table(shuffled))
        assert got == want and got.membership_digest == want.membership_digest
    assert list(want.sample_ids) == sorted(want.sample_ids)
    other = evaluate(spec(all_of(label(1), between("feature:x", 1, 6))), build_table(rows))
    assert (
        other.membership_digest != want.membership_digest
    )  # a different member set has a different digest


def test_evaluation_slices_and_the_new_engine_select_the_same_samples() -> None:
    """The Phase 4 slice model is a special case (AND of conditions): both must agree."""
    y_true = [0, 1, 1, 0, 2, 1, 2, 0]
    y_pred = [0, 1, 2, 0, 2, 0, 2, 1]
    x = [0.5, 1.0, 2.0, 3.0, 4.0, 0.0, 2.5, 9.0]
    old = EvalSliceSpec(
        "old",
        (
            SliceCondition(SliceKind.TARGET_EQUALS, value=1),
            SliceCondition(SliceKind.FEATURE_RANGE, field="x", low=0.5, high=2.0),
        ),
    )
    idx = slice_indices(old, y_true, y_pred, {"x": x})
    tbl = build_table((i, {"target": y_true[i], "feature:x": x[i]}) for i in range(8))
    new = evaluate(
        spec(all_of(label(1), between("feature:x", 0.5, 2.0))), tbl
    )  # [low, high) like FEATURE_RANGE
    assert list(new.sample_ids) == idx == [1]
    pred_old = slice_indices(
        EvalSliceSpec("p", (SliceCondition(SliceKind.PREDICTED_EQUALS, value=2),)),
        y_true,
        y_pred,
        {},
    )
    tbl2 = build_table((i, {"predicted": y_pred[i]}) for i in range(8))
    assert list(evaluate(spec(eq("predicted", 2)), tbl2).sample_ids) == pred_old


# -- pure slice statistics -----------------------------------------------------------------------------------------


def baseline(correct: list[bool], task: TaskType = TaskType.CLASSIFICATION) -> Baseline:
    rows = {
        i: {
            "index": i,
            "true": 1,
            "predicted": 1 if c else 0,
            "correct": c,
            "confidence": 0.9,
            "scores": [0.1, 0.9] if c else [0.9, 0.1],
        }
        for i, c in enumerate(correct)
    }
    return Baseline(
        "run_" + "0" * 32,
        task,
        (0, 1),
        "test",
        "dst_" + "0" * 32,
        "sha256:" + "0" * 64,
        "sha256:" + "1" * 64,
        rows,
    )


def test_slice_vs_rest_and_population_ground_truth() -> None:
    b = baseline(
        [True] * 8 + [False] * 2 + [True] * 6 + [False] * 4
    )  # ids 0-9: 80%, ids 10-19: 60%
    cfg = an.SliceConfig(resamples=200)
    a_ids, rest = list(range(10)), list(range(10, 20))
    c = an.compare_groups(b, a_ids, rest, cfg)
    assert (c["mean_a"], c["mean_b"], c["n_a"], c["n_b"]) == (0.8, 0.6, 10, 10)
    assert c["descriptive"]["difference"] == pytest.approx(0.2)
    assert c["descriptive"]["relative_difference"] == pytest.approx(0.2 / 0.6)
    inf = c["inference"]
    assert (
        inf["pairing"] == "UNPAIRED" and inf["test"]["exact"] is False
    )  # C(20,10) > the exact limit: seeded Monte Carlo
    md = next(e for e in inf["effects"] if e["name"] == "mean_difference")
    assert md["value"] == pytest.approx(0.2) and "better or worse overall" in c["direction_note"]
    pop = an.compare_to_population(b, a_ids, cfg)
    assert pop["descriptive_vs_population"]["difference"] == pytest.approx(
        0.8 - 0.7
    )  # population accuracy 14/20
    assert pop["vs_rest"]["descriptive"]["difference"] == pytest.approx(0.2)
    assert pop["n_slice"] == 10 and pop["n_population"] == 20


def test_overlapping_and_tiny_groups_are_not_tested() -> None:
    b = baseline([True, False] * 10)
    cfg = an.SliceConfig(resamples=100)
    ov = an.compare_groups(b, list(range(0, 8)), list(range(4, 12)), cfg)
    assert ov["status"] == "UNDEFINED" and ov["inference"] is None and "share 4" in ov["reason"]
    assert ov["descriptive"]["difference"] is not None  # the raw difference is still reported
    tiny = an.compare_groups(b, [0, 1], list(range(2, 20)), cfg)
    assert tiny["status"] == an.INSUFFICIENT and "min_members" in tiny["reason"]
    zero = an.compare_groups(
        baseline([False] * 6 + [True] * 6),
        list(range(6, 12)),
        list(range(6)),
        an.SliceConfig(resamples=100),
    )
    assert (
        zero["descriptive"]["relative_difference"] is None
        and "0" in zero["descriptive"]["relative_reason"]
    )


def test_slice_metrics_reuse_the_metric_engine_and_flag_small_slices() -> None:
    b = baseline([True] * 6 + [False] * 2)
    m = an.slice_metrics(b, list(range(8)), an.SliceConfig(resamples=200))
    acc = next(x for x in m["metrics"] if x["metric_id"] == "accuracy")
    assert (
        acc["value"] == 0.75
        and acc["status"] == "COMPUTED"
        and acc["interval"]["status"] == "COMPUTED"
    )
    assert (
        acc["interval"]["seed"] == 0
        and acc["interval"]["resamples"] == 200
        and acc["interval"]["confidence"] == 0.95
    )
    assert m["errors"]["n_errors"] == 2 and m["errors"]["errors_by_true_class"] == {"1": 2}
    assert m["calibration"]["status"] == "COMPUTED" and m["latency"]["status"] == "UNAVAILABLE"
    assert m["evidence"] == "SUFFICIENT"
    small = an.slice_metrics(b, [0, 1, 2], an.SliceConfig(resamples=100))
    assert small["evidence"] == an.INSUFFICIENT and "min_members" in small["reason"]
    noscores = replace(
        b, rows={i: {k: v for k, v in r.items() if k != "scores"} for i, r in b.rows.items()}
    )
    assert (
        an.slice_metrics(noscores, list(range(8)), an.SliceConfig())["calibration"]["status"]
        == "UNAVAILABLE"
    )


def test_regression_slices_use_mae_and_multiple_comparison_configuration() -> None:
    rows = {
        i: {
            "index": i,
            "true": float(i),
            "predicted": float(i) + (0.5 if i < 6 else 2.0),
            "correct": None,
            "confidence": None,
        }
        for i in range(12)
    }
    b = Baseline(
        "run_" + "0" * 32,
        TaskType.REGRESSION,
        None,
        "test",
        None,
        "sha256:" + "0" * 64,
        "sha256:" + "1" * 64,
        rows,
    )
    assert an.measure_of(b.task) == ("mae", False)
    c = an.compare_groups(b, list(range(6)), list(range(6, 12)), an.SliceConfig(resamples=100))
    assert (c["mean_a"], c["mean_b"]) == (0.5, 2.0) and c["descriptive"]["difference"] == -1.5
    comps: dict[str, dict[str, Any]] = {
        "a vs b": {"inference": {"test": {"p_value": 0.01}}},
        "a vs c": {"inference": {"test": {"p_value": 0.04}}},
        "x": {"inference": None},
    }
    rec = an.correct_family(comps, an.SliceConfig(correction="BONFERRONI", alpha=0.05))
    assert rec["n_hypotheses"] == 2 and rec["excluded"] == ["x"]
    assert comps["a vs b"]["multiplicity"]["adjusted_p"] == pytest.approx(0.02)
    assert comps["a vs b"]["multiplicity"]["significant_after_correction"] is True
    assert comps["a vs c"]["multiplicity"]["significant_after_correction"] is False
    none: dict[str, dict[str, Any]] = {"p": {"inference": {"test": {"p_value": 0.01}}}}
    an.correct_family(none, an.SliceConfig())
    assert (
        none["p"]["multiplicity"]["significant_after_correction"] is None
    )  # no correction requested: no claim


def test_slice_config_is_strict() -> None:
    assert an.SliceConfig.from_dict(
        an.SliceConfig(min_members=3, metrics=("accuracy", "f1")).to_dict()
    ) == an.SliceConfig(min_members=3, metrics=("accuracy", "f1"))
    for bad in (
        {"min_members": 0},
        {"min_trials": 1},
        {"confidence": 1.0},
        {"method": "x"},
        {"correction": "holm"},
        {"metrics": ("z", "a")},
        {"nope": 1},
    ):
        with pytest.raises(ValidationError):
            an.SliceConfig.from_dict(bad)
