"""Data-quality specification and checks on hand-built tables: identity, validation, and every check
against values computed by hand. No workspace. Adversarial inputs are included on purpose: reordered
columns and samples, ragged rows, mismatched lengths, empty and single-row tables, all-missing and
constant features, non-finite values, unexpected categories."""

import json
import math
from typing import Any

import pytest

from experionyx.data_quality import checks as ck
from experionyx.data_quality.data import Table, cell_key, classify
from experionyx.data_quality.results import Status, worst
from experionyx.data_quality.spec import CheckSpec, QualityConfig, QualitySpec
from experionyx.errors import ValidationError
from experionyx.interactions.taxonomy import Pairing
from experionyx.stats import core as st

DS = "dst_" + "a" * 32
NAN, INF = float("nan"), float("inf")


def tbl(columns: list[str], rows: list[list[Any]], target: list[Any] | None = None, *, ids: list[int] | None = None, split: str | None = "s", **kw: Any) -> Table:  # fmt: skip
    ids = ids if ids is not None else list(range(len(rows)))
    return Table(split, tuple(ids), tuple(columns), tuple(tuple(r) for r in rows), None if target is None else tuple(target), kw.get("ragged", ()), kw.get("duplicate_ids", ()), kw.get("declared_n", len(rows)), kw.get("target_len", None if target is None else len(target)))  # fmt: skip


def run(kind: str, config: dict[str, Any] | None, t: Table, *, features: list[tuple[str, str]] | None = None, target: str | None = None, **spec_kw: Any) -> Any:  # fmt: skip
    feats = features if features is not None else [(c, "NUMERIC") for c in t.columns]
    body: dict[str, Any] = {"dataset_id": DS, "checks": [{"type": kind, "config": config or {}}], "features": [{"name": n, "type": k} for n, k in feats], "config": {"min_members": 5, "resamples": 50, "permutations": 100}}  # fmt: skip
    if target:
        body["target"] = {"task": target}
    body.update(spec_kw)
    spec = QualitySpec.from_dict(body)
    ctx = ck.Ctx(spec, {t.split: t}, frozenset(spec.missing_values))
    return ck.PER_TABLE_FUNCS[kind](spec.checks[0], ctx, t, {"split": t.split})


def rule_names(r: Any) -> set[str]:
    return {v["rule"] for v in r.violations}


# -- identity and validation ------------------------------------------------------------------------------------


def spec_of(checks: list[dict[str, Any]], **over: Any) -> QualitySpec:
    return QualitySpec.from_dict({"dataset_id": DS, "checks": checks, **over})


def test_check_identity_is_deterministic_and_normalized() -> None:
    a = CheckSpec("missingness", {"features": ["b", "a"], "max_feature_rate": 0.5})
    b = CheckSpec(
        "missingness",
        {"max_feature_rate": 0.5, "features": ["a", "b"], "patterns": 5, "examples": 20},
    )  # defaults spelled out
    assert a.check_id == b.check_id and a.check_id.startswith("qcs_") and len(a.check_id) == 36
    assert (
        CheckSpec("numeric", {"bounds": {"x": [0.0, 10.0]}}).check_id
        == CheckSpec("numeric", {"bounds": {"x": [0, 10]}}).check_id
    )  # 10.0 == 10
    for other in (
        CheckSpec("missingness", {"features": ["a", "b"], "max_feature_rate": 0.6}),
        CheckSpec("missingness", {"features": ["a"], "max_feature_rate": 0.5}),
        CheckSpec("duplicates", {}),
    ):
        assert other.check_id != a.check_id
    assert CheckSpec("duplicates", {}).to_dict() == json.loads(
        json.dumps(CheckSpec("duplicates", {}).to_dict())
    )
    assert CheckSpec.from_dict(a.to_dict()) == a


def test_spec_identity_ignores_order_and_labels_but_not_meaning() -> None:
    c1, c2 = {"type": "duplicates"}, {"type": "missingness", "config": {"max_feature_rate": 0.1}}
    a = spec_of([c1, c2], splits=["train", "test"], missing_values=["", "NA"])
    b = spec_of([c2, c1], splits=["test", "train"], missing_values=["NA", ""])
    assert a.spec_id == b.spec_id and a.spec_id.startswith("dqs_")
    for changed in (
        spec_of([c1, c2], splits=["train"], missing_values=["", "NA"]),
        spec_of([c1, c2], splits=["train", "test"], missing_values=["", "NA", "?"]),
        spec_of(
            [c1, {"type": "missingness", "config": {"max_feature_rate": 0.2}}],
            splits=["train", "test"],
            missing_values=["", "NA"],
        ),
        spec_of([c1, c2], splits=["train", "test"], missing_values=["", "NA"], config={"seed": 1}),
        spec_of([c1, c2], splits=["train", "test"], missing_values=["", "NA"], id_column="id"),
        QualitySpec.from_dict({**a.to_dict(), "dataset_id": "dst_" + "b" * 32}),
    ):
        assert changed.spec_id != a.spec_id
    assert QualitySpec.from_dict(json.loads(json.dumps(a.to_dict()))) == a


@pytest.mark.parametrize(
    ("body", "match"),
    [
        ({"type": "nope"}, "unknown check type"),
        ({"type": "missingness", "config": {"max_feature_rate": 1.5}}, "must be in"),
        ({"type": "missingness", "config": {"max_feature_rate": -0.1}}, "must be in"),
        ({"type": "missingness", "config": {"surprise": 1}}, "unexpected"),
        ({"type": "sample_count", "config": {"min": 10, "max": 5}}, "min <= max"),
        ({"type": "sample_count", "config": {"min": -1}}, "integer >= 0"),
        ({"type": "sample_count", "config": {"min": 1.5}}, "integer"),
        ({"type": "numeric", "config": {"bounds": {"x": [5, 1]}}}, "low <= high"),
        ({"type": "numeric", "config": {"bounds": {"x": [None, None]}}}, "low and/or a high"),
        ({"type": "numeric", "config": {"bounds": {"x": [float("nan"), 1]}}}, "finite"),
        ({"type": "numeric", "config": {"outlier_method": "zscore"}}, "outlier_method"),
        ({"type": "numeric", "config": {"outlier_k": 0}}, "> 0"),
        ({"type": "numeric", "config": {"near_constant_ratio": 0}}, "0, 1"),
        ({"type": "identifiers", "config": {"unique_ratio": 0}}, "0, 1"),
        ({"type": "duplicates", "config": {"include_target": "yes"}}, "boolean"),
        ({"type": "categorical", "config": {"allowed": {"c": []}}}, "non-empty"),
        ({"type": "categorical", "config": {"allowed": {"c": [[1]]}}}, "bool, number or string"),
        ({"type": "target", "config": {"min_class_proportion": 2}}, "must be in"),
        ({"type": "target_leakage", "config": {}}, "candidates"),
        ({"type": "target_leakage", "config": {"candidates": []}}, "explicitly named"),
        ({"type": "split_overlap", "config": {"reference": "a", "comparison": "a"}}, "two different"),
        ({"type": "group_comparison", "config": {"reference": {"split": "a"}, "comparison": {"split": "a"}}}, "two different groups"),
        ({"type": "group_comparison", "config": {"reference": {"window": {"start": 0, "end": 1}}, "comparison": {"split": "b"}}}, "declared `split`"),
        ({"type": "group_comparison", "config": {"reference": {"split": "a"}, "comparison": {"split": "b"}, "aspects": ["mood"]}}, "aspects"),
        ({"type": "schema", "config": {"required": ["a", "a"]}}, "twice"),
        ({"type": "schema", "config": {"required": [" a"]}}, "whitespace"),
    ],
)  # fmt: skip
def test_invalid_rules_are_refused_before_execution(body: dict[str, Any], match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        spec_of(
            [body],
            splits=["a", "b"],
            features=[{"name": "x", "type": "NUMERIC"}, {"name": "c", "type": "CATEGORICAL"}],
        )


def test_cross_references_are_validated_at_spec_time() -> None:
    feats = [{"name": "x", "type": "NUMERIC"}, {"name": "c", "type": "CATEGORICAL"}]
    bad: tuple[tuple[list[dict[str, Any]], dict[str, Any], str], ...] = (
        ([{"type": "numeric"}], {}, "needs declared `features`"),
        (
            [{"type": "numeric", "config": {"features": ["zz"]}}],
            {"features": feats},
            "not declared",
        ),
        ([{"type": "numeric", "config": {"features": ["c"]}}], {"features": feats}, "NUMERIC"),
        (
            [{"type": "categorical", "config": {"features": ["x"]}}],
            {"features": feats},
            "CATEGORICAL",
        ),
        ([{"type": "target"}], {}, "declared `target`"),
        (
            [{"type": "target_leakage", "config": {"candidates": ["x"]}}],
            {"features": feats},
            "declared `target`",
        ),
        (
            [{"type": "target_leakage", "config": {"candidates": ["nope"]}}],
            {"features": feats, "target": {"task": "CLASSIFICATION"}},
            "not a declared feature",
        ),
        (
            [{"type": "split_overlap", "config": {"reference": "a", "comparison": "b"}}],
            {"splits": ["a"]},
            "not listed in `splits`",
        ),
        (
            [{"type": "temporal_order", "config": {"reference": "a", "comparison": "b"}}],
            {"splits": ["a", "b"]},
            "declared `ordering`",
        ),
        (
            [
                {
                    "type": "group_comparison",
                    "config": {
                        "reference": {"split": "a", "window": {"start": 0, "end": 5}},
                        "comparison": {"split": "b"},
                    },
                }
            ],
            {"splits": ["a", "b"]},
            "window group needs",
        ),
        (
            [
                {
                    "type": "group_comparison",
                    "config": {"reference": {"split": "a"}, "comparison": {"split": "zz"}},
                }
            ],
            {"splits": ["a"]},
            "not listed",
        ),
        (
            [{"type": "target", "config": {"allowed": [1]}}],
            {"target": {"task": "REGRESSION"}},
            "classification only",
        ),
        ([{"type": "duplicates"}, {"type": "duplicates"}], {}, "twice"),
        ([], {}, "between 1 and"),
    )
    for checks, over, match in bad:
        with pytest.raises(ValidationError, match=match):
            spec_of(checks, **over)
    with pytest.raises(ValidationError, match="target task"):
        spec_of([{"type": "duplicates"}], target={"task": "CLUSTERING"})
    with pytest.raises(ValidationError, match="unsupported feature type"):
        spec_of([{"type": "duplicates"}], features=[{"name": "t", "type": "TEXT"}])
    with pytest.raises(ValidationError, match="declared twice"):
        spec_of(
            [{"type": "duplicates"}],
            features=[{"name": "t", "type": "NUMERIC"}, {"name": "t", "type": "BOOLEAN"}],
        )
    with pytest.raises(ValidationError, match="unexpected"):
        QualitySpec.from_dict({"dataset_id": DS, "checks": [{"type": "duplicates"}], "surprise": 1})
    with pytest.raises(ValidationError, match="dst"):
        QualitySpec.from_dict({"dataset_id": "nope", "checks": [{"type": "duplicates"}]})
    with pytest.raises(ValidationError, match="quality_schema"):
        QualitySpec.from_dict(
            {"dataset_id": DS, "checks": [{"type": "duplicates"}], "quality_schema": 9}
        )
    with pytest.raises(ValidationError, match="slice-capable"):
        spec_of(
            [{"type": "schema"}],
            slices=[{"name": "s", "condition": {"op": "eq", "field": "target", "values": [1]}}],
        )
    for kw in ({"min_members": 1}, {"confidence": 1.0}, {"alpha": 0}, {"correction": "HOLM"}, {"seed": -1}, {"slice_checks": ["schema"]}, {"slice_checks": ["target", "missingness"]}):  # fmt: skip
        with pytest.raises(ValidationError):
            QualityConfig(**kw)
    with pytest.raises(ValidationError, match="unknown quality setting"):
        QualityConfig.from_dict({"bogus": 1})


# -- statuses ------------------------------------------------------------------------------------------------------


def test_status_aggregation_and_the_absence_of_a_score() -> None:
    assert worst([Status.PASS, Status.WARNING, Status.FAIL]) is Status.FAIL
    assert worst([Status.PASS, Status.INCONCLUSIVE]) is Status.INCONCLUSIVE
    assert worst([Status.PASS, Status.PASS]) is Status.PASS and worst([]) is Status.NOT_APPLICABLE
    assert {s.value for s in Status} == {
        "PASS",
        "FAIL",
        "WARNING",
        "INCONCLUSIVE",
        "UNAVAILABLE",
        "NOT_APPLICABLE",
    }


def test_cell_classification_and_canonical_keys() -> None:
    toks = frozenset({"NA", ""})
    got = classify([1, 2.5, None, NAN, INF, "NA", "", "x", True], "NUMERIC", toks)
    assert [s for s, _ in got] == [
        "valid",
        "valid",
        "missing",
        "missing",
        "nonfinite",
        "missing",
        "missing",
        "invalid",
        "invalid",
    ]
    assert [s for s, _ in classify(["a", 1, 1.0, None, [1]], "CATEGORICAL", frozenset())] == [
        "valid",
        "valid",
        "valid",
        "missing",
        "invalid",
    ]
    assert (
        cell_key(1) == cell_key(1.0)
        and cell_key(NAN) == cell_key(float("nan"))
        and cell_key(None) != cell_key("None")
    )


# -- schema and sample count ---------------------------------------------------------------------------------------


def test_schema_reports_required_extra_ragged_and_type_violations() -> None:
    t = tbl(["a", "b", "extra"], [[1, "x", 0], [2, 3.5, 0], [3, "z"]], ragged=(2,))
    r = run(
        "schema",
        {"required": ["a", "id"], "allow_extra": False},
        t,
        features=[("a", "NUMERIC"), ("b", "NUMERIC")],
    )
    assert r.status is Status.FAIL and rule_names(r) == {
        "required_column_missing",
        "undeclared_column",
        "ragged_rows",
        "invalid_type",
    }
    bad = next(v for v in r.violations if v["rule"] == "invalid_type")
    assert bad["subject"] == "b" and bad["affected_rows"] == [0, 2] and bad["count"] == 2
    ok = run("schema", {"required": ["a"]}, tbl(["a", "b"], [[1, 2], [3, 4]]))
    assert ok.status is Status.PASS and ok.violations == ()


def test_column_and_row_order_do_not_change_results() -> None:
    rows = [[1.0, "x", None], [2.0, "y", 3.0], [2.0, "y", 3.0], [4.0, "x", 5.0]]
    feats = [("a", "NUMERIC"), ("b", "CATEGORICAL"), ("c", "NUMERIC")]
    base = tbl(["a", "b", "c"], rows)
    shuffled_cols = tbl(["c", "a", "b"], [[r[2], r[0], r[1]] for r in rows])
    shuffled_rows = tbl(["a", "b", "c"], rows[::-1], ids=list(range(3, -1, -1)))
    for kind, cfg in (
        ("missingness", {}),
        ("duplicates", {}),
        ("numeric", {"features": ["a", "c"]}),
        ("categorical", {"features": ["b"]}),
    ):
        want = run(kind, cfg, base, features=feats).to_dict()
        for other in (shuffled_cols, shuffled_rows):
            got = run(kind, cfg, other, features=feats).to_dict()
            for k in ("status", "thresholds", "evidence"):
                assert got[k] == want[k], (kind, k)
            if kind != "duplicates":
                assert got["observations"] == want["observations"], kind
    dup = run("duplicates", {}, shuffled_rows, features=feats)
    assert dup.observations["n_duplicate_rows"] == 1 and dup.observations["n_duplicate_groups"] == 1


def test_sample_count_rules_and_consistency() -> None:
    t = tbl(["a"], [[1], [2], [3]])
    assert run("sample_count", {"min": 3, "max": 3, "expected": 3}, t).status is Status.PASS
    r = run("sample_count", {"min": 5}, t)
    assert (
        r.status is Status.FAIL
        and r.violations[0]["observed"] == 3
        and r.violations[0]["configured"] == 5
    )
    assert rule_names(run("sample_count", {"max": 2}, t)) == {"max_violated"}
    assert rule_names(run("sample_count", {"expected": 4}, t)) == {"expected_violated"}
    mism = run("sample_count", {}, tbl(["a"], [[1], [2]], declared_n=5))
    assert mism.status is Status.FAIL and rule_names(mism) == {"metadata_count_mismatch"}
    lens = run(
        "sample_count",
        {},
        tbl(["a"], [[1], [2], [3]], [0, 1], target_len=2),
        target="CLASSIFICATION",
    )
    assert "feature_target_length_mismatch" in rule_names(lens)  # target shorter than the rows
    assert "target_absent" in rule_names(
        run("sample_count", {}, tbl(["a"], [[1]]), target="CLASSIFICATION")
    )
    empty = run("sample_count", {}, tbl(["a"], []))
    assert empty.status is Status.WARNING and rule_names(empty) == {"empty_table"}
    one = run("sample_count", {}, tbl(["a"], [[1]]))
    assert one.status is Status.PASS and one.observations["n_rows"] == 1
    assert (
        run("sample_count", {}, tbl(["a"], [[1], [2]], duplicate_ids=(1,))).status is Status.WARNING
    )


# -- missingness -------------------------------------------------------------------------------------------------------


def test_missingness_counts_rates_patterns_and_representations() -> None:
    rows: list[list[Any]] = [[None, 1.0, "NA"], [NAN, None, 2.0], [3.0, 4.0, 5.0], [None, 2.0, ""]]
    t = tbl(["a", "b", "c"], rows)
    r = run(
        "missingness",
        {},
        t,
        features=[("a", "NUMERIC"), ("b", "NUMERIC"), ("c", "NUMERIC")],
        missing_values=["", "NA"],
    )
    f = r.observations["features"]
    assert (f["a"]["n_missing"], f["b"]["n_missing"], f["c"]["n_missing"]) == (
        3,
        1,
        2,
    )  # None, NaN and the declared tokens
    assert f["a"]["rate"] == 0.75 and f["a"]["all_missing"] is False
    w = st.proportion_interval(3, 4, 0.95)
    assert (f["a"]["interval"]["lower"], f["a"]["interval"]["upper"]) == (w.lower, w.upper)
    assert r.observations["rows_with_any_missing"] == 3
    assert r.observations["missing_per_row_histogram"] == {0: 1, 2: 3}
    pats = {tuple(p["features"]): p["rows"] for p in r.observations["patterns"]}
    assert pats == {("a", "c"): 2, ("a", "b"): 1}
    assert (
        r.status is Status.PASS and "informative" in r.observations["note"]
    )  # measured, not judged
    no_tokens = run("missingness", {}, t, features=[("c", "NUMERIC")])
    assert (
        no_tokens.observations["features"]["c"]["n_missing"] == 0
    )  # a string is not missing unless declared so


def test_missingness_thresholds_fail_and_all_missing_is_reported() -> None:
    t = tbl(["a", "b"], [[None, 1], [None, 2], [None, None], [None, 4]])
    r = run("missingness", {"max_feature_rate": 0.5, "max_row_rate": 0.5}, t)
    assert r.status is Status.FAIL and rule_names(r) == {
        "feature_missing_rate_exceeded",
        "row_missing_rate_exceeded",
    }
    assert r.observations["features"]["a"]["all_missing"] is True
    rows = next(v for v in r.violations if v["rule"] == "row_missing_rate_exceeded")
    assert rows["affected_rows"] == [2] and rows["count"] == 1
    assert (
        run("missingness", {"max_feature_rate": 1.0}, t).status is Status.PASS
    )  # the rate is not above the maximum
    assert run("missingness", {}, tbl(["a"], [])).status is Status.NOT_APPLICABLE


# -- duplicates and identity ------------------------------------------------------------------------------------------------


def test_duplicate_rows_are_reported_not_judged_unless_a_threshold_is_configured() -> None:
    t = tbl(["a", "b"], [[1, "x"], [1, "x"], [2, "y"], [1, "x"], [3, "z"]], [0, 0, 1, 1, 0])
    r = run(
        "duplicates",
        {},
        t,
        features=[("a", "NUMERIC"), ("b", "CATEGORICAL")],
        target="CLASSIFICATION",
    )
    o = r.observations
    assert (o["n_rows"], o["n_unique_rows"], o["n_duplicate_rows"], o["n_duplicate_groups"]) == (
        5,
        3,
        2,
        1,
    )
    assert (
        o["duplicate_rate"] == 0.4 and r.status is Status.WARNING and not r.fails
        if hasattr(r, "fails")
        else True
    )
    assert {v["severity"] for v in r.violations} == {
        "WARNING"
    }  # never FAIL without a configured threshold
    assert (
        o["conflicting_groups"] == 1
        and o["conflicting_rows"] == 3
        and o["conflicts_available"] is True
    )  # rows 0,1,3 share features; targets 0,0,1
    assert "identical_features_different_target" in rule_names(r)
    strict = run("duplicates", {"max_duplicate_rate": 0.3, "max_conflicting_rate": 0.9}, t, features=[("a", "NUMERIC"), ("b", "CATEGORICAL")], target="CLASSIFICATION")  # fmt: skip
    assert strict.status is Status.FAIL and "duplicate_rate_exceeded" in rule_names(strict)
    lax = run("duplicates", {"max_duplicate_rate": 0.5, "max_conflicting_rate": 0.9}, t, features=[("a", "NUMERIC"), ("b", "CATEGORICAL")], target="CLASSIFICATION")  # fmt: skip
    assert (
        lax.status is Status.PASS or lax.status is Status.WARNING
    )  # under both thresholds the phenomenon is still observed
    with_target = run("duplicates", {"include_target": True}, t, features=[("a", "NUMERIC"), ("b", "CATEGORICAL")], target="CLASSIFICATION")  # fmt: skip
    assert (
        with_target.observations["n_duplicate_rows"] == 1
    )  # (1,'x',0) twice: the third copy has another target
    none = run("duplicates", {}, tbl(["a"], [[1], [2], [3]]))
    assert none.status is Status.PASS and none.observations["conflicts_available"] is False


def test_duplicate_ids_and_conflicting_rows_sharing_an_identifier() -> None:
    t = tbl(
        ["id", "v"],
        [[7, 1.0], [8, 2.0], [7, 1.0], [9, 3.0], [8, 5.0], [None, 1.0]],
        [0, 1, 0, 1, 1, 0],
    )
    r = run(
        "identifiers", {}, t, features=[("v", "NUMERIC")], target="CLASSIFICATION", id_column="id"
    )
    d = r.observations["id_column"]
    assert (
        d["n_distinct"],
        d["n_duplicated_values"],
        d["n_rows_with_duplicated_id"],
        d["n_conflicting_groups"],
        d["n_missing"],
    ) == (3, 2, 4, 1, 1)
    assert (
        rule_names(r) >= {"duplicate_ids", "conflicting_rows_share_id", "missing_ids"}
        and r.status is Status.WARNING
    )
    conf = next(v for v in r.violations if v["rule"] == "conflicting_rows_share_id")
    assert conf["affected_rows"] == [1, 4] and conf["n_groups"] == 1  # id 8: v differs
    strict = run(
        "identifiers",
        {"require_unique": True},
        t,
        features=[("v", "NUMERIC")],
        target="CLASSIFICATION",
        id_column="id",
    )
    assert strict.status is Status.FAIL
    clean = run(
        "identifiers",
        {},
        tbl(["id", "v"], [[1, 1.0], [2, 2.0]]),
        features=[("v", "NUMERIC")],
        id_column="id",
    )
    assert (
        clean.status is Status.PASS and clean.observations["id_column"]["n_duplicated_values"] == 0
    )
    assert "adapter_served_ids_twice" in rule_names(
        run("identifiers", {}, tbl(["v"], [[1.0], [2.0]], duplicate_ids=(1,)))
    )


def test_identifier_like_columns_are_flagged_only_when_they_could_identify_rows() -> None:
    n = 40
    t = tbl(["uid", "cont", "cat", "const"], [[i, i * 0.37, f"c{i}", 1] for i in range(n)])
    feats = [("uid", "NUMERIC"), ("cont", "NUMERIC"), ("cat", "CATEGORICAL"), ("const", "NUMERIC")]
    r = run("identifiers", {"min_samples": 20}, t, features=feats)
    flagged = {v["subject"] for v in r.violations if v["rule"] == "identifier_like"}
    assert flagged == {
        "uid",
        "cat",
    }  # integer-valued and categorical, all distinct; a continuous column is not a candidate
    assert (
        r.observations["candidates"]["cont"]["eligible"] is False
        and r.observations["candidates"]["const"]["unique_ratio"] == 1 / n
    )
    thin = run("identifiers", {"min_samples": 100}, t, features=feats)
    assert (
        thin.status is Status.PASS
        and thin.observations["candidates"]["uid"]["status"] == "INCONCLUSIVE"
    )


# -- numeric -------------------------------------------------------------------------------------------------------------------


def test_numeric_counts_nonfinite_constant_and_bounds() -> None:
    vals = [1.0, 2.0, 3.0, 4.0, NAN, INF, -INF, None, "oops", 5.0]
    t = tbl(["x", "k", "allnan"], [[v, 9.0, None] for v in vals])
    r = run(
        "numeric",
        {"bounds": {"x": [2, 4]}},
        t,
        features=[("x", "NUMERIC"), ("k", "NUMERIC"), ("allnan", "NUMERIC")],
    )
    x = r.observations["features"]["x"]
    assert (x["n_total"], x["n_valid"], x["n_missing"], x["n_nonfinite"], x["n_invalid"]) == (
        10,
        5,
        2,
        2,
        1,
    )
    assert (
        x["bounds"]["n_outside"] == 2
        and x["bounds"]["observed_min"] == 1.0
        and x["bounds"]["observed_max"] == 5.0
    )
    out = next(v for v in r.violations if v["rule"] == "outside_bounds")
    assert out["affected_rows"] == [0, 9] and out["low"] == 2 and out["high"] == 4
    assert {"nonfinite_values", "invalid_type", "outside_bounds"} <= rule_names(
        r
    ) and r.status is Status.FAIL
    k = r.observations["features"]["k"]
    assert k["constant"] is True and k["n_unique"] == 1 and k["status"] == "WARNING"
    assert (
        r.observations["features"]["allnan"]["status"] == "INCONCLUSIVE"
    )  # nothing valid: no statistic is invented
    assert x["summary"]["mean"]["value"] == pytest.approx(3.0)


def test_near_zero_variance_is_configured_not_assumed() -> None:
    t = tbl(["x"], [[1.0]] * 99 + [[2.0]])
    r = run(
        "numeric",
        {"near_constant_ratio": 0.99, "min_variance": 0.1},
        t,
        features=[("x", "NUMERIC")],
    )
    o = r.observations["features"]["x"]
    assert (
        o["modal_fraction"] == 0.99
        and o["constant"] is False
        and o["variance"] == pytest.approx(0.01 * 0.99 * 100 / 99)
    )
    assert (
        rule_names(r) == {"near_constant_feature", "variance_below_minimum"}
        and r.status is Status.FAIL
    )
    loose = run("numeric", {"near_constant_ratio": 1.0}, t, features=[("x", "NUMERIC")])
    assert loose.status is Status.PASS  # 0.99 < 1.0 and no variance floor configured


def test_outliers_use_documented_methods_and_never_change_status_alone() -> None:
    xs = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 100.0]
    t = tbl(["x"], [[v] for v in xs])
    r = run("numeric", {}, t, features=[("x", "NUMERIC")])
    o = r.observations["features"]["x"]["outliers"]
    q1, q3 = 12.25, 16.75  # type-7 quartiles of the sorted values
    assert (o["q1"], o["q3"]) == (q1, q3) and o["upper_fence"] == pytest.approx(
        q3 + 1.5 * (q3 - q1)
    )
    assert (
        o["n_outliers"] == 1 and o["affected_rows"] == [9] and "not evidence of error" in o["note"]
    )
    assert r.status is Status.PASS  # an outlier is an observation
    capped = run("numeric", {"max_outlier_rate": 0.05}, t, features=[("x", "NUMERIC")])
    assert capped.status is Status.FAIL and rule_names(capped) == {"outlier_rate_exceeded"}
    mad = run("numeric", {"outlier_method": "mad"}, t, features=[("x", "NUMERIC")]).observations[
        "features"
    ]["x"]["outliers"]
    assert mad["method"] == "mad" and mad["median"] == 14.5 and mad["n_outliers"] == 1
    flat = run("numeric", {"outlier_method": "mad"}, tbl(["x"], [[1.0]] * 5), features=[("x", "NUMERIC")]).observations["features"]["x"]["outliers"]  # fmt: skip
    assert flat["n_outliers"] == 0 and "MAD is zero" in flat["scoring_note"]
    assert (
        run("numeric", {}, tbl(["x"], [[5.0]]), features=[("x", "NUMERIC")]).status
        is Status.WARNING
    )  # a single row is constant


def test_numeric_check_without_numeric_features_is_not_applicable() -> None:
    r = run("numeric", {}, tbl(["c"], [["a"], ["b"]]), features=[("c", "CATEGORICAL")])
    assert r.status is Status.NOT_APPLICABLE and "no NUMERIC feature" in r.reason


# -- categorical --------------------------------------------------------------------------------------------------------------


def test_invalid_rare_and_new_categories_are_three_different_things() -> None:
    vals = ["a"] * 60 + ["b"] * 38 + ["rare"] * 1 + ["bad"] * 1
    t = tbl(["c"], [[v] for v in vals])
    r = run(
        "categorical",
        {"allowed": {"c": ["a", "b", "rare"]}, "rare_fraction": 0.02},
        t,
        features=[("c", "CATEGORICAL")],
    )
    o = r.observations["features"]["c"]
    assert o["cardinality"] == 4 and o["counts"]['"a"'] == 60 and o["proportions"]['"b"'] == 0.38
    assert o["rare_categories"] == ['"bad"', '"rare"'] and o["invalid_categories"] == [
        '"bad"'
    ]  # rare is broader than invalid
    assert rule_names(r) == {"invalid_category"} and r.violations[0]["affected_rows"] == [99]
    assert r.status is Status.FAIL and "not invalid" in o["rare_note"]
    only_rare = run("categorical", {"rare_fraction": 0.02}, t, features=[("c", "CATEGORICAL")])
    assert (
        only_rare.status is Status.PASS
        and len(only_rare.observations["features"]["c"]["rare_categories"]) == 2
    )
    card = run("categorical", {"max_cardinality": 3}, t, features=[("c", "CATEGORICAL")])
    assert card.status is Status.FAIL and rule_names(card) == {"cardinality_exceeded"}


def test_boolean_and_invalid_typed_categorical_cells() -> None:
    t = tbl(["f", "c"], [[True, "a"], [False, "a"], [1, "b"], [2, "b"], [None, [1]], ["yes", "a"]])
    r = run(
        "categorical",
        {"allowed": {"f": [True]}},
        t,
        features=[("f", "BOOLEAN"), ("c", "CATEGORICAL")],
    )
    f = r.observations["features"]["f"]
    assert f["counts"] == {"false": 1, "true": 2} and f["n_invalid"] == 2 and f["n_missing"] == 1
    assert {"invalid_type", "invalid_category"} <= rule_names(
        r
    )  # 'false' is valid for the type but not in the allowed set
    all_missing = run(
        "categorical", {}, tbl(["c"], [[None], [None]]), features=[("c", "CATEGORICAL")]
    )
    assert all_missing.status is Status.INCONCLUSIVE


def test_distribution_moments_match_hand_computation() -> None:
    xs = [1.0, 2.0, 3.0, 4.0, 10.0]
    m = sum(xs) / 5
    m2 = sum((x - m) ** 2 for x in xs) / 5
    g1 = sum((x - m) ** 3 for x in xs) / 5 / m2**1.5
    skew, kurt = ck.moments(xs)
    assert skew == pytest.approx(g1, abs=1e-12) and kurt == pytest.approx(
        sum((x - m) ** 4 for x in xs) / 5 / m2**2 - 3, abs=1e-12
    )
    assert ck.moments([2.0, 2.0]) == (None, None)
    t = tbl(["x", "c"], [[x, "a" if i < 4 else "b"] for i, x in enumerate(xs)])
    feats = [("x", "NUMERIC"), ("c", "CATEGORICAL")]
    plain = run("distribution", {}, t, features=feats)
    assert plain.status is Status.PASS and plain.observations["features"]["x"][
        "skewness"
    ] == pytest.approx(g1)
    flagged = run(
        "distribution",
        {"max_abs_skew": 0.5, "max_modal_fraction": 0.7, "min_unique": 6},
        t,
        features=feats,
    )
    assert flagged.status is Status.WARNING and rule_names(flagged) == {
        "skew_exceeds_configured",
        "dominant_value",
        "few_distinct_values",
    }


# -- target ----------------------------------------------------------------------------------------------------------------------


def test_class_counts_proportions_singletons_and_missing_targets() -> None:
    tg = [0] * 12 + [1] * 6 + [2] + [None] + [0]
    t = tbl(["x"], [[i] for i in range(len(tg))], tg)
    r = run("target", {"min_class_count": 2}, t, target="CLASSIFICATION")
    o = r.observations
    assert (o["n_classes"], o["n_missing"], o["singleton_classes"]) == (3, 1, ["2"])
    assert o["classes"]["0"]["count"] == 13 and o["classes"]["0"]["proportion"] == 13 / 20
    w = st.proportion_interval(13, 20, 0.95)
    assert (o["classes"]["0"]["interval"]["lower"], o["classes"]["0"]["interval"]["upper"]) == (
        w.lower,
        w.upper,
    )
    assert o["imbalance_ratio"] == 13 and "not a defect" in o["imbalance_note"]
    assert (
        rule_names(r) == {"missing_targets", "singleton_class", "class_below_min_count"}
        and r.status is Status.FAIL
    )
    imbalanced = run(
        "target",
        {},
        tbl(["x"], [[i] for i in range(20)], [0] * 18 + [1] * 2),
        target="CLASSIFICATION",
    )
    # class imbalance alone is not a failure
    assert imbalanced.status is Status.PASS
    assert imbalanced.observations["imbalance_ratio"] == 9
    prop = run(
        "target",
        {"min_class_proportion": 0.1},
        tbl(["x"], [[i] for i in range(20)], [0] * 19 + [1]),
        target="CLASSIFICATION",
    )
    assert prop.status is Status.FAIL and rule_names(prop) == {
        "class_below_min_proportion",
        "singleton_class",
    }
    miss = run("target", {"max_missing_rate": 0.01}, t, target="CLASSIFICATION")
    assert "missing_target_rate_exceeded" in rule_names(miss)


def test_target_validity_types_and_availability() -> None:
    mixed = run(
        "target", {}, tbl(["x"], [[1], [2], [3], [4]], [0, 1, "one", 1]), target="CLASSIFICATION"
    )
    assert (
        mixed.status is Status.FAIL
        and "inconsistent_target_types" in rule_names(mixed)
        and mixed.observations["value_types"] == ["number", "str"]
    )
    allowed = run(
        "target",
        {"allowed": [0, 1]},
        tbl(["x"], [[1], [2], [3]], [0, 1, 7]),
        target="CLASSIFICATION",
    )
    bad = next(v for v in allowed.violations if v["rule"] == "invalid_target_value")
    assert bad["values"] == ["7"] and bad["affected_rows"] == [2]
    assert (
        run("target", {}, tbl(["x"], [[1]]), target="CLASSIFICATION").status is Status.UNAVAILABLE
    )  # a declared target that is not served
    lens = tbl(["x"], [[1], [2], [3]], None, target_len=2)
    assert (
        run("target", {}, lens, target="CLASSIFICATION").status is Status.UNAVAILABLE
    )  # misaligned targets are never used
    assert (
        run("target", {}, tbl(["x"], [], []), target="CLASSIFICATION").status
        is Status.NOT_APPLICABLE
    )
    none = run("target", {}, tbl(["x"], [[1], [2]], [None, None]), target="CLASSIFICATION")
    assert none.status is Status.WARNING or none.status is Status.INCONCLUSIVE
    single = run("target", {}, tbl(["x"], [[1]], [0]), target="CLASSIFICATION")
    assert single.observations["singleton_classes"] == ["0"]  # a single row is a singleton class
    inf_t = run("target", {}, tbl(["x"], [[1], [2], [3]], [0, INF, 1]), target="CLASSIFICATION")
    assert "nonfinite_target" in rule_names(inf_t)


def test_regression_targets() -> None:
    ys = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 200.0]
    t = tbl(["x"], [[i] for i in range(len(ys))], ys)
    r = run("target", {}, t, target="REGRESSION")
    assert r.status is Status.PASS and r.observations["summary"]["mean"]["value"] == pytest.approx(
        24.5
    )
    assert r.observations["outliers"]["n_outliers"] == 1 and r.observations["outliers"][
        "affected_rows"
    ] == [9]
    bad = run(
        "target", {}, tbl(["x"], [[1], [2], [3], [4]], [1.0, "a", INF, NAN]), target="REGRESSION"
    )
    assert (
        rule_names(bad) == {"non_numeric_target", "nonfinite_target", "missing_targets"}
        and bad.status is Status.FAIL
    )
    assert run("target", {}, tbl(["x"], [[1]], [NAN]), target="REGRESSION").status in (
        Status.WARNING,
        Status.INCONCLUSIVE,
    )


# -- leakage indicators ---------------------------------------------------------------------------------------------------------------


def test_leakage_indicators_are_conservative_and_named_as_indicators() -> None:
    n = 60
    tg = [i % 2 for i in range(n)]
    rows = [
        [float(tg[i]), tg[i] * 10 + (i % 3) * 0.0, float(i), i % 7, (i * 13) % 2] for i in range(n)
    ]  # copy, exact function, unrelated, weak, coincidence
    t = tbl(["copy", "func", "noise", "weak", "coin"], rows, tg)
    feats = [
        ("copy", "NUMERIC"),
        ("func", "NUMERIC"),
        ("noise", "NUMERIC"),
        ("weak", "NUMERIC"),
        ("coin", "NUMERIC"),
    ]
    r = run(
        "target_leakage",
        {"candidates": ["copy", "func", "noise", "weak"]},
        t,
        features=feats,
        target="CLASSIFICATION",
    )
    c = r.observations["candidates"]
    assert c["copy"]["equality_rate"] == 1.0 and set(c["copy"]["indicators"]) >= {
        "identical_to_target",
        "value_determines_target",
    }
    assert c["func"]["equality_rate"] < 1.0 and c["func"]["indicators"] == [
        "value_determines_target"
    ]
    assert (
        c["noise"]["indicators"] == []
        and c["weak"]["indicators"] == []
        and c["noise"]["status"] == "PASS"
    )
    assert r.status is Status.WARNING and {v["subject"] for v in r.violations} == {"copy", "func"}
    assert (
        "not proof" in r.violations[0]["note"]
        and "evidence, not conclusions" in r.observations["note"]
    )
    reg = tbl(
        ["a", "b"], [[float(i), 2.0 * i + 1] for i in range(40)], [2.0 * i + 1 for i in range(40)]
    )
    rr = run(
        "target_leakage",
        {"candidates": ["a", "b"]},
        reg,
        features=[("a", "NUMERIC"), ("b", "NUMERIC")],
        target="REGRESSION",
    )
    assert (
        rr.observations["candidates"]["a"]["pearson_r"] == pytest.approx(1.0)
        and "near_perfect_correlation" in rr.observations["candidates"]["a"]["indicators"]
    )
    assert "identical_to_target" in rr.observations["candidates"]["b"]["indicators"]
    thin = run(
        "target_leakage",
        {"candidates": ["copy"], "min_rows": 100},
        t,
        features=feats,
        target="CLASSIFICATION",
    )
    assert thin.status is Status.INCONCLUSIVE
    assert (
        run(
            "target_leakage",
            {"candidates": ["copy"]},
            tbl(["copy"], [[1.0]] * 30),
            features=[("copy", "NUMERIC")],
            target="CLASSIFICATION",
        ).status
        is Status.UNAVAILABLE
    )


def test_a_constant_feature_does_not_look_like_leakage_when_the_target_is_lopsided() -> None:
    tg = [0] * 58 + [1, 1]
    t = tbl(["k"], [[5.0]] * 60, tg)
    r = run(
        "target_leakage",
        {"candidates": ["k"]},
        t,
        features=[("k", "NUMERIC")],
        target="CLASSIFICATION",
    )
    assert r.status is Status.PASS and r.observations["candidates"]["k"]["n_distinct_values"] == 1


# -- cross-split ------------------------------------------------------------------------------------------------------------------------


def two(spec_body: dict[str, Any], tr: Table, te: Table, kind: str, cfg: dict[str, Any]) -> Any:
    body = {"dataset_id": DS, "splits": ["test", "train"], "checks": [{"type": kind, "config": cfg}], "config": {"min_members": 5, "resamples": 50, "permutations": 100}, **spec_body}  # fmt: skip
    spec = QualitySpec.from_dict(body)
    ctx = ck.Ctx(spec, {"train": tr, "test": te}, frozenset(spec.missing_values))
    return ck.CROSS_FUNCS[kind](spec.checks[0], ctx)


def test_split_overlap_detects_ids_vectors_and_label_conflicts() -> None:
    tr = tbl(
        ["a", "b"],
        [[1, "x"], [2, "y"], [3, "z"], [4, "w"]],
        [0, 1, 0, 1],
        ids=[0, 1, 2, 3],
        split="train",
    )
    te = tbl(
        ["a", "b"],
        [[3, "z"], [9, "q"], [1, "x"], [2, "y"]],
        [0, 0, 1, 1],
        ids=[2, 10, 11, 12],
        split="test",
    )
    r = two(
        {"target": {"task": "CLASSIFICATION"}},
        tr,
        te,
        "split_overlap",
        {"reference": "train", "comparison": "test"},
    )
    o = r.observations
    assert (
        o["sample_id_overlap"],
        o["comparison_rows_with_identical_vector_in_reference"],
        o["label_conflicts"],
        o["label_agreements"],
    ) == (1, 3, 1, 2)
    assert r.status is Status.WARNING and rule_names(r) == {
        "sample_id_overlap",
        "identical_feature_vector_across_splits",
        "identical_features_different_target_across_splits",
    }
    assert "indicator" in o["note"].lower() and "interpretation is required" in o["note"]
    conflict = next(v for v in r.violations if v["rule"].endswith("different_target_across_splits"))
    assert conflict["affected_rows"] == [11]  # (1,'x') has target 0 in train and 1 in test
    strict = two(
        {}, tr, te, "split_overlap", {"reference": "train", "comparison": "test", "max_overlap": 1}
    )
    assert strict.status is Status.FAIL
    clean = two(
        {},
        tr,
        tbl(["a", "b"], [[7, "n"]], ids=[50], split="test"),
        "split_overlap",
        {"reference": "train", "comparison": "test"},
    )
    assert clean.status is Status.PASS and clean.observations["sample_id_overlap"] == 0
    subset = two(
        {"features": [{"name": "a", "type": "NUMERIC"}]},
        tr,
        te,
        "split_overlap",
        {"reference": "train", "comparison": "test", "features": ["a"]},
    )
    assert subset.observations["columns_compared"] == ["a"]
    assert (
        two(
            {"features": [{"name": "a", "type": "NUMERIC"}]},
            tbl(["z"], [[1]], split="train"),
            te,
            "split_overlap",
            {"reference": "train", "comparison": "test", "features": ["a"]},
        ).status
        is Status.NOT_APPLICABLE
    )


def test_temporal_order_between_declared_splits() -> None:
    tr = tbl(["ts"], [[1.0], [2.0], [3.0], [None]], ids=[0, 1, 2, 3], split="train")
    ok = tbl(["ts"], [[4.0], [5.0]], ids=[10, 11], split="test")
    body = {"ordering": {"field": "feature:ts"}}
    r = two(body, tr, ok, "temporal_order", {"reference": "train", "comparison": "test"})
    assert r.status is Status.WARNING and rule_names(r) == {
        "unusable_ordering_key"
    }  # only the missing key; the order itself holds
    assert r.observations["reference_range"] == [1.0, 3.0] and r.observations[
        "comparison_range"
    ] == [4.0, 5.0]
    early = tbl(["ts"], [[2.5], [6.0]], ids=[10, 11], split="test")
    bad = two(body, tr, early, "temporal_order", {"reference": "train", "comparison": "test"})
    assert {
        "comparison_rows_not_after_reference",
        "reference_rows_not_before_comparison",
    } <= rule_names(bad)
    assert next(v for v in bad.violations if v["rule"] == "comparison_rows_not_after_reference")[
        "affected_rows"
    ] == [10]
    tie = tbl(["ts"], [[3.0], [4.0]], ids=[10, 11], split="test")
    assert "comparison_rows_not_after_reference" not in rule_names(
        two(body, tr, tie, "temporal_order", {"reference": "train", "comparison": "test"})
    )
    assert "comparison_rows_not_after_reference" in rule_names(
        two(
            body,
            tr,
            tie,
            "temporal_order",
            {"reference": "train", "comparison": "test", "allow_ties": False},
        )
    )
    idx = two({"ordering": {"field": "index"}}, tbl(["a"], [[1], [2]], ids=[0, 1], split="train"), tbl(["a"], [[1]], ids=[1], split="test"), "temporal_order", {"reference": "train", "comparison": "test", "allow_ties": False})  # fmt: skip
    assert "reference_rows_not_before_comparison" in rule_names(
        idx
    )  # sample index is the declared order
    none = two(
        body,
        tbl(["ts"], [[None]], split="train"),
        ok,
        "temporal_order",
        {"reference": "train", "comparison": "test"},
    )
    assert none.status is Status.INCONCLUSIVE


# -- group comparison and its statistics ---------------------------------------------------------------------------------------------------


def test_group_comparison_reuses_phase_10_statistics_and_corrects_families() -> None:
    n = 40
    tr = tbl(
        ["a", "b", "site"],
        [[None if i < 4 else 1.0, None if i < 20 else 2.0, "x"] for i in range(n)],
        ids=list(range(n)),
        split="train",
    )
    te = tbl(
        ["a", "b", "site"],
        [[None if i < 4 else 1.0, None if i < 2 else 2.0, "x"] for i in range(n)],
        ids=list(range(100, 100 + n)),
        split="test",
    )
    body = {"features": [{"name": "a", "type": "NUMERIC"}, {"name": "b", "type": "NUMERIC"}, {"name": "site", "type": "CATEGORICAL"}], "config": {"min_members": 10, "resamples": 100, "permutations": 200, "correction": "BONFERRONI"}}  # fmt: skip
    r = two(
        body,
        tr,
        te,
        "group_comparison",
        {
            "reference": {"split": "train"},
            "comparison": {"split": "test"},
            "aspects": ["missingness"],
        },
    )
    m = r.observations["missingness"]
    assert (
        m["a"]["difference"] == 0.0
        and m["b"]["reference"]["rate"] == 0.5
        and m["b"]["comparison"]["rate"] == 0.05
    )
    assert m["b"]["difference"] == pytest.approx(-0.45)
    want = st.compare(
        [1.0] * 20 + [0.0] * 20,
        [1.0] * 2 + [0.0] * 38,
        pairing=Pairing.UNPAIRED,
        method="percentile",
        confidence=0.95,
        resamples=100,
        permutations=200,
        seed=0,
    )
    assert m["b"]["comparison_statistics"]["test"]["p_value"] == pytest.approx(want.test.p_value)
    assert m["b"]["comparison_statistics"]["interval"]["estimate"] == pytest.approx(
        want.interval.estimate
    )
    fam = r.statistics["missingness"]
    assert (
        fam["method"] == "BONFERRONI"
        and fam["n_hypotheses"] == 3
        and set(fam["raw_p"]) == {"a", "b", "site"}
    )
    assert fam["adjusted"]["b"] == pytest.approx(
        min(1.0, m["b"]["comparison_statistics"]["test"]["p_value"] * 3)
    )
    assert (
        r.status is Status.PASS
        and r.observations["missingness"]["a"]["comparison_statistics"]["test"]["p_value"] == 1.0
    )  # a difference in evidence is never a verdict


def test_new_invalid_and_rare_categories_are_distinguished_between_groups() -> None:
    ref = tbl(["c"], [["a"]] * 30 + [["b"]] * 30, ids=list(range(60)), split="train")
    cmp = tbl(
        ["c"],
        [["a"]] * 40 + [["b"]] * 50 + [["new"]] * 8 + [["oops"]] * 1 + [["c"]] * 1,
        ids=list(range(100, 200)),
        split="test",
    )
    body = {"features": [{"name": "c", "type": "CATEGORICAL"}]}
    checks_body = {
        "reference": {"split": "train"},
        "comparison": {"split": "test"},
        "aspects": ["categories"],
    }
    plain = two(body, ref, cmp, "group_comparison", checks_body)
    new = {
        x["category"]: x
        for x in plain.observations["categories"]["c"]["observed_new_in_comparison"]
    }
    assert set(new) == {'"new"', '"oops"', '"c"'} and all(
        x["declared_invalid"] is None for x in new.values()
    )  # no schema declared: not judged
    assert plain.status is Status.WARNING and rule_names(plain) == {"observed_new_category"}
    declared = QualitySpec.from_dict({"dataset_id": DS, "splits": ["test", "train"], **body, "checks": [{"type": "categorical", "config": {"allowed": {"c": ["a", "b", "new", "c"]}, "rare_fraction": 0.02}}, {"type": "group_comparison", "config": checks_body}]})  # fmt: skip
    ctx = ck.Ctx(declared, {"train": ref, "test": cmp}, frozenset())
    gc = next(c for c in declared.checks if c.type == "group_comparison")
    r = ck.check_group_comparison(gc, ctx)
    got = {
        x["category"]: x for x in r.observations["categories"]["c"]["observed_new_in_comparison"]
    }
    assert (
        got['"new"']["declared_invalid"] is False and got['"new"']["rare"] is False
    )  # new, allowed, common enough
    assert (
        got['"oops"']["declared_invalid"] is True and got['"oops"']["rare"] is True
    )  # new, forbidden, rare
    assert (
        got['"c"']["declared_invalid"] is False and got['"c"']["rare"] is True
    )  # new, allowed, rare
    assert r.observations["categories"]["c"]["absent_from_comparison"] == []


def test_group_comparison_target_windows_and_refusals() -> None:
    n = 60
    tr = tbl(
        ["ts"],
        [[float(i)] for i in range(n)],
        [0] * 45 + [1] * 15,
        ids=list(range(n)),
        split="train",
    )
    te = tbl(
        ["ts"],
        [[float(i)] for i in range(n, 2 * n)],
        [0] * 15 + [1] * 45,
        ids=list(range(n, 2 * n)),
        split="test",
    )
    body = {
        "target": {"task": "CLASSIFICATION"},
        "ordering": {"field": "feature:ts"},
        "features": [{"name": "ts", "type": "NUMERIC"}],
    }
    r = two(
        body,
        tr,
        te,
        "group_comparison",
        {"reference": {"split": "train"}, "comparison": {"split": "test"}, "aspects": ["target"]},
    )
    t = r.observations["target"]
    assert (
        t["dimension"] == "TARGET"
        and t["measures"]["total_variation_distance"] == pytest.approx(0.5)
        and t["test"]["p_value"] < 0.05
    )
    assert r.statistics["target"]["n_hypotheses"] == 1
    win = two(body, tr, te, "group_comparison", {"reference": {"split": "train", "window": {"start": 0, "end": 30}}, "comparison": {"split": "train", "window": {"start": 30, "end": 60}}, "aspects": ["target"]})  # fmt: skip
    assert win.evidence["n_reference"] == 30 and win.evidence["n_comparison"] == 30
    assert win.observations["target"]["comparison"]["n_valid"] == 30
    overlap = two(body, tr, te, "group_comparison", {"reference": {"split": "train", "window": {"start": 0, "end": 40}}, "comparison": {"split": "train", "window": {"start": 30, "end": 60}}, "aspects": ["target"]})  # fmt: skip
    assert overlap.status is Status.INCONCLUSIVE and "share samples" in overlap.reason
    empty = two(body, tr, te, "group_comparison", {"reference": {"split": "train", "window": {"start": 500, "end": 600}}, "comparison": {"split": "test"}, "aspects": ["target"]})  # fmt: skip
    assert empty.status is Status.INCONCLUSIVE and "no rows" in empty.reason
    small = two(body, tbl(["ts"], [[1.0]] * 4, [0] * 4, split="train", ids=[0, 1, 2, 3]), te, "group_comparison", {"reference": {"split": "train"}, "comparison": {"split": "test"}, "aspects": ["missingness"]})  # fmt: skip
    assert small.status is Status.INCONCLUSIVE and "min_members" in small.reason
    assert (
        small.observations["missingness"]["ts"]["status"] == "INCONCLUSIVE"
        and "comparison_statistics" not in small.observations["missingness"]["ts"]
    )


def test_no_result_carries_a_score_verdict_or_causal_claim() -> None:
    t = tbl(["x"], [[float(i)] for i in range(30)])
    r = run("numeric", {}, t, features=[("x", "NUMERIC")])

    def keys(x: Any) -> set[str]:
        if isinstance(x, dict):
            return set(x) | {k for v in x.values() for k in keys(v)}
        return {k for v in x for k in keys(v)} if isinstance(x, list) else set()

    for banned in (
        "quality_score",
        "score",
        "verdict",
        "grade",
        "rank",
        "overall",
        "bad_data",
        "caused_by",
    ):
        assert banned not in keys(r.to_dict()), banned
    assert "does not say the data is good or bad" in r.note
    assert math.isfinite(
        json.loads(json.dumps(r.to_dict()))["observations"]["features"]["x"]["summary"]["mean"][
            "value"
        ]
    )
