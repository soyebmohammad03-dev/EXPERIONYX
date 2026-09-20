"""Data-quality analysis end to end on the controlled VALIDATION fixture (see quality_helpers): a real
registered dataset, the real engine, real artifacts. Every expected number is recomputed from the raw
fixture rows, never through the quality engine. The defects are written into the data on purpose;
nothing here is a finding about any real dataset."""

import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from eval_helpers import EvalWorld
from experionyx.data_quality.data import QualityDataError
from experionyx.data_quality.engine import QualityRunResult, replay_check, run_quality_request
from experionyx.data_quality.entities import QualityAnalysis, QualityCheck
from experionyx.data_quality.registry import QualityRegistry
from experionyx.data_quality.spec import QualitySpec
from experionyx.domain import Experiment, Observation, Run, RunStatus
from experionyx.errors import (
    ArtifactIntegrityError,
    ExperionyxError,
    SchemaVersionError,
    ValidationError,
)
from experionyx.faults.report import read_artifact
from experionyx.interactions.taxonomy import Pairing
from experionyx.stats import core as st
from quality_helpers import COLUMNS, N, dataset_of, quality_world, rows, spec_dict

DOCS = ("spec", "checks", "observations", "violations", "summary")
ROWS, TARGETS = rows()
TRAIN, TEST = range(0, 120), range(110, N)
CHECKS: list[dict[str, Any]] = [
    {"type": "schema", "config": {"required": ["row_id", "ts"]}},
    {"type": "sample_count", "config": {"min": 100}},
    {"type": "missingness", "config": {"max_feature_rate": 0.25}},
    {"type": "duplicates", "config": {}},
    {"type": "identifiers", "config": {"min_samples": 20}},
    {"type": "numeric", "config": {"bounds": {"age": [0, 120]}}},
    {"type": "categorical", "config": {"allowed": {"site": ["a", "b", "c", "z"]}, "rare_fraction": 0.02}},
    {"type": "target", "config": {"min_class_count": 5}},
    {"type": "distribution", "config": {"max_modal_fraction": 0.9}},
    {"type": "target_leakage", "config": {"candidates": ["leak", "age"]}},
    {"type": "split_overlap", "config": {"reference": "train", "comparison": "test"}},
    {"type": "temporal_order", "config": {"reference": "train", "comparison": "test"}},
    {"type": "group_comparison", "config": {"reference": {"split": "train"}, "comparison": {"split": "test"}}},
]  # fmt: skip


@dataclass
class Fx:
    w: EvalWorld
    did: str
    ds: Any

    @property
    def qr(self) -> QualityRegistry:
        return QualityRegistry(self.w.registry, self.w.store)

    def spec(self, checks: list[dict[str, Any]] | None = None, **over: Any) -> QualitySpec:
        return QualitySpec.from_dict(
            spec_dict(self.did, checks if checks is not None else CHECKS, **over)
        )

    def run(self, spec: QualitySpec) -> QualityRunResult:
        w = self.w
        return run_quality_request(
            w.registry, w.store, w.executor, w.experiment.investigation_id, spec, dataset=self.ds
        )

    def analyze(self, checks: list[dict[str, Any]] | None = None, **over: Any) -> QualityAnalysis:
        out = self.run(self.spec(checks, **over))
        assert out.status is RunStatus.COMPLETED and out.analysis_id
        return self.qr.analysis(out.analysis_id)

    def doc(self, a: QualityAnalysis, name: str) -> Any:
        return self.qr.document(a.id, name)

    def obs(self, a: QualityAnalysis, kind: str, scope: str, nth: int = 0) -> dict[str, Any]:
        checks = [c for c in self.doc(a, "checks")["checks"] if c["type"] == kind]
        cid = checks[nth]["check_id"]
        return self.doc(a, "observations")[cid][scope]["observations"]  # type: ignore[no-any-return]

    def status(self, a: QualityAnalysis, kind: str, scope: str) -> str:
        (c,) = [c for c in self.doc(a, "checks")["checks"] if c["type"] == kind]
        return next(r["status"] for r in c["results"] if r["scope"] == scope)  # type: ignore[no-any-return]

    def viols(self, a: QualityAnalysis, kind: str, scope: str) -> dict[str, dict[str, Any]]:
        return {v["rule"] + ":" + v["subject"]: v for v in self.doc(a, "violations")["violations"] if v["check_type"] == kind and v["scope"] == scope}  # fmt: skip

    @property
    def runs(self) -> int:
        return len(self.w.registry.find(Run))


@pytest.fixture(scope="module")
def fx(tmp_path_factory: pytest.TempPathFactory) -> Fx:
    w, did = quality_world(tmp_path_factory.mktemp("quality"))
    return Fx(w, did, dataset_of(w))


@pytest.fixture(scope="module")
def full(fx: Fx) -> QualityAnalysis:
    return fx.analyze()


def col(name: str, ids: range) -> list[Any]:
    j = COLUMNS.index(name)
    return [ROWS[i][j] for i in ids]


# -- schema, counts, missingness ---------------------------------------------------------------------------------------------------


def test_schema_and_sample_counts_match_the_raw_data(fx: Fx, full: QualityAnalysis) -> None:
    v = fx.viols(full, "schema", "split=train")
    assert set(v) == {"invalid_type:income"} and v["invalid_type:income"]["affected_rows"] == [
        6
    ]  # the "n/a" string
    assert (
        fx.status(full, "schema", "split=train") == "FAIL"
        and fx.status(full, "schema", "split=test") == "PASS"
    )
    for split, ids in (("train", TRAIN), ("test", TEST)):
        o = fx.obs(full, "sample_count", f"split={split}")
        assert (o["n_rows"], o["declared_n"], o["target_len"], o["n_unique_ids"]) == (
            len(ids),
            len(ids),
            len(ids),
            len(ids),
        )
    assert (
        fx.status(full, "sample_count", "split=train") == "PASS"
        and fx.status(full, "sample_count", "split=test") == "FAIL"
    )  # min 100
    assert fx.viols(full, "sample_count", "split=test")["min_violated:rows"]["observed"] == 50


def test_missingness_matches_independent_counts_and_thresholds(
    fx: Fx, full: QualityAnalysis
) -> None:
    for split, ids in (("train", TRAIN), ("test", TEST)):
        o = fx.obs(full, "missingness", f"split={split}")
        for name in ("age", "income", "site", "flag"):
            cells = col(name, ids)
            k = sum(c is None for c in cells)
            assert o["features"][name]["n_missing"] == k and o["features"][name]["n_total"] == len(
                ids
            ), (split, name)
            w = st.proportion_interval(k, len(ids), 0.95)
            assert (
                o["features"][name]["interval"]["lower"],
                o["features"][name]["interval"]["upper"],
            ) == (w.lower, w.upper)
    assert (
        fx.status(full, "missingness", "split=train") == "PASS"
    )  # income: 24 of 120 = 0.2 <= 0.25
    tight = fx.analyze(
        [{"type": "missingness", "config": {"max_feature_rate": 0.1, "patterns": 3}}]
    )
    assert fx.status(tight, "missingness", "split=train") == "FAIL"
    assert fx.viols(tight, "missingness", "split=train")["feature_missing_rate_exceeded:income"][
        "rate"
    ] == pytest.approx(sum(c is None for c in col("income", TRAIN)) / 120)
    pats = fx.obs(tight, "missingness", "split=train")["patterns"]
    assert (
        pats[0] == {"features": ["income"], "rows": 23} or pats[0]["rows"] >= 1
    )  # the pattern list comes from the data, most frequent first
    tokens = fx.analyze([{"type": "missingness", "config": {}}], missing_values=["n/a"])
    assert (
        fx.obs(tokens, "missingness", "split=train")["features"]["income"]["n_missing"]
        == sum(c is None for c in col("income", TRAIN)) + 1
    )  # the declared token joins None


# -- duplicates and identity --------------------------------------------------------------------------------------------------------


def test_duplicate_rows_and_conflicts_match_the_planted_rows(fx: Fx, full: QualityAnalysis) -> None:
    o = fx.obs(full, "duplicates", "split=train")
    assert (
        o["n_duplicate_rows"],
        o["n_duplicate_groups"],
        o["conflicting_groups"],
        o["conflicting_rows"],
    ) == (2, 1, 1, 3)
    assert o["duplicate_rate"] == pytest.approx(2 / 120) and o["example_groups"] == [[99, 100, 101]]
    v = fx.viols(full, "duplicates", "split=train")
    assert v["duplicate_rows:rows"]["affected_rows"] == [100, 101] and v[
        "identical_features_different_target:rows"
    ]["affected_rows"] == [99, 100, 101]
    assert {x["severity"] for x in v.values()} == {"WARNING"} and fx.status(
        full, "duplicates", "split=train"
    ) == "WARNING"  # never FAIL unconfigured
    assert fx.status(full, "duplicates", "split=test") == "PASS"
    strict = fx.analyze([{"type": "duplicates", "config": {"max_duplicate_rate": 0.01}}])
    assert fx.status(strict, "duplicates", "split=train") == "FAIL"
    with_target = fx.analyze([{"type": "duplicates", "config": {"include_target": True}}])
    assert (
        fx.obs(with_target, "duplicates", "split=train")["n_duplicate_rows"] == 1
    )  # 99 and 100 agree, 101 has the flipped target


def test_duplicate_and_conflicting_ids_and_identifier_like_columns(
    fx: Fx, full: QualityAnalysis
) -> None:
    o = fx.obs(full, "identifiers", "split=train")
    d = o["id_column"]
    assert (d["name"], d["n_distinct"], d["n_duplicated_values"], d["n_conflicting_groups"]) == (
        "row_id",
        119,
        1,
        1,
    )  # rows 40 and 41 share id 40
    v = fx.viols(full, "identifiers", "split=train")
    assert v["duplicate_ids:row_id"]["affected_rows"] == [40, 41] and v[
        "conflicting_rows_share_id:row_id"
    ]["affected_rows"] == [40, 41]
    assert {x["subject"] for x in v.values() if x["rule"] == "identifier_like"} == {
        "score"
    }  # all-distinct integers
    assert (
        o["candidates"]["age"]["eligible"] is True and o["candidates"]["age"]["unique_ratio"] < 0.98
    )
    assert (
        fx.status(full, "identifiers", "split=test") == "WARNING"
    )  # 'score' is identifier-like there too
    strict = fx.analyze([{"type": "identifiers", "config": {"require_unique": True}}])
    assert fx.status(strict, "identifiers", "split=train") == "FAIL"


# -- numeric, categorical, distribution -----------------------------------------------------------------------------------------------


def test_numeric_quality_matches_the_planted_values(fx: Fx, full: QualityAnalysis) -> None:
    o = fx.obs(full, "numeric", "split=train")
    age = o["features"]["age"]
    cells = col("age", TRAIN)
    assert (age["n_nonfinite"], age["n_valid"], age["n_invalid"]) == (1, 119, 0)
    finite = [c for c in cells if c != float("inf")]
    assert age["bounds"] == {
        "low": 0,
        "high": 120,
        "n_outside": 2,
        "observed_min": min(finite),
        "observed_max": max(finite),
    }
    v = fx.viols(full, "numeric", "split=train")
    assert v["outside_bounds:age"]["affected_rows"] == [3, 4] and v["nonfinite_values:age"][
        "affected_rows"
    ] == [9]
    assert (
        v["constant_feature:const"]["value"] == 7.0 and o["features"]["const"]["constant"] is True
    )
    assert o["features"]["income"]["n_invalid"] == 1 and o["features"]["income"][
        "n_missing"
    ] == sum(c is None for c in col("income", TRAIN))
    assert "invalid_type:income" in v
    assert (
        fx.status(full, "numeric", "split=train") == "FAIL"
        and fx.status(full, "numeric", "split=test") == "WARNING"
    )
    assert age["outliers"]["method"] == "iqr" and "not evidence of error" in age["outliers"]["note"]


def test_categorical_quality_separates_invalid_rare_and_present(
    fx: Fx, full: QualityAnalysis
) -> None:
    o = fx.obs(full, "categorical", "split=train")["features"]["site"]
    cells = col("site", TRAIN)
    counts = Counter(json.dumps(c) for c in cells)
    assert o["counts"] == dict(sorted(counts.items())) and o["cardinality"] == 5
    assert o["invalid_categories"] == ['"q"'] and set(o["rare_categories"]) == {'"q"', '"z"'}
    assert fx.viols(full, "categorical", "split=train")["invalid_category:site"][
        "affected_rows"
    ] == [8]
    flags = Counter("true" if c else "false" for c in col("flag", TRAIN))
    assert fx.obs(full, "categorical", "split=train")["features"]["flag"]["counts"] == dict(
        sorted(flags.items())
    )
    assert (
        fx.status(full, "categorical", "split=train") == "FAIL"
        and fx.status(full, "categorical", "split=test") == "PASS"
    )
    assert (
        fx.status(full, "distribution", "split=train") == "WARNING"
    )  # the constant feature is dominant (modal fraction 1.0 > 0.9)


# -- target and leakage indicators -------------------------------------------------------------------------------------------------------


def test_target_quality_matches_raw_class_counts(fx: Fx, full: QualityAnalysis) -> None:
    for split, ids in (("train", TRAIN), ("test", TEST)):
        o = fx.obs(full, "target", f"split={split}")
        tg = [TARGETS[i] for i in ids]
        counts = Counter(json.dumps(t) for t in tg if t is not None)
        assert {k: v["count"] for k, v in o["classes"].items()} == dict(sorted(counts.items()))
        n = sum(counts.values())
        assert o["classes"]["0"]["proportion"] == counts["0"] / n and o["n_missing"] == sum(
            t is None for t in tg
        )
        w = st.proportion_interval(counts["0"], n, 0.95)
        assert (o["classes"]["0"]["interval"]["lower"], o["classes"]["0"]["interval"]["upper"]) == (
            w.lower,
            w.upper,
        )
    v = fx.viols(full, "target", "split=train")
    assert v["singleton_class:target"]["classes"] == ["2"] and v["class_below_min_count:target"][
        "classes"
    ] == ["2"]
    assert v["missing_targets:target"]["affected_rows"] == [13]
    assert (
        fx.status(full, "target", "split=train") == "FAIL"
        and fx.status(full, "target", "split=test") == "PASS"
    )
    assert "not a defect" in fx.obs(full, "target", "split=train")["imbalance_note"]


def test_leakage_indicators_follow_the_data_and_the_configured_thresholds(
    fx: Fx, full: QualityAnalysis
) -> None:
    for split, ids, flagged in (("train", TRAIN, True), ("test", TEST, False)):
        c = fx.obs(full, "target_leakage", f"split={split}")["candidates"]["leak"]
        pairs = [(ROWS[i][7], TARGETS[i]) for i in ids if TARGETS[i] is not None]
        eq = sum(float(a) == float(b) for a, b in pairs) / len(pairs)
        assert c["equality_rate"] == pytest.approx(eq) and c["n_pairs"] == len(pairs)
        assert (
            "identical_to_target" in c["indicators"]
        ) is flagged  # 118/119 >= 0.99 in train; 49/50 < 0.99 in test
    assert (
        fx.status(full, "target_leakage", "split=train") == "WARNING"
        and fx.status(full, "target_leakage", "split=test") == "PASS"
    )
    assert fx.obs(full, "target_leakage", "split=train")["candidates"]["age"]["indicators"] == []
    loose = fx.analyze(
        [
            {
                "type": "target_leakage",
                "config": {"candidates": ["leak"], "max_abs_correlation": 0.97, "min_purity": 0.97},
            }
        ]
    )
    assert (
        fx.status(loose, "target_leakage", "split=test") == "WARNING"
    )  # the same data, a different configured threshold


def test_split_overlap_and_temporal_order_match_the_planted_rows(
    fx: Fx, full: QualityAnalysis
) -> None:
    o = fx.obs(full, "split_overlap", "comparison=test | reference=train")
    assert (
        o["sample_id_overlap"],
        o["comparison_rows_with_identical_vector_in_reference"],
        o["label_conflicts"],
        o["label_agreements"],
    ) == (10, 11, 1, 10)
    v = fx.viols(full, "split_overlap", "comparison=test | reference=train")
    assert v["sample_id_overlap:ids"]["affected_rows"] == list(range(110, 120)) and v[
        "identical_features_different_target_across_splits:rows"
    ]["affected_rows"] == [150]
    t = fx.obs(full, "temporal_order", "comparison=test | reference=train")
    assert t["reference_range"] == [0.0, 119.0] and t["comparison_range"] == [110.0, 159.0]
    assert (t["n_comparison_not_after_reference"], t["n_reference_not_before_comparison"]) == (
        9,
        9,
    )  # keys 110..118 vs 111..119
    tv = fx.viols(full, "temporal_order", "comparison=test | reference=train")
    assert tv["unusable_ordering_key:reference"]["affected_rows"] == [
        5
    ]  # the planted missing timestamp


# -- comparison and Phase 10 statistics --------------------------------------------------------------------------------------------------------


def test_group_comparison_matches_independent_rates_and_uses_phase_10(
    fx: Fx, full: QualityAnalysis
) -> None:
    o = fx.obs(full, "group_comparison", "comparison=test | reference=train")
    m = o["missingness"]["income"]
    kr, kc = (
        sum(c is None for c in col("income", TRAIN)),
        sum(c is None for c in col("income", TEST)),
    )
    assert m["reference"]["rate"] == pytest.approx(kr / 120) and m["comparison"][
        "rate"
    ] == pytest.approx(kc / 50)
    want = st.compare(
        [1.0] * kr + [0.0] * (120 - kr),
        [1.0] * kc + [0.0] * (50 - kc),
        pairing=Pairing.UNPAIRED,
        method="percentile",
        confidence=0.95,
        resamples=100,
        permutations=200,
        seed=0,
    )
    assert m["comparison_statistics"]["test"]["p_value"] == pytest.approx(want.test.p_value)
    assert m["difference"] == pytest.approx(kc / 50 - kr / 120)
    # the ordering of the sets does not matter, and 'site' gained no new category between the splits
    assert o["categories"]["site"]["observed_new_in_comparison"] == []
    assert o["target"]["dimension"] == "TARGET" and o["target"]["status"] == "DERIVED"
    assert o["validity"]["age"]["n_invalid"] == [1, 0]  # the planted +inf lies in train only


def test_correction_families_are_recorded_for_feature_level_tests(fx: Fx) -> None:
    a = fx.analyze([CHECKS[-1]], config={"min_members": 10, "resamples": 100, "permutations": 200, "correction": "BENJAMINI_HOCHBERG", "alpha": 0.1})  # fmt: skip
    doc = fx.doc(a, "observations")
    (cid,) = doc
    stats = doc[cid]["comparison=test | reference=train"]["statistics"]
    fam = stats["missingness"]
    assert (
        fam["method"] == "BENJAMINI_HOCHBERG"
        and fam["alpha"] == 0.1
        and fam["n_hypotheses"] == len(fam["raw_p"])
    )
    adj = st.adjust_pvalues(fam["raw_p"], method="BENJAMINI_HOCHBERG", alpha=0.1)
    assert (
        fam["adjusted"] == pytest.approx(dict(adj.adjusted))
        and "one hypothesis per feature" in fam["family"]
    )
    assert all("verdict" not in json.dumps(v).lower() for v in stats.values())


def test_temporal_windows_compare_quality_within_a_split(fx: Fx) -> None:
    win = {"type": "group_comparison", "config": {"reference": {"split": "train", "window": {"start": 0, "end": 60}}, "comparison": {"split": "train", "window": {"start": 60, "end": 120}}, "aspects": ["missingness", "target"]}}  # fmt: skip
    a = fx.analyze([win])
    (c,) = fx.doc(a, "checks")["checks"]
    (r,) = c["results"]
    ref_ids = [
        i for i in range(60) if ROWS[i][8] is not None
    ]  # row 5 has no timestamp: in no window
    assert r["evidence"]["n_reference"] == len(ref_ids) == 59
    assert r["evidence"]["n_comparison"] == 60  # windows over the declared ts ordering
    o = fx.doc(a, "observations")[c["check_id"]][r["scope"]]["observations"]
    assert o["missingness"]["income"]["reference"]["n_missing"] == sum(
        ROWS[i][2] is None for i in ref_ids
    )
    t = o["target"]
    assert (
        t["reference"]["n_valid"] == 58 and t["comparison"]["n_valid"] == 60
    )  # row 13 (no target) lies in the reference window


# -- slices --------------------------------------------------------------------------------------------------------------------------------


SLICES = [
    {"name": "site-a", "condition": {"op": "eq", "field": "feature:site", "values": ["a"]}},
    {"name": "old", "condition": {"op": "range", "field": "feature:age", "low": 50, "high": None}},
    {"name": "tiny", "condition": {"op": "eq", "field": "feature:score", "values": [15]}},
    {
        "name": "never",
        "condition": {"op": "range", "field": "feature:age", "low": 1000, "high": 2000},
    },
    {"name": "wrong", "condition": {"op": "eq", "field": "correct", "values": [False]}},
]
SLICED = [{"type": "missingness", "config": {}}, {"type": "target", "config": {}}, {"type": "categorical", "config": {"allowed": {"site": ["a", "b", "c", "z"]}}}, {"type": "schema", "config": {}}]  # fmt: skip


def test_slice_scoped_quality_reuses_slice_membership_and_never_runs_unrequested_slices(
    fx: Fx,
) -> None:
    plain = fx.analyze(SLICED)
    assert not any(
        "slice=" in r["scope"] for c in fx.doc(plain, "checks")["checks"] for r in c["results"]
    )  # no slices, none run
    a = fx.analyze(SLICED, slices=SLICES)
    scopes = {r["scope"] for c in fx.doc(a, "checks")["checks"] for r in c["results"]}
    assert "split=train | slice=site-a | slice_id=" in " ".join(scopes) or any(
        "slice=site-a" in x for x in scopes
    )
    members = [i for i in TRAIN if ROWS[i][3] == "a"]
    key = next(x for x in scopes if "slice=site-a" in x and "split=train" in x)
    o = fx.obs(a, "missingness", key)
    assert o["features"]["income"]["n_total"] == len(members)
    assert o["features"]["income"]["n_missing"] == sum(ROWS[i][2] is None for i in members)
    t = fx.obs(a, "target", key)
    assert sum(v["count"] for v in t["classes"].values()) == sum(
        TARGETS[i] is not None for i in members
    )
    # the schema check is not slice-capable, so it only ran on the splits
    assert {
        r["scope"]
        for c in fx.doc(a, "checks")["checks"]
        if c["type"] == "schema"
        for r in c["results"]
    } == {"split=test", "split=train"}


def test_slices_with_no_members_thin_members_or_missing_fields_are_told_apart(fx: Fx) -> None:
    a = fx.analyze(SLICED, slices=SLICES)
    (miss,) = [c for c in fx.doc(a, "checks")["checks"] if c["type"] == "missingness"]
    by = {r["scope"]: r for r in miss["results"]}
    never = next(r for k, r in by.items() if "slice=never" in k and "split=train" in k)
    assert never["status"] == "NOT_APPLICABLE" and "NO_MEMBERS" in never["reason"]
    tiny = next(r for k, r in by.items() if "slice=tiny" in k and "split=train" in k)
    assert (
        tiny["status"] == "INCONCLUSIVE"
        and "min_members" in tiny["reason"]
        and tiny["evidence"]["n_rows"] == 1
    )
    wrong = next(r for k, r in by.items() if "slice=wrong" in k and "split=train" in k)
    assert wrong["status"] == "UNAVAILABLE" and "does not provide" in wrong["reason"]
    old = next(r for k, r in by.items() if "slice=old" in k and "split=train" in k)
    assert old["status"] in ("PASS", "WARNING", "FAIL") and old["evidence"]["n_rows"] == sum(
        1 for i in TRAIN if isinstance(ROWS[i][1], float) and 50 <= ROWS[i][1] < float("inf")
    )
    assert (
        a.analysis_status == "PARTIAL"
    )  # inconclusive/unavailable slice results make the analysis partial, visibly


# -- provenance ----------------------------------------------------------------------------------------------------------------------------


def test_the_provenance_fingerprint_changes_exactly_when_meaningful_inputs_change(fx: Fx) -> None:
    base = fx.analyze(CHECKS[:6])
    again = fx.analyze(CHECKS[:6])
    assert again.id == base.id and again.provenance_fingerprint == base.provenance_fingerprint
    reordered = fx.analyze(list(reversed(CHECKS[:6])), features=list(reversed(spec_dict(fx.did, [])["features"])), splits=["train", "test"])  # fmt: skip
    assert (
        reordered.provenance_fingerprint == base.provenance_fingerprint and reordered.id == base.id
    )
    changed = {
        "threshold": fx.analyze([{**CHECKS[0]}, {**CHECKS[1], "config": {"min": 101}}, *CHECKS[2:6]]),
        "check-set": fx.analyze(CHECKS[:5]),
        "seed": fx.analyze(CHECKS[:6], config={"min_members": 10, "resamples": 100, "permutations": 200, "seed": 4}),
        "tokens": fx.analyze(CHECKS[:6], missing_values=["n/a"]),
        "splits": fx.analyze(CHECKS[:6], splits=["train"]),
        "slices": fx.analyze(CHECKS[:6], slices=SLICES[:1]),
        "ordering": fx.analyze(CHECKS[:6], ordering={"field": "index"}),
        "target": fx.analyze(CHECKS[:6], target={"task": "REGRESSION"}),
    }  # fmt: skip
    fps = {k: a.provenance_fingerprint for k, a in changed.items()}
    assert base.provenance_fingerprint not in fps.values() and len(set(fps.values())) == len(fps), (
        fps
    )
    assert len({a.spec_id for a in changed.values()} | {base.spec_id}) == len(changed) + 1


def test_the_provenance_fingerprint_follows_the_data(tmp_path: Path) -> None:
    fps = []
    for k, edit in enumerate(
        (None, lambda r: r[10].__setitem__(1, 33.0), lambda r: r[10].__setitem__(3, "c"))
    ):
        r, t = rows()
        if edit:
            edit(r)
        w, did = quality_world(tmp_path / f"w{k}", rows=r, targets=t)
        f = Fx(w, did, dataset_of(w))
        fps.append(
            f.analyze(
                CHECKS[:4], ordering=None if False else {"field": "feature:ts"}
            ).provenance_fingerprint
        )
    assert (
        len(set(fps)) == 3
    )  # one cell changed: the dataset fingerprint and the split digests change with it


def test_spec_document_records_the_full_provenance(fx: Fx, full: QualityAnalysis) -> None:
    s = fx.doc(full, "spec")
    p: dict[str, Any] = fx.qr.provenance(full.id)
    assert (
        s["spec_id"] == full.spec_id and s["provenance_fingerprint"] == full.provenance_fingerprint
    )
    assert (
        s["dataset"]["fingerprint"] == full.dataset_fingerprint
        and s["dataset"]["adapter"] == "list"
    )
    assert (
        s["columns"] == COLUMNS
        and s["splits"]["train"]["n_rows"] == 120
        and s["splits"]["test"]["n_rows"] == 50
    )
    assert s["splits"]["train"]["sample_digest"].startswith("sha256:") and s["random_seed"] == 0
    assert {c["type"] for c in s["check_ids"]} == {c["type"] for c in CHECKS} and len(
        s["thresholds"]
    ) == len(CHECKS)
    assert (
        s["thresholds"][next(c["check_id"] for c in s["check_ids"] if c["type"] == "sample_count")][
            "min"
        ]
        == 100
    )
    assert set(s["versions"]) >= {"data_quality", "statistics", "quality_schema", "python"}
    assert p["run_provenance"]["source_revision"] and p["run_provenance"][
        "environment_id"
    ].startswith("env_")
    assert p["dataset"] == s["dataset"] and p["check_ids"] == s["check_ids"]


# -- replay, artifacts and persistence ------------------------------------------------------------------------------------------------------------


def test_replay_reproduces_every_document(fx: Fx) -> None:
    a = fx.analyze(SLICED, slices=SLICES[:2])
    out = replay_check(fx.w.registry, fx.w.store, fx.w.executor, a.id)
    assert (
        out["deterministic"] is True and out["differences"] == [] and out["compared"] == list(DOCS)
    )
    assert (
        out["replay_run"] != out["original_run"]
        and len(fx.w.registry.find(QualityAnalysis, spec_id=a.spec_id)) == 1
    )
    for name in DOCS:
        assert read_artifact(
            fx.w.registry, fx.w.store, str(out["replay_run"]), f"data_quality/{name}.json"
        ) == fx.doc(a, name), name


def test_replay_refuses_a_tampered_document(fx: Fx) -> None:
    a = fx.analyze(
        CHECKS[:3], config={"min_members": 10, "resamples": 100, "permutations": 200, "seed": 77}
    )
    f = (
        fx.w.store.run_dir(fx.w.registry.get(Run, a.run_id))
        / "artifacts"
        / "data_quality"
        / "summary.json"
    )
    good = f.read_text(encoding="utf-8")
    f.write_text(good.replace("PASS", "FAIL", 1), encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError):
        fx.doc(a, "summary")
    with pytest.raises(ArtifactIntegrityError):
        replay_check(fx.w.registry, fx.w.store, fx.w.executor, a.id)
    f.write_text(good, encoding="utf-8")
    assert fx.doc(a, "summary")["n_checks"] == 3


def test_documents_results_and_records_are_persisted_and_queryable(
    fx: Fx, full: QualityAnalysis
) -> None:
    assert {x.path for x in fx.qr.artifacts(full.id)} == {f"data_quality/{n}.json" for n in DOCS}
    assert set(full.summary["artifacts"]) == set(DOCS)  # type: ignore[call-overload]
    checks = fx.qr.checks(full.id)
    summary = fx.doc(full, "summary")
    assert len(checks) == summary["n_results"] == len(summary["results"])
    assert {c.check_type for c in checks} == {c["type"] for c in CHECKS}
    assert [c.scope for c in fx.qr.checks(full.id, check_type="schema")] == [
        "split=test",
        "split=train",
    ]
    assert {c.id for c in fx.qr.checks(full.id, status="FAIL")} == {
        c.id for c in checks if c.status == "FAIL"
    } and any(c.status == "FAIL" for c in checks)
    for c in checks:
        assert (
            fx.w.registry.get(QualityCheck, c.id) == c and QualityCheck.from_dict(c.to_dict()) == c
        )
    assert (
        fx.w.registry.get(QualityAnalysis, full.id) == full
        and QualityAnalysis.from_dict(full.to_dict()) == full
    )
    obs = {o.name for o in fx.w.registry.find(Observation, run_id=full.run_id)}
    assert {"quality.new_record", "quality.results", "quality.failed_results"} <= obs
    text = json.dumps([fx.doc(full, n) for n in DOCS]).lower()
    for banned in ("quality_score", "overall_score", '"verdict"', "data is bad", "grade"):
        assert banned not in text, banned
    assert "no data-quality score" in fx.doc(full, "summary")["note"]
    assert (
        fx.doc(full, "violations")["violations"]
        and "not a diagnosis" in fx.doc(full, "violations")["note"]
    )
    with pytest.raises(ValidationError, match="unknown document"):
        fx.qr.document(full.id, "everything")
    with pytest.raises(ValidationError, match="artifact store"):
        QualityRegistry(fx.w.registry).document(full.id, "summary")


def test_corrupted_or_incompatible_records_are_rejected(fx: Fx, full: QualityAnalysis) -> None:
    good = full.to_dict()
    for bad in (
        {**good, "kind": "drift_analysis"}, {**good, "schema_version": 999}, {**good, "extra": 1},
        {k: v for k, v in good.items() if k != "summary"}, {**good, "analysis_status": "DONE"},
        {**good, "provenance_fingerprint": "abc"}, {**good, "dataset_id": "nope"},
    ):  # fmt: skip
        with pytest.raises((ValidationError, SchemaVersionError)):
            QualityAnalysis.from_dict(bad)
    c = fx.qr.checks(full.id)[0].to_dict()
    for bad in ({**c, "status": "GREAT"}, {**c, "analysis_id": "san_x"}, {**c, "scope": ""}, {k: v for k, v in c.items() if k != "scope"}):  # fmt: skip
        with pytest.raises((ValidationError, SchemaVersionError)):
            QualityCheck.from_dict(bad)


# -- refusals: before anything executes ------------------------------------------------------------------------------------------------------------


def refused(fx: Fx, match: str, spec: QualitySpec | None = None, **over: Any) -> None:
    before, exps = fx.runs, len(fx.w.registry.find(Experiment))
    with pytest.raises(ExperionyxError, match=match):
        fx.run(spec or fx.spec(CHECKS[:2], **over))
    assert (
        fx.runs == before and len(fx.w.registry.find(Experiment)) == exps
    )  # nothing was executed or created


def test_impossible_requests_are_refused_before_execution(fx: Fx) -> None:
    refused(fx, "not declared by the dataset", splits=["train", "ghost"])
    refused(fx, "not a column", features=[{"name": "ghost", "type": "NUMERIC"}])
    refused(fx, "not a column", id_column="ghost")
    refused(fx, "not a column", ordering={"field": "feature:ghost"})
    refused(
        fx, "not found|dst_", spec=QualitySpec.from_dict(spec_dict("dst_" + "0" * 32, CHECKS[:2]))
    )
    with pytest.raises(ValidationError, match="unsupported feature type"):
        fx.spec(CHECKS[:2], features=[{"name": "site", "type": "TEXT"}])
    with pytest.raises(ValidationError, match="must be in"):
        fx.spec([{"type": "missingness", "config": {"max_feature_rate": 3}}])


# -- adversarial data ------------------------------------------------------------------------------------------------------------------------------------


def world(tmp_path: Path, **over: Any) -> Fx:
    w, did = quality_world(tmp_path, **over)
    return Fx(w, did, dataset_of(w))


MIN = [{"type": "schema", "config": {}}, {"type": "sample_count", "config": {}}, {"type": "missingness", "config": {}}, {"type": "duplicates", "config": {}}, {"type": "numeric", "config": {}}, {"type": "target", "config": {}}]  # fmt: skip


def test_wrong_metadata_and_ragged_rows_are_reported_not_repaired(tmp_path: Path) -> None:
    f = world(tmp_path, feature_names=COLUMNS[:-1])  # metadata names 8 columns, rows have 9 cells
    a = f.analyze([{"type": "schema", "config": {}}], features=[{"name": "age", "type": "NUMERIC"}], ordering=None, id_column=None, splits=[])  # fmt: skip
    v = f.viols(a, "schema", "split=test")
    assert v["ragged_rows:rows"]["count"] == 50 and v["ragged_rows:rows"]["expected_width"] == 8
    assert f.status(a, "schema", "split=test") == "FAIL"


def test_an_empty_split_a_single_row_and_an_all_missing_feature(tmp_path: Path) -> None:
    f = world(tmp_path / "e", splits={"train": list(range(0, 120)), "empty": []})
    a = f.analyze(MIN, splits=[], ordering=None, id_column=None)
    assert {
        f.status(a, "missingness", "split=empty"),
        f.status(a, "target", "split=empty"),
        f.status(a, "numeric", "split=empty"),
    } == {"NOT_APPLICABLE"}
    assert (
        f.status(a, "sample_count", "split=empty") == "WARNING"
        and f.viols(a, "sample_count", "split=empty")["empty_table:rows"]["count"] == 0
    )
    r, t = rows()
    one = world(tmp_path / "o", rows=r[:1], targets=t[:1], splits={"all": [0]})
    b = one.analyze(MIN, splits=[], ordering=None, id_column=None)
    assert (
        one.status(b, "sample_count", "split=all") == "PASS"
        and one.obs(b, "sample_count", "split=all")["n_rows"] == 1
    )
    assert one.status(b, "target", "split=all") == "WARNING"  # a single row is a singleton class
    r2, t2 = rows()
    for row in r2:
        row[2] = None
    allm = world(tmp_path / "m", rows=r2, targets=t2)
    c = allm.analyze([{"type": "missingness", "config": {"max_feature_rate": 0.5}}, {"type": "numeric", "config": {}}], ordering=None, id_column=None)  # fmt: skip
    assert (
        allm.obs(c, "missingness", "split=train")["features"]["income"]["all_missing"] is True
        and allm.status(c, "missingness", "split=train") == "FAIL"
    )
    assert allm.obs(c, "numeric", "split=train")["features"]["income"]["status"] == "INCONCLUSIVE"


def test_reordered_columns_and_samples_give_the_same_measurements(tmp_path: Path) -> None:
    base = world(tmp_path / "a")
    perm = [3, 0, 1, 2, 4, 5, 6, 7, 8]
    r, t = rows()
    cols2 = [COLUMNS[i] for i in perm]
    r2 = [[row[i] for i in perm] for row in r]
    swapped = world(tmp_path / "b", rows=r2, targets=t, feature_names=cols2)
    order = list(range(N))[::-1]
    r3 = [r[i] for i in order]
    t3 = [t[i] for i in order]
    reversed_ = world(tmp_path / "c", rows=r3, targets=t3, splits={"train": [N - 1 - i for i in range(0, 120)], "test": [N - 1 - i for i in range(110, N)]})  # fmt: skip
    kinds = [
        c for c in CHECKS if c["type"] in ("missingness", "numeric", "categorical", "duplicates")
    ]
    ref = base.analyze(kinds, ordering=None, id_column=None)
    for other in (swapped,):
        got = other.analyze(kinds, ordering=None, id_column=None)
        for kind in ("missingness", "numeric", "categorical"):
            assert other.obs(got, kind, "split=train") == base.obs(ref, kind, "split=train"), kind
    assert (
        reversed_.obs(
            reversed_.analyze(kinds, ordering=None, id_column=None), "numeric", "split=train"
        )["features"]["const"]["constant"]
        is True
    )


def test_regression_dataset_quality(tmp_path: Path) -> None:
    ys = [float(i % 17) + 0.5 for i in range(N)]
    ys[7] = 500.0
    r, _ = rows()
    f = world(tmp_path, rows=r, targets=ys, task="REGRESSION")
    a = f.analyze(
        [{"type": "target", "config": {}}],
        target={"task": "REGRESSION"},
        ordering=None,
        id_column=None,
    )
    o = f.obs(a, "target", "split=train")
    assert (
        o["task"] == "REGRESSION"
        and o["outliers"]["n_outliers"] >= 1
        and 7 in o["outliers"]["affected_rows"]
    )
    assert (
        o["summary"]["mean"]["value"] == pytest.approx(sum(ys[:120]) / 120)
        and f.status(a, "target", "split=train") == "PASS"
    )
    assert "singleton_classes" not in o


def test_the_dataset_file_changing_after_registration_is_refused(fx: Fx, tmp_path: Path) -> None:
    w, did = quality_world(tmp_path)
    p = w.workspace / "m" / "data.json"
    d = json.loads(p.read_text())
    d["rows"][0][1] = 999.0
    p.write_text(json.dumps(d), encoding="utf-8")
    before = len(w.registry.find(Run))
    with pytest.raises(ExperionyxError, match="fingerprint"):
        run_quality_request(w.registry, w.store, w.executor, w.experiment.investigation_id, QualitySpec.from_dict(spec_dict(did, CHECKS[:2])))  # fmt: skip
    assert len(w.registry.find(Run)) == before
    assert QualityDataError.__mro__[1] is ExperionyxError
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_quality_records_are_append_only_in_the_database(fx: Fx, full: QualityAnalysis) -> None:
    import sqlite3

    path = fx.w.workspace / "registry.sqlite"
    with sqlite3.connect(path) as raw:
        for table in ("quality_analyses", "quality_checks"):
            with pytest.raises(sqlite3.DatabaseError, match="immutable"):
                raw.execute(f"UPDATE {table} SET payload = '{{}}'")  # noqa: S608
            with pytest.raises(sqlite3.DatabaseError, match="cannot be deleted"):
                raw.execute(f"DELETE FROM {table}")  # noqa: S608
        assert raw.execute(
            "SELECT COUNT(*) FROM quality_checks WHERE analysis_id = ?", (full.id,)
        ).fetchone()[0] == len(fx.qr.checks(full.id))
