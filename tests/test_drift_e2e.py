"""Drift analysis end to end on the controlled temporal VALIDATION fixture (see drift_helpers): real
evaluation run, real engine, real artifacts. Every expected number is recomputed from the raw rows
or the raw per-sample prediction file, not through the drift engine. The fixture's changes are
written into the data on purpose; nothing here is a finding about any real system."""

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from drift_helpers import HALF, N, dataset_of, drift_world, rows, spec_dict
from eval_helpers import EvalWorld, eval_world
from experionyx.artifacts import LocalArtifactStore
from experionyx.domain import Experiment, Observation, Run, RunStatus
from experionyx.drift import engine
from experionyx.drift.data import DriftDataError, load_frame
from experionyx.drift.engine import DriftRunResult, replay_check, run_drift_request
from experionyx.drift.entities import DriftAnalysis, DriftWindow
from experionyx.drift.registry import DriftRegistry
from experionyx.drift.spec import ShiftSpec
from experionyx.errors import (
    ArtifactIntegrityError,
    ExperionyxError,
    SchemaVersionError,
    ValidationError,
)
from experionyx.slices.data import load_baseline
from experionyx.stats import core as st

PAIR = f"[0, {HALF}) vs [{HALF}, {N})"
DOCS = (
    "spec",
    "windows",
    "feature_results",
    "distribution_results",
    "performance_results",
    "summary",
)


@dataclass
class Fx:
    w: EvalWorld
    base: str
    ds: Any

    @property
    def dr(self) -> DriftRegistry:
        return DriftRegistry(self.w.registry, self.w.store)

    def spec(self, **over: Any) -> ShiftSpec:
        return ShiftSpec.from_dict(spec_dict(self.base, **over))

    def run(self, spec: ShiftSpec) -> DriftRunResult:
        w = self.w
        return run_drift_request(
            w.registry, w.store, w.executor, w.experiment.investigation_id, spec, dataset=self.ds
        )

    def analyze(self, **over: Any) -> DriftAnalysis:
        out = self.run(self.spec(**over))
        assert out.status is RunStatus.COMPLETED and out.analysis_id
        return self.dr.analysis(out.analysis_id)

    def doc(self, a: DriftAnalysis, name: str) -> Any:
        return self.dr.document(a.id, name)

    def pop(self, a: DriftAnalysis, doc: str, pop: str = "POPULATION", pair: str = PAIR) -> Any:
        return self.doc(a, doc)["results"][pair]["populations"][pop]

    @property
    def runs(self) -> int:
        return len(self.w.registry.find(Run))


@pytest.fixture(scope="module")
def fx(tmp_path_factory: pytest.TempPathFactory) -> Fx:
    w, base = drift_world(tmp_path_factory.mktemp("drift"))
    return Fx(w, base, dataset_of(w))


def raw_predictions(fx: Fx) -> dict[int, dict[str, Any]]:
    run = fx.w.registry.get(Run, fx.base)
    path = fx.w.store.run_dir(run) / "artifacts" / "evaluation" / "predictions.jsonl"
    return {r["index"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())}


EARLY, LATE = range(HALF), range(HALF, N)
DATA, TARGETS = rows()


def col(name: str, ids: range) -> list[Any]:
    j = ["x0", "stable", "shifted", "const", "nanheavy", "site", "flag", "ts"].index(name)
    return [DATA[i][j] for i in ids]


def ks(x: list[float], y: list[float]) -> float:
    a, b = np.sort(x), np.sort(y)
    grid = np.unique(np.concatenate([a, b]))
    return float(
        np.max(
            np.abs(
                np.searchsorted(a, grid, "right") / len(a)
                - np.searchsorted(b, grid, "right") / len(b)
            )
        )
    )


def w1(x: list[float], y: list[float]) -> float:
    a, b = np.sort(x), np.sort(y)
    grid = np.unique(np.concatenate([a, b]))
    gap = np.abs(
        np.searchsorted(a, grid, "right") / len(a) - np.searchsorted(b, grid, "right") / len(b)
    )
    return float(np.sum(gap[:-1] * np.diff(grid)))


def jsd(p: list[float], q: list[float]) -> float:
    p_, q_ = np.array(p), np.array(q)
    m = (p_ + q_) / 2

    def kl(a: Any, b: Any) -> float:
        return float(np.sum(a[a > 0] * np.log2(a[a > 0] / b[a > 0])))

    return 0.5 * kl(p_, m) + 0.5 * kl(q_, m)


# -- covariate shift ---------------------------------------------------------------------------------------------------


def test_numeric_covariate_shift_matches_independent_recomputation(fx: Fx) -> None:
    a = fx.analyze()
    res = fx.pop(a, "feature_results")
    for name in ("x0", "stable", "shifted"):
        r = res[name]
        x, y = col(name, EARLY), col(name, LATE)
        assert r["status"] == "DERIVED" and r["kind"] == "NUMERIC" and r["dimension"] == "COVARIATE"
        assert r["measures"]["ks_statistic"] == pytest.approx(ks(x, y), abs=1e-12)
        assert r["measures"]["wasserstein_1"] == pytest.approx(w1(x, y), abs=1e-12)
        loc = r["measures"]["location"]["mean_difference_interval"]
        assert loc["estimate"] == pytest.approx(
            np.mean(y) - np.mean(x), abs=1e-12
        )  # comparison minus reference
        assert (r["reference"]["n_valid"], r["comparison"]["n_valid"]) == (HALF, HALF)
    assert res["shifted"]["measures"]["ks_statistic"] == pytest.approx(0.6) and res["shifted"][
        "measures"
    ]["wasserstein_1"] == pytest.approx(6.0)
    assert res["shifted"]["test"]["p_value"] < 0.01 and res["stable"]["test"]["p_value"] > 0.5
    # the raw p-value, the magnitude and the sample counts are separate fields
    assert set(res["shifted"]) >= {
        "measures",
        "test",
        "multiplicity",
        "reference",
        "comparison",
        "method",
        "status",
    }


def test_identical_and_constant_features_are_reported_as_such(fx: Fx) -> None:
    res = fx.pop(fx.analyze(), "feature_results")
    flag = res["flag"]  # alternates in both windows: the two windows are identical
    assert flag["kind"] == "BOOLEAN" and flag["measures"]["jensen_shannon_divergence"] == 0.0
    assert flag["test"]["p_value"] == 1.0
    const = res["const"]
    assert (
        const["measures"]["ks_statistic"] == 0.0
        and const["measures"]["wasserstein_1_scaled"] is None
    )
    assert any("constant" in w for w in const["warnings"])


def test_missing_and_nonfinite_values_are_counted_never_dropped_silently(fx: Fx) -> None:
    r = fx.pop(fx.analyze(), "feature_results")["nanheavy"]
    for side, ids in (("reference", EARLY), ("comparison", LATE)):
        cells = col("nanheavy", ids)
        c = r[side]
        assert (
            c["n_total"]
            == len(cells)
            == c["n_valid"] + c["n_missing"] + c["n_nonfinite"] + c["n_invalid"]
        )
        assert c["n_missing"] == sum(v is None for v in cells) == 40
        assert c["n_nonfinite"] == sum(v is not None and math.isinf(v) for v in cells) == 5
        assert c["n_valid"] == 35 and c["n_invalid"] == 0
    assert any("excluded" in w for w in r["warnings"])


def test_categorical_shift_matches_independent_proportions_and_flags_new_categories(fx: Fx) -> None:
    r = fx.pop(fx.analyze(), "feature_results")["site"]
    x, y = col("site", EARLY), col("site", LATE)
    cats = sorted(set(x) | set(y))
    cx, cy = Counter(x), Counter(y)
    p = [cx[c] / len(x) for c in cats]
    q = [cy[c] / len(y) for c in cats]
    assert r["measures"]["jensen_shannon_divergence"] == pytest.approx(jsd(p, q), abs=1e-12)
    assert r["measures"]["total_variation_distance"] == pytest.approx(
        0.5 * sum(abs(a - b) for a, b in zip(p, q, strict=True)), abs=1e-12
    )
    rows_ = {c["category"]: c for c in r["measures"]["categories"]}
    for c in cats:
        got = rows_[json.dumps(c)]
        assert (got["n_reference"], got["n_comparison"]) == (cx[c], cy[c])
        w = st.proportion_interval(cy[c], len(y))
        assert (got["interval_comparison"]["lower"], got["interval_comparison"]["upper"]) == (
            w.lower,
            w.upper,
        )
    assert r["measures"]["only_in_comparison"] == ['"d"'] and any(
        "unseen in the reference" in w for w in r["warnings"]
    )
    assert r["test"]["name"] == "jensen_shannon_permutation" and r["test"]["p_value"] < 0.01


# -- label, prediction and performance ----------------------------------------------------------------------------


def test_label_and_prediction_shift_come_from_the_targets_and_the_stored_predictions(
    fx: Fx,
) -> None:
    a = fx.analyze()
    out = fx.pop(a, "distribution_results")
    preds = raw_predictions(fx)
    for dim, early, late in (
        ("label", [TARGETS[i] for i in EARLY], [TARGETS[i] for i in LATE]),
        (
            "prediction",
            [preds[i]["predicted"] for i in EARLY],
            [preds[i]["predicted"] for i in LATE],
        ),
    ):
        r = out[dim]
        assert r["status"] == "DERIVED" and r["kind"] == "CATEGORICAL"
        cats = {c["category"]: c for c in r["measures"]["categories"]}
        assert cats["1"]["n_reference"] == sum(early) and cats["1"]["n_comparison"] == sum(late)
        assert cats["1"]["difference"] == pytest.approx(
            sum(late) / HALF - sum(early) / HALF
        )  # explicit proportion difference
        w = st.proportion_interval(sum(early), HALF)
        assert (
            cats["1"]["interval_reference"]["lower"],
            cats["1"]["interval_reference"]["upper"],
        ) == (w.lower, w.upper)
    assert out["label"]["dimension"] == "LABEL" and out["prediction"]["dimension"] == "PREDICTION"
    conf = out["confidence"]  # the stored per-sample confidence, treated as a numeric distribution
    assert (
        conf["kind"] == "NUMERIC"
        and conf["status"] == "DERIVED"
        and conf["dimension"] == "CONFIDENCE"
    )
    ce, cl = [preds[i]["confidence"] for i in EARLY], [preds[i]["confidence"] for i in LATE]
    assert conf["measures"]["ks_statistic"] == pytest.approx(ks(ce, cl), abs=1e-12)
    assert conf["measures"]["wasserstein_1"] == pytest.approx(w1(ce, cl), abs=1e-12)


def test_performance_drift_reuses_the_evaluation_metrics(fx: Fx) -> None:
    a = fx.analyze()
    p = fx.pop(a, "performance_results")
    preds = raw_predictions(fx)

    def acc(ids: range) -> float:
        return float(sum(preds[i]["true"] == preds[i]["predicted"] for i in ids) / len(ids))

    assert p["measure"] == "accuracy" and p["task"] == "CLASSIFICATION" and p["status"] == "DERIVED"
    m = {x["metric_id"]: x for x in p["metrics"]}["accuracy"]
    assert m["reference_value"] == pytest.approx(acc(EARLY)) and m[
        "comparison_value"
    ] == pytest.approx(acc(LATE))
    assert m["difference"] == pytest.approx(acc(LATE) - acc(EARLY)) and m["change"] == "LOWER"
    assert m["higher_is_better"] is True and m["worse_on_this_metric"] is True
    assert m["reference_interval"]["seed"] == 0 and m["reference_interval"]["resamples"] == 100
    assert (
        p["reference"]["n_samples"] == p["comparison"]["n_samples"] == HALF
        and p["reference"]["status"] == "OBSERVED"
    )
    inf = p["inference"]
    assert (
        inf["measure"] == "accuracy"
        and inf["n_a"] == inf["n_b"] == HALF
        and inf["inference"]["pairing"] == "UNPAIRED"
    )
    assert inf["descriptive"]["difference"] == pytest.approx(
        acc(LATE) - acc(EARLY)
    )  # a = comparison, b = reference
    assert {x["metric_id"] for x in p["metrics"]} >= {"accuracy", "f1_macro"} or len(
        p["metrics"]
    ) > 1
    assert "unpaired" in p["note"] and "not shown to be caused" in p["note"]


def test_equal_metrics_are_reported_as_equal_and_lower_is_better_metrics_flip_the_reading(
    fx: Fx,
) -> None:
    a = fx.analyze(reference={"start": 0, "end": 40}, comparisons=[{"start": 40, "end": 80}], config={"min_samples": 10, "resamples": 100, "permutations": 200})  # fmt: skip
    p = fx.pop(a, "performance_results", pair="[0, 40) vs [40, 80)")
    by = {x["metric_id"]: x for x in p["metrics"]}
    assert by["accuracy"]["change"] == "EQUAL" and by["accuracy"]["worse_on_this_metric"] is None
    hib = {k: v["higher_is_better"] for k, v in by.items()}
    lower_better = [k for k, h in hib.items() if not h and by[k]["difference"] not in (None, 0)]
    for k in lower_better:  # a metric where lower is better: a positive difference is WORSE
        assert by[k]["worse_on_this_metric"] is (by[k]["difference"] > 0)


# -- windows, ordering, skipped windows ------------------------------------------------------------------------------


def test_windows_document_records_identity_members_digests_and_ordering(fx: Fx) -> None:
    a = fx.analyze()
    w = fx.doc(a, "windows")
    assert (
        w["ordering"] == {"field": "index", "unique": False}
        and w["n_baseline_samples"] == N
        and w["n_with_ordering_key"] == N
    )
    spec = fx.spec()
    ref, cmp = spec.reference, spec.comparisons[0]
    assert ref is not None
    ids = {ref.window_id("index"), cmp.window_id("index")}
    assert set(w["windows"]) == ids
    rw = w["windows"][ref.window_id("index")]
    assert (
        rw["sample_ids"] == list(EARLY)
        and rw["n_samples"] == HALF
        and rw["order_min"] == 0
        and rw["order_max"] == HALF - 1
    )
    assert (
        w["windows"][cmp.window_id("index")]["window"]["role"] == "COMPARISON"
        and rw["window"]["role"] == "REFERENCE"
    )
    assert w["pairs"] == [
        {
            "key": PAIR,
            "reference_window_id": ref.window_id("index"),
            "comparison_window_id": cmp.window_id("index"),
        }
    ]
    assert fx.dr.window(ref.window_id("index")).window().window_id("index") == ref.window_id(
        "index"
    )  # registered under the same id
    assert w["skipped"] == []
    ev = fx.doc(a, "performance_results")["results"][PAIR]["evidence"]
    assert (
        ev["reference_window_id"] == ref.window_id("index")
        and ev["n_reference"] == HALF
        and ev["baseline_run_id"] == fx.base
    )
    assert (
        ev["reference_digest"] == rw["sample_digest"]
        and ev["dataset_fingerprint"] == a.dataset_fingerprint
    )


def test_a_timestamp_feature_gives_the_same_numbers_under_different_identities(fx: Fx) -> None:
    by_index = fx.analyze()
    by_ts = fx.analyze(ordering={"field": "feature:ts"}, reference={"start": 0, "end": HALF}, comparisons=[{"start": HALF, "end": N}])  # fmt: skip
    assert by_ts.id != by_index.id and by_ts.spec_id != by_index.spec_id
    assert by_ts.provenance_fingerprint != by_index.provenance_fingerprint
    a, b = fx.pop(by_index, "feature_results"), fx.pop(by_ts, "feature_results")
    for name in a:
        assert a[name]["measures"] == b[name]["measures"] and a[name]["test"] == b[name]["test"]
    wi, wt = fx.doc(by_index, "windows"), fx.doc(by_ts, "windows")
    assert (
        set(wi["windows"]) & set(wt["windows"]) == set()
    )  # the ordering is part of a window's identity
    assert {v["sample_digest"] for v in wi["windows"].values()} == {
        v["sample_digest"] for v in wt["windows"].values()
    }


def test_window_boundaries_are_inclusive_or_exclusive_exactly_as_declared(fx: Fx) -> None:
    a = fx.analyze(reference={"start": 0, "end": 40}, comparisons=[{"start": 40, "end": 80, "start_inclusive": False, "end_inclusive": True}], config={"min_samples": 10, "resamples": 100, "permutations": 200})  # fmt: skip
    w = fx.doc(a, "windows")
    sizes = sorted(v["n_samples"] for v in w["windows"].values())
    assert sizes == [40, 40]
    cmp = next(v for v in w["windows"].values() if v["window"]["role"] == "COMPARISON")
    assert (
        cmp["sample_ids"] == list(range(41, 81))
        and cmp["order_min"] == 41
        and cmp["order_max"] == 80
    )


def test_rolling_windows_preserve_boundaries_step_and_list_what_was_skipped(fx: Fx) -> None:
    spec: dict[str, Any] = {"reference": None, "comparisons": [], "rolling": {"start": 0, "stop": 170, "width": 40, "step": 40, "reference": "PREVIOUS"}, "config": {"min_samples": 10, "resamples": 50, "permutations": 100, "dimensions": ["label", "performance", "prediction"]}, "features": []}  # fmt: skip
    a = fx.analyze(**spec)
    w = fx.doc(a, "windows")
    assert [p["key"] for p in w["pairs"]] == [
        "[0, 40) vs [40, 80)",
        "[40, 80) vs [80, 120)",
        "[80, 120) vs [120, 160)",
    ]
    assert sorted((s["window"]["start"], s["reason"]) for s in w["skipped"]) == [
        (0, "NO_REFERENCE"),
        (160, "PARTIAL_WINDOW"),
    ]
    assert (
        a.analysis_status == "PARTIAL" and dict(a.summary)["n_skipped_windows"] == 2
    )  # skipped windows make the analysis partial, visibly
    assert (
        len(w["windows"]) == 6
    )  # each pair's two windows; the same interval as reference and as comparison has two identities (role is identity)
    res = fx.doc(a, "performance_results")["results"]
    assert sorted(res) == sorted(p["key"] for p in w["pairs"])
    drops = [res[k]["populations"]["POPULATION"]["metrics"][0]["difference"] for k in sorted(res)]
    assert drops[0] == 0 or abs(drops[0]) < 0.3  # the fixture is stable until the late half
    assert (
        res["[40, 80) vs [80, 120)"]["populations"]["POPULATION"]["comparison"]["n_samples"] == 40
    )


def test_expanding_references_and_generated_empty_windows(fx: Fx) -> None:
    a = fx.analyze(reference=None, comparisons=[], rolling={"start": 0, "stop": 240, "width": 80, "step": 80, "reference": "EXPANDING"}, features=[], config={"min_samples": 10, "resamples": 50, "permutations": 100, "dimensions": ["label", "performance", "prediction"]})  # fmt: skip
    w = fx.doc(a, "windows")
    assert [p["key"] for p in w["pairs"]] == [
        "[0, 80) vs [80, 160)"
    ]  # [160, 240) is empty: nothing there
    reasons = sorted((s["window"]["start"], s["reason"]) for s in w["skipped"])
    assert reasons == [(0, "NO_REFERENCE"), (160, "EMPTY_WINDOW")]
    assert all("detail" in s for s in w["skipped"])


def test_missing_ordering_keys_exclude_samples_and_are_recorded(tmp_path: Path) -> None:
    import drift_helpers as h

    r, t = h.rows()
    for i in (3, 90, 91):
        r[i][7] = None  # no timestamp
    r[100][7] = float("inf")
    r[101][7] = "later"
    w = eval_world(
        tmp_path,
        model={"threshold": 5.0, "proba": True},
        data={**h.data(), "rows": r, "targets": t},
        config=h.EVAL,
    )
    base = w.run().run.id
    f = Fx(w, base, h.dataset_of(w))
    a = f.analyze(ordering={"field": "feature:ts"}, features=[], config={"min_samples": 10, "resamples": 50, "permutations": 100})  # fmt: skip
    win = f.doc(a, "windows")
    assert win["n_with_ordering_key"] == N - 5 and win["order_excluded"]["MISSING"][
        "sample_ids"
    ] == [3, 90, 91]
    assert win["order_excluded"]["NON_FINITE"]["n"] == 1 and win["order_excluded"]["INVALID"][
        "sample_ids"
    ] == [101]
    members = {i for v in win["windows"].values() for i in v["sample_ids"]}
    assert not members & {3, 90, 91, 100, 101}  # belong to no window; never placed by position


# -- refusals: before anything executes --------------------------------------------------------------------------------


def refused(fx: Fx, match: str, **over: Any) -> None:
    before, exps = fx.runs, len(fx.w.registry.find(Experiment))
    with pytest.raises(ExperionyxError, match=match):
        fx.run(fx.spec(**over))
    assert (
        fx.runs == before and len(fx.w.registry.find(Experiment)) == exps
    )  # nothing was executed or created


def test_unsupported_or_impossible_requests_are_refused_before_execution(fx: Fx) -> None:
    refused(fx, "no sample's index key falls inside", reference={"start": 1000, "end": 1100})
    refused(fx, "no sample's index key falls inside", comparisons=[{"start": 500, "end": 600}])
    refused(fx, "not a feature of this dataset", features=[{"name": "nope", "type": "NUMERIC"}], config={"min_samples": 10})  # fmt: skip
    refused(fx, "not a feature of this dataset", ordering={"field": "feature:nope"})
    refused(fx, "unsupported feature type", features=[{"name": "site", "type": "TEXT"}])
    refused(fx, "overlaps", reference={"start": 0, "end": 100})
    refused(fx, "not both", comparisons=[{"start": 80, "end": 160}], rolling={"start": 0, "stop": 10, "width": 5, "step": 5})  # fmt: skip
    refused(fx, "at least one|min_samples", config={"min_samples": 1})
    refused(fx, "not a feature of this dataset", slices=[{"name": "s", "condition": {"op": "eq", "field": "feature:nope", "values": [1]}}])  # fmt: skip
    refused(fx, "fmd_", failure_modes=["fmd_" + "0" * 32])


def test_duplicate_ordering_keys_are_refused_only_when_uniqueness_is_required(
    tmp_path: Path,
) -> None:
    import drift_helpers as h

    r, t = h.rows()
    for i in range(0, N, 2):
        r[i][7] = float(i)
        if i + 1 < N:
            r[i + 1][7] = float(i)  # every timestamp occurs twice
    w = eval_world(
        tmp_path,
        model={"threshold": 5.0, "proba": True},
        data={**h.data(), "rows": r, "targets": t},
        config=h.EVAL,
    )
    f = Fx(w, w.run().run.id, h.dataset_of(w))
    refused(f, "required to be unique", ordering={"field": "feature:ts", "unique": True})
    a = f.analyze(ordering={"field": "feature:ts"}, features=[], config={"min_samples": 10, "resamples": 50, "permutations": 100})  # ties are legitimate: membership is by key value  # fmt: skip
    assert sum(v["n_samples"] for v in f.doc(a, "windows")["windows"].values()) == N


def test_an_unfinished_baseline_and_missing_evidence_are_refused(fx: Fx) -> None:
    with pytest.raises(ExperionyxError):
        engine.validate(fx.w.registry, ShiftSpec.from_dict(spec_dict("run_" + "9" * 32)))
    base = load_baseline(fx.w.registry, fx.w.store, fx.base)
    with pytest.raises(ExperionyxError, match="need the dataset"):
        load_frame(base, None, fx.spec())


# -- insufficient evidence ----------------------------------------------------------------------------------------------------


def test_small_windows_keep_measures_but_withhold_inference(fx: Fx) -> None:
    a = fx.analyze(reference={"start": 0, "end": 8}, comparisons=[{"start": 80, "end": 88}])
    assert a.analysis_status == "PARTIAL"
    pair = "[0, 8) vs [80, 88)"
    res = fx.pop(a, "feature_results", pair=pair)
    assert all(r["status"] == "INSUFFICIENT_EVIDENCE" and r["test"] is None for r in res.values())
    assert res["shifted"]["measures"]["ks_statistic"] == pytest.approx(
        0.75
    )  # described, not withheld: [0..7] vs [6..13]
    assert (
        "min_samples=10" in res["shifted"]["reason"]
        and "location" not in res["shifted"]["measures"]
    )
    perf = fx.pop(a, "performance_results", pair=pair)
    assert (
        perf["status"] == "INSUFFICIENT_EVIDENCE"
        and perf["metrics"]
        and "min_samples" in perf["reason"]
    )
    fam = fx.doc(a, "feature_results")["multiple_comparisons"]
    assert all(
        f["n_hypotheses"] == 0 for f in fam.values()
    )  # no valid p-value, so nothing enters a family
    assert fx.doc(a, "summary")["status_counts"].get("INSUFFICIENT_EVIDENCE", 0) >= 7


# -- multiple comparisons in a real analysis -------------------------------------------------------------------------------


def test_feature_family_correction_is_recorded_per_result_and_per_family(fx: Fx) -> None:
    base = fx.analyze()
    for method in ("BONFERRONI", "BENJAMINI_HOCHBERG"):
        a = fx.analyze(
            config={
                "min_samples": 10,
                "resamples": 100,
                "permutations": 200,
                "correction": method,
                "alpha": 0.01,
            }
        )
        res = fx.doc(a, "feature_results")
        r = res["results"][PAIR]["populations"]["POPULATION"]
        ((_, fam),) = res["multiple_comparisons"].items()
        assert fam["method"] == method and fam["alpha"] == 0.01 and fam["n_hypotheses"] == 7
        assert fam["definition"].startswith("the hypothesis tests of one comparison family")
        raw = {n: x["multiplicity"]["raw_p"] for n, x in r.items()}
        assert raw == {
            n: x["test"]["p_value"] for n, x in r.items()
        }  # the raw p-value is preserved
        adj = st.adjust_pvalues(
            {f"{PAIR} | {n}": p for n, p in raw.items()}, method=method, alpha=0.01
        )
        for n, x in r.items():
            m = x["multiplicity"]
            assert (
                m["adjusted_p"] == pytest.approx(adj.adjusted[f"{PAIR} | {n}"])
                and m["method"] == method
            )
            assert m["adjusted_p_below_alpha"] is adj.rejected[f"{PAIR} | {n}"]
            assert (
                x["status"] == "DERIVED"
            )  # a p-value never turns a result into a "drift detected" verdict
    none = fx.pop(base, "feature_results")
    assert all(
        x["multiplicity"]["method"] == "NONE"
        and x["multiplicity"]["adjusted_p_below_alpha"] is None
        for x in none.values()
    )
    assert all(x["multiplicity"]["adjusted_p"] == x["multiplicity"]["raw_p"] for x in none.values())


# -- slices ------------------------------------------------------------------------------------------------------------------------------


SLICES = [
    {"name": "hi-x0", "condition": {"op": "range", "field": "feature:x0", "low": 7, "high": None}},
    {"name": "class-1", "condition": {"op": "eq", "field": "target", "values": [1]}},
    {
        "name": "late-only",
        "condition": {"op": "range", "field": "feature:x0", "low": 9.25, "high": None},
    },
    {"name": "tiny", "condition": {"op": "eq", "field": "feature:x0", "values": [9]}},
    {"name": "never", "condition": {"op": "range", "field": "feature:x0", "low": 100, "high": 200}},
    {"name": "wrong", "condition": {"op": "eq", "field": "correct", "values": [False]}},
]


def test_slice_membership_is_evaluated_independently_in_each_window(fx: Fx) -> None:
    a = fx.analyze(slices=SLICES)
    w = fx.doc(a, "windows")
    spec = fx.spec(slices=SLICES)
    ref, cmp = (x.window_id("index") for x in (spec.reference, spec.comparisons[0]))  # type: ignore[union-attr]
    x0 = col("x0", range(N))
    preds = raw_predictions(fx)
    want = {
        "hi-x0": (sum(x0[i] >= 7 for i in EARLY), sum(x0[i] >= 7 for i in LATE)),
        "class-1": (sum(TARGETS[i] == 1 for i in EARLY), sum(TARGETS[i] == 1 for i in LATE)),
        "late-only": (0, sum(x0[i] >= 9.25 for i in LATE)),
        "tiny": (sum(x0[i] == 9 for i in EARLY), sum(x0[i] == 9 for i in LATE)),
        "never": (0, 0),
        "wrong": (sum(preds[i]["true"] != preds[i]["predicted"] for i in EARLY), sum(preds[i]["true"] != preds[i]["predicted"] for i in LATE)),
    }  # fmt: skip
    for name, (n_ref, n_cmp) in want.items():
        got = w["slices"][name]["windows"]
        assert (got[ref]["n_members"], got[cmp]["n_members"]) == (n_ref, n_cmp), name
        assert got[ref]["n_window_samples"] == HALF and got[ref]["status"] == (
            "COMPUTED" if n_ref else "EMPTY"
        )
    assert want["hi-x0"][0] != want["hi-x0"][1]  # membership is not assumed identical over time
    assert "independently inside each window" in w["membership_note"]
    # identity and digests are recorded per slice and window
    assert (
        w["slices"]["hi-x0"]["slice_id"]
        == spec.slices[[s.name for s in spec.slices].index("hi-x0")].slice_id
    )


def test_slice_results_distinguish_no_members_from_insufficient_evidence(fx: Fx) -> None:
    a = fx.analyze(slices=SLICES)
    perf = fx.doc(a, "performance_results")["results"][PAIR]["populations"]
    assert perf["never"]["status"] == "UNDEFINED" and perf["never"]["reason_code"] == "NO_MEMBERS"
    assert "absence of samples" in perf["never"]["reason"]
    assert (
        perf["late-only"]["reason_code"] == "NO_MEMBERS"
        and "reference window" in perf["late-only"]["reason"]
    )
    assert (perf["late-only"]["n_reference"], perf["late-only"]["n_comparison"]) == (0, 8)
    tiny = perf["tiny"]  # 8 members per window < min_samples=10: present, thin
    assert (
        tiny["status"] == "INSUFFICIENT_EVIDENCE"
        and "reason_code" not in tiny
        and tiny["comparison"]["n_samples"] == 8
    )
    assert perf["hi-x0"]["status"] == "DERIVED" and perf["POPULATION"]["status"] == "DERIVED"
    assert (
        a.analysis_status == "PARTIAL" and dict(a.summary)["stubs"] == 3
    )  # never, late-only and wrong (the baseline made no early errors)
    feats = fx.doc(a, "feature_results")["results"][PAIR]["populations"]
    assert (
        feats["never"]["reason_code"] == "NO_MEMBERS"
        and feats["hi-x0"]["x0"]["reference"]["n_valid"] == 24
    )


def test_overlapping_slices_are_analyzed_independently_and_membership_is_reported(fx: Fx) -> None:
    a = fx.analyze(slices=SLICES[:2])
    perf = fx.doc(a, "performance_results")["results"][PAIR]["populations"]
    assert (
        perf["hi-x0"]["reference"]["n_samples"] == 24
        and perf["hi-x0"]["comparison"]["n_samples"] == 48
    )
    late_both = [i for i in LATE if DATA[i][0] >= 7 and TARGETS[i] == 1]
    # the two slices genuinely overlap in the late window without being equal
    assert 0 < len(late_both) < 48
    assert len(late_both) < sum(TARGETS[i] == 1 for i in LATE)
    assert perf["class-1"]["reference"]["n_samples"] == sum(TARGETS[i] == 1 for i in EARLY)


def test_slice_windows_feed_the_slice_scoped_covariate_and_label_results(fx: Fx) -> None:
    a = fx.analyze(slices=SLICES[:1])
    pops = fx.doc(a, "distribution_results")["results"][PAIR]["populations"]
    lab = pops["hi-x0"]["label"]
    early = [TARGETS[i] for i in EARLY if DATA[i][0] >= 7]
    late = [TARGETS[i] for i in LATE if DATA[i][0] >= 7]
    cats = {c["category"]: c for c in lab["measures"]["categories"]}
    assert (lab["reference"]["n_valid"], lab["comparison"]["n_valid"]) == (len(early), len(late))
    assert (cats["1"]["n_reference"], cats["1"]["n_comparison"]) == (sum(early), sum(late))


# -- provenance ---------------------------------------------------------------------------------------------------------------------------


def test_the_provenance_fingerprint_changes_exactly_when_meaningful_inputs_change(fx: Fx) -> None:
    base = fx.analyze()
    again = fx.analyze()
    assert (
        again.id == base.id and again.provenance_fingerprint == base.provenance_fingerprint
    )  # an identical request is the same record
    same_features_reordered = fx.analyze(features=list(reversed(spec_dict(fx.base)["features"])))
    assert (
        same_features_reordered.provenance_fingerprint == base.provenance_fingerprint
        and same_features_reordered.id == base.id
    )
    cfg = {"min_samples": 10, "resamples": 100, "permutations": 200}
    changed = {
        "seed": fx.analyze(config={**cfg, "seed": 1}),
        "correction": fx.analyze(config={**cfg, "correction": "BONFERRONI"}),
        "permutations": fx.analyze(config={**cfg, "permutations": 300}),
        "min_samples": fx.analyze(config={**cfg, "min_samples": 12}),
        "metrics": fx.analyze(config={**cfg, "metrics": ["accuracy"]}),
        "window": fx.analyze(comparisons=[{"start": HALF, "end": N - 10}]),
        "features": fx.analyze(features=spec_dict(fx.base)["features"][:3]),
        "slices": fx.analyze(slices=SLICES[:1]),
        "ordering": fx.analyze(ordering={"field": "feature:ts"}),
    }
    fps = {k: a.provenance_fingerprint for k, a in changed.items()}
    assert base.provenance_fingerprint not in fps.values() and len(set(fps.values())) == len(fps), (
        fps
    )
    assert len({a.spec_id for a in changed.values()} | {base.spec_id}) == len(changed) + 1


def test_the_provenance_fingerprint_follows_the_data_and_the_model(tmp_path: Path) -> None:
    import drift_helpers as h

    w1, b1 = drift_world(tmp_path / "a")
    r, t = h.rows()
    r[10][2] += 1.0  # one cell of one feature differs
    w2 = eval_world(
        tmp_path / "b",
        model={"threshold": 5.0, "proba": True},
        data={**h.data(), "rows": r, "targets": t},
        config=h.EVAL,
    )
    w3 = eval_world(
        tmp_path / "c", model={"threshold": 4.0, "proba": True}, data=h.data(), config=h.EVAL
    )
    fps = []
    for w, b in ((w1, b1), (w2, w2.run().run.id), (w3, w3.run().run.id)):
        f = Fx(w, b, h.dataset_of(w))
        fps.append(f.analyze().provenance_fingerprint)
    assert len(set(fps)) == 3  # dataset content and model identity are inputs to the fingerprint


def test_spec_document_records_the_full_provenance(fx: Fx) -> None:
    a = fx.analyze()
    s = fx.doc(a, "spec")
    p: dict[str, Any] = fx.dr.provenance(a.id)
    assert s["spec_id"] == a.spec_id and s["provenance_fingerprint"] == a.provenance_fingerprint
    assert s["dataset_fingerprint"] == a.dataset_fingerprint and s["baseline_run"] == fx.base
    assert s["model_fingerprint"].startswith("sha256:") and s["evaluation_config_hash"].startswith(
        "sha256:"
    )
    assert s["ordering"] == {"field": "index", "unique": False} and s["random_seed"] == 0
    assert s["features"]["site"] == "CATEGORICAL" and s["features"]["flag"] == "BOOLEAN"
    assert set(s["versions"]) >= {"drift_analysis", "statistics", "slice_analysis", "python"}
    assert s["methods"]["correction"] == "NONE" and "ks_permutation test" in s["methods"]["numeric"]
    assert s["window_ids"] and s["dimensions"] == [
        "covariate",
        "label",
        "performance",
        "prediction",
    ]
    assert p["model_fingerprint"] == s["model_fingerprint"] and p["window_ids"] == s["window_ids"]
    assert p["run_provenance"]["source_revision"] and p["run_provenance"][
        "environment_id"
    ].startswith("env_")


# -- replay --------------------------------------------------------------------------------------------------------------------------------


def test_replay_reproduces_every_document(fx: Fx) -> None:
    a = fx.analyze(slices=SLICES[:2])
    out = replay_check(fx.w.registry, fx.w.store, fx.w.executor, a.id)
    assert (
        out["deterministic"] is True and out["differences"] == [] and out["compared"] == list(DOCS)
    )
    assert out["replay_run"] != out["original_run"] and out["replay_status"] == "COMPLETED"
    assert (
        len(fx.w.registry.find(DriftAnalysis, spec_id=a.spec_id)) == 1
    )  # replay adds no second analysis
    replayed = fx.dr.provenance(a.id)  # the record still points at its own run
    assert replayed["run_id"] == a.run_id
    # the replay's own documents equal the original's, value for value
    from experionyx.faults.report import read_artifact

    for name in DOCS:
        assert read_artifact(
            fx.w.registry, fx.w.store, str(out["replay_run"]), f"drift/{name}.json"
        ) == fx.doc(a, name), name


def test_replay_detects_a_tampered_or_changed_input(fx: Fx) -> None:
    a = fx.analyze(comparisons=[{"start": HALF, "end": N - 5}])
    run = fx.w.registry.get(Run, a.run_id)
    f = fx.w.store.run_dir(run) / "artifacts" / "drift" / "summary.json"
    good = f.read_text(encoding="utf-8")
    f.write_text(good.replace("DERIVED", "OBSERVED", 1), encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError):
        fx.doc(a, "summary")  # the digest recorded at write time no longer matches
    with pytest.raises(ArtifactIntegrityError):
        replay_check(fx.w.registry, fx.w.store, fx.w.executor, a.id)
    f.write_text(good, encoding="utf-8")
    assert fx.doc(a, "summary")["n_pairs"] == 1  # restored


# -- artifacts and persistence ---------------------------------------------------------------------------------------------------------


def test_every_document_is_stored_registered_and_digest_verified(fx: Fx) -> None:
    a = fx.analyze()
    stored = {x.path for x in fx.dr.artifacts(a.id)}
    assert stored == {f"drift/{n}.json" for n in DOCS}
    assert set(a.summary["artifacts"]) == set(DOCS)  # type: ignore[call-overload]
    for n in DOCS:
        assert isinstance(fx.doc(a, n), dict)
    with pytest.raises(ValidationError, match="unknown document"):
        fx.dr.document(a.id, "everything")
    text = json.dumps([fx.doc(a, n) for n in DOCS]).lower()
    assert "drift_score" not in text and "overall_score" not in text and 'verdict"' not in text
    assert "concept drift" in json.dumps(fx.doc(a, "summary"))  # stated as a limit
    assert "no composite drift score" in fx.doc(a, "summary")["note"]
    obs = {o.name for o in fx.w.registry.find(Observation, run_id=a.run_id)}
    assert {
        "drift.new_record",
        "drift.window_pairs",
        "drift.skipped_windows",
        "drift.populations",
    } <= obs


def test_records_round_trip_and_reject_corruption(fx: Fx) -> None:
    a = fx.analyze()
    assert fx.w.registry.get(DriftAnalysis, a.id) == a and DriftAnalysis.from_dict(a.to_dict()) == a
    w = fx.dr.windows()[0]
    assert DriftWindow.from_dict(w.to_dict()) == w
    good = a.to_dict()
    for bad in (
        {**good, "kind": "slice_analysis"}, {**good, "schema_version": 999}, {**good, "extra": 1},
        {k: v for k, v in good.items() if k != "summary"}, {**good, "analysis_status": "DONE"},
        {**good, "provenance_fingerprint": "abc"}, {**good, "baseline_run_id": "nope"},
    ):  # fmt: skip
        with pytest.raises((ValidationError, SchemaVersionError)):
            DriftAnalysis.from_dict(bad)
    gw = w.to_dict()
    for bad in ({**gw, "start": "0"}, {**gw, "start_inclusive": 1}, {**gw, "role": "OTHER"}, {**gw, "ordering_field": "row"}, {**gw, "start": 1e9, "end": 1}):  # fmt: skip
        with pytest.raises((ValidationError, SchemaVersionError)):
            DriftWindow.from_dict(bad)
    assert w.id == w.window().window_id(w.ordering_field)


def test_registry_listing_and_lookup(fx: Fx) -> None:
    a = fx.analyze()
    assert a.id in {x.id for x in fx.dr.analyses(baseline_run_id=fx.base)}
    assert a.id in {x.id for x in fx.dr.analyses(analysis_status=a.analysis_status)}
    assert {w.role for w in fx.dr.windows()} == {"REFERENCE", "COMPARISON"}
    assert all(w.ordering_field == "index" for w in fx.dr.windows(ordering="index"))
    assert fx.dr.windows(ordering="feature:none") == []
    assert isinstance(fx.w.store, LocalArtifactStore)
    with pytest.raises(ValidationError, match="artifact store"):
        DriftRegistry(fx.w.registry).document(a.id, "summary")
    assert DriftDataError.__mro__[1] is ExperionyxError
