"""Statistical engine end to end: persistence, provenance, reproducibility, the interaction
integration and the CLI, on a REAL design produced by the fault laboratory. Ground truth is
recomputed independently from each run's stored evaluation artifacts (no mocking)."""

import json
import math
import statistics
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from eval_helpers import load_evaluation
from experionyx.cli import main
from experionyx.domain import (
    Claim,
    Evidence,
    EvidenceRelation,
    EvidenceTarget,
    Run,
)
from experionyx.errors import (
    ArtifactIntegrityError,
    DuplicateError,
    ValidationError,
)
from experionyx.interactions.config import InteractionConfig
from experionyx.interactions.entities import InteractionEffect
from experionyx.stats import core
from experionyx.stats import store as st
from experionyx.stats.entities import StatisticalAnalysis
from interaction_helpers import Design, four_cell, world

STRONG = {"sigma": 6.0, "p": 0.5}


@pytest.fixture(scope="module")
def design(tmp_path_factory: pytest.TempPathFactory) -> Design:
    return four_cell(
        world(tmp_path_factory.mktemp("stats")), seeds=(1, 2, 3, 4), order=False, **STRONG
    )


@pytest.fixture(scope="module")
def ian(design: Design) -> str:
    from experionyx.interactions.engine import run_interaction

    out = run_interaction(
        design.w.registry, design.w.store, design.w.executor, design.investigation,
        design.spec(InteractionConfig(bootstrap_resamples=300)),
    )  # fmt: skip
    assert out.analysis_id
    return out.analysis_id


def R(a: Any) -> Any:
    """An analysis or effect payload as untyped JSON (typed dicts would drown the assertions)."""
    return a.result if hasattr(a, "result") else a.record


def accuracy(d: Design, run_id: str) -> float:
    ev = load_evaluation(d.w.store, d.w.registry.get(Run, run_id))
    return next(m.value for m in ev.metrics if m.metric_id == "accuracy" and m.value is not None)


# -- interaction-sourced analysis: real evidence, referenced not copied --------------------------------


def src(ian: str, cell: str = "A", ref: str = "CONTROL") -> dict[str, object]:
    return {"kind": "interaction", "analysis_id": ian, "measure": "accuracy",
            "reference_cell": ref, "treatment_cell": cell}  # fmt: skip


def test_compare_from_interaction_trials_matches_run_evaluations(design: Design, ian: str) -> None:
    reg, store = design.w.registry, design.w.store
    a, new = st.create(reg, store, "COMPARE", src(ian), {"resamples": 300})
    assert (
        new and a.analysis_kind == "COMPARE" and a.analysis_status == "INCONCLUSIVE"
    )  # one control run
    a_runs = list(design.spec().a)
    want = statistics.fmean(accuracy(design, r) for r in a_runs) - accuracy(design, design.baseline)
    res = R(a)
    md = next(e for e in res["effects"] if e["name"] == "mean_difference")
    assert md["value"] == pytest.approx(want, abs=1e-12)
    assert res["treatment"]["n"] == len(a_runs) and res["reference"]["n"] == 1
    assert res["declared_pairing"] == "UNPAIRED" and res["pairing"] == "UNPAIRED"
    assert a.config["pairing"] == "UNPAIRED" and a.config["seed"] == 0
    assert (
        "reference" not in a.sources and a.sources["analysis_id"] == ian
    )  # referenced, not copied
    assert res["test"]["status"] == "INCONCLUSIVE"  # one control run: no test


def test_analysis_is_idempotent_and_reproducible_from_persisted_evidence(
    design: Design, ian: str
) -> None:
    reg, store = design.w.registry, design.w.store
    a, _ = st.create(
        reg, store, "BOOTSTRAP", src(ian, "AB", "A"), {"resamples": 200, "method": "bca"}
    )
    again, new = st.create(
        reg, store, "BOOTSTRAP", src(ian, "AB", "A"), {"resamples": 200, "method": "bca"}
    )
    assert not new and again.id == a.id
    other, new2 = st.create(
        reg, store, "BOOTSTRAP", src(ian, "AB", "A"), {"resamples": 200, "seed": 5}
    )
    assert new2 and other.id != a.id  # a different setting is a different analysis
    ok = st.verify(reg, store, a.id)
    assert ok["reproduced"] and ok["inputs_match"] and ok["result_match"]
    iv = R(a)
    assert iv["method"] == "bca" and iv["seed"] == 0 and iv["resamples"] == 200
    assert (
        iv["estimator"] == "mean"
        and iv["n_samples"] == (4, 4)
        and iv["status"] in ("DERIVED", "UNDEFINED")
    )


def test_verify_detects_a_changed_result_record_and_a_changed_artifact(
    design: Design, ian: str
) -> None:
    reg, store = design.w.registry, design.w.store
    good, _ = st.create(reg, store, "COMPARE", src(ian, "AB", "B"), {"resamples": 100})
    forged = replace(good, result_hash="sha256:" + "0" * 64, created_at=datetime.now(UTC))
    reg2_id = forged.id  # identity ignores the result, so the forged record collides with `good`
    assert reg2_id == good.id
    with pytest.raises(DuplicateError):
        reg.add(forged)
    from experionyx.interactions.entities import InteractionAnalysis

    run_dir = store.run_dir(reg.get(Run, reg.get(InteractionAnalysis, ian).run_id))
    target = Path(run_dir) / "artifacts" / "interaction" / "trials.json"
    original = target.read_text(encoding="utf-8")
    try:
        target.write_text(original.replace("0.", "9.", 1), encoding="utf-8")
        with pytest.raises(ArtifactIntegrityError):
            st.verify(reg, store, good.id)  # the digest-verified read refuses silently altered data
    finally:
        target.write_text(original, encoding="utf-8")
    assert st.verify(reg, store, good.id)["reproduced"]


def test_records_are_immutable_and_survive_reopen(design: Design, ian: str) -> None:
    reg = design.w.registry
    a, _ = st.create(
        reg, design.w.store, "PROPORTION", {"kind": "inline", "successes": 2, "trials": 8}
    )
    assert reg.get(StatisticalAnalysis, a.id) == a
    with pytest.raises(Exception, match="immutable"):
        reg._conn.execute(
            "UPDATE statistical_analyses SET analysis_status = 'X' WHERE id = ?", (a.id,)
        )
    with pytest.raises(Exception, match="cannot be deleted"):
        reg._conn.execute("DELETE FROM statistical_analyses WHERE id = ?", (a.id,))
    assert R(a)["estimate"] == 0.25 and a.sources == {
        "kind": "inline",
        "successes": 2,
        "trials": 8,
    }


def test_invalid_inputs_are_rejected_and_nothing_is_stored(design: Design, ian: str) -> None:
    reg, store = design.w.registry, design.w.store
    n = len(reg.find(StatisticalAnalysis))
    bad: list[tuple[dict[str, object], dict[str, object]]] = [
        ({"kind": "inline", "reference": [1.0, math.nan], "treatment": [1.0, 2.0]}, {}),
        ({"kind": "inline", "reference": [1.0, 2.0], "treatment": [1.0, 2.0]}, {"pairing": "PAIRED"}),
        ({"kind": "inline", "reference": {"a": 1.0}, "treatment": {"b": 1.0}}, {"pairing": "PAIRED"}),
        ({"kind": "inline", "reference": [1.0], "treatment": [2.0]}, {"bogus": 1}),
        (src(ian, "ZZ"), {}),
        ({"kind": "nope"}, {}),
    ]  # fmt: skip
    for sources, cfg in bad:
        with pytest.raises(ValidationError):
            st.create(reg, store, "COMPARE", sources, cfg)
    assert len(reg.find(StatisticalAnalysis)) == n


def test_paired_analysis_keeps_identity_by_key(design: Design) -> None:
    reg, store = design.w.registry, design.w.store
    ref = {"s1": 0.50, "s2": 0.60, "s3": 0.70, "s4": 0.80}
    trt = {"s4": 0.75, "s3": 0.62, "s2": 0.51, "s1": 0.40}  # different order, same keys
    a, _ = st.create(reg, store, "COMPARE", {"kind": "inline", "reference": ref, "treatment": trt},
                     {"pairing": "PAIRED", "resamples": 200})  # fmt: skip
    diffs = [-0.10, -0.09, -0.08, -0.05]
    assert R(a)["test"]["p_value"] == pytest.approx(2 / 16)  # exact sign-flip, all one sign
    assert R(a)["test"]["exact"] is True and R(a)["test"]["name"] == "paired_sign_flip"
    dz = next(e for e in R(a)["effects"] if e["name"] == "paired_standardized_difference_dz")
    assert dz["value"] == pytest.approx(statistics.fmean(diffs) / statistics.stdev(diffs))


# -- multiple comparisons and the claim/evidence link --------------------------------------------------


def test_correction_over_registered_analyses_and_claim_evidence(design: Design, ian: str) -> None:
    reg, store = design.w.registry, design.w.store
    ids = [
        st.create(reg, store, "COMPARE", {"kind": "inline", "reference": [1.0, 2.0, 3.0], "treatment": t}, {"resamples": 100})[0].id
        for t in ([4.0, 5.0, 6.0], [1.0, 2.0, 3.5], [3.0, 4.0, 2.5])
    ]  # fmt: skip
    c, _ = st.create(
        reg,
        store,
        "CORRECTION",
        {"kind": "analyses", "ids": ids},
        {"method": "BENJAMINI_HOCHBERG", "alpha": 0.2},
    )
    r = R(c)
    assert r["n_hypotheses"] == 3 and r["method"] == "BENJAMINI_HOCHBERG" and r["alpha"] == 0.2
    raw = {i: R(reg.get(StatisticalAnalysis, i))["test"]["p_value"] for i in ids}
    want = core.adjust_pvalues(raw, method="BENJAMINI_HOCHBERG", alpha=0.2)
    assert r["adjusted"] == pytest.approx(dict(want.adjusted))
    assert c.config == {"method": "BENJAMINI_HOCHBERG", "alpha": 0.2}
    assert st.verify(reg, store, c.id)["reproduced"]
    with pytest.raises(ValidationError, match="not COMPARE"):
        st.create(reg, store, "CORRECTION", {"kind": "analyses", "ids": [c.id]})
    # a claim can cite the analysis as evidence; the registry checks the target exists
    claim = Claim(
        design.investigation, "the shifts differ from the reference", "tester", datetime.now(UTC)
    )
    reg.add(claim)
    ev = Evidence(
        claim.id,
        EvidenceTarget.STATISTICAL_ANALYSIS,
        c.id,
        EvidenceRelation.SUPPORTS,
        datetime.now(UTC),
        "corrected p-values",
    )
    reg.add(ev)
    assert reg.get(Evidence, ev.id).target_id == c.id
    with pytest.raises(Exception, match=r"does not exist|not found|missing"):
        reg.add(replace(ev, target_id="sta_" + "1" * 32))


# -- interaction integration ---------------------------------------------------------------------------


def test_default_interaction_config_identity_is_unchanged() -> None:
    cfg = InteractionConfig()
    assert "multiplicity_correction" not in cfg.to_dict()  # pre-Phase-10 spec IDs stay valid
    assert InteractionConfig.from_dict(cfg.to_dict()) == cfg
    bon = InteractionConfig(multiplicity_correction="BONFERRONI")
    assert (
        bon.to_dict()["multiplicity_correction"] == "BONFERRONI"
        and bon.config_hash != cfg.config_hash
    )
    assert InteractionConfig.from_dict(bon.to_dict()) == bon
    with pytest.raises(ValidationError):
        InteractionConfig(multiplicity_correction="holm")


@pytest.mark.parametrize("method", ["BONFERRONI", "BENJAMINI_HOCHBERG"])
def test_interaction_reports_the_requested_correction_without_changing_labels(
    design: Design, method: str
) -> None:
    from experionyx.interactions.engine import run_interaction

    w = design.w
    plain = run_interaction(w.registry, w.store, w.executor, design.investigation,
                            design.spec(InteractionConfig(bootstrap_resamples=300)))  # fmt: skip
    cfg = InteractionConfig(bootstrap_resamples=300, multiplicity_correction=method)
    out = run_interaction(w.registry, w.store, w.executor, design.investigation, design.spec(cfg))
    assert out.analysis_id and plain.analysis_id
    assert out.analysis_id != plain.analysis_id  # a different rule set
    eff = w.registry.find(InteractionEffect, analysis_id=out.analysis_id)
    base = {e.measure: e for e in w.registry.find(InteractionEffect, analysis_id=plain.analysis_id)}
    fam = {}
    for e in eff:
        rec = R(e)
        assert (
            rec["interpreted"]["class"] == R(base[e.measure])["interpreted"]["class"]
        )  # unchanged
        if rec["status"] == "COMPUTED":
            m = rec["multiplicity"]
            assert m["adjusted_p"] >= m["raw_bootstrap_p"] and 0.0 <= m["adjusted_p"] <= 1.0
            fam[e.measure] = m["raw_bootstrap_p"]
            assert (
                rec["bootstrap"]["intervals"]["interaction_contrast"]["bootstrap_p"]
                == m["raw_bootstrap_p"]
            )
        else:
            assert "multiplicity" not in rec
    corr = core.adjust_pvalues(fam, method=method, alpha=1 - cfg.confidence)
    listed = design.w.registry.get(
        type(w.registry.find(InteractionEffect, analysis_id=out.analysis_id)[0]), eff[0].id
    )
    assert listed  # persisted
    for e in eff:
        if e.measure in fam:
            assert R(e)["multiplicity"]["adjusted_p"] == pytest.approx(corr.adjusted[e.measure])
    assert len(fam) == corr.n_hypotheses and corr.method == method


def test_interaction_default_reports_no_adjustment_and_raw_p(design: Design, ian: str) -> None:
    (e,) = design.w.registry.find(InteractionEffect, analysis_id=ian, measure="accuracy")
    assert "multiplicity" not in R(e)  # nothing is corrected unless asked for
    p = R(e)["bootstrap"]["intervals"]["interaction_contrast"]["bootstrap_p"]
    assert 0.0 < p <= 1.0  # +1 smoothing: a bootstrap p is never exactly 0


def test_paired_interaction_source_pairs_by_trial_key_and_refuses_a_single_control(
    design: Design,
) -> None:
    from experionyx.interactions.engine import run_interaction
    from experionyx.interactions.taxonomy import Pairing

    w = design.w
    cfg = InteractionConfig(bootstrap_resamples=200, pairing=Pairing.PAIRED)
    out = run_interaction(w.registry, w.store, w.executor, design.investigation, design.spec(cfg))
    assert out.analysis_id
    a, _ = st.create(
        w.registry, w.store, "COMPARE", src(out.analysis_id, "AB", "A"), {"resamples": 100}
    )
    assert a.config["pairing"] == "PAIRED"  # taken from the design, recorded in the analysis
    assert R(a)["test"]["name"] == "paired_sign_flip" and R(a)["reference"]["n"] == 4
    with pytest.raises(ValidationError, match="share no trial key"):
        st.create(w.registry, w.store, "COMPARE", src(out.analysis_id, "A", "CONTROL"), {})
    forced, _ = st.create(
        w.registry, w.store, "COMPARE", src(out.analysis_id, "A", "CONTROL"),
        {"pairing": "UNPAIRED", "resamples": 100},
    )  # fmt: skip
    assert R(forced)["pairing"] == "UNPAIRED" and R(forced)["test"]["status"] == "INCONCLUSIVE"


def test_failure_mode_prevalence_interval_uses_the_recorded_denominator(design: Design) -> None:
    from experionyx.failures.config import DiscoveryConfig
    from experionyx.failures.engine import run_discovery
    from experionyx.failures.entities import FailureCluster, FailureMode

    w = design.w
    d = run_discovery(
        w.registry, w.store, w.executor, design.investigation, [],
        [design.a.fault_experiment.id, design.b.fault_experiment.id, design.ab.fault_experiment.id],
        DiscoveryConfig(),
    )  # fmt: skip
    modes = sorted(w.registry.find(FailureMode), key=lambda m: m.id)
    assert modes and d.run_id
    seen = {}
    for m in modes:
        prev: Any = w.registry.get(FailureCluster, m.cluster_id).metrics["prevalence"]
        a, _ = st.create(
            w.registry, w.store, "PROPORTION", {"kind": "failure_mode", "mode_id": m.id}
        )
        r = R(a)
        assert (r["successes"], r["trials"]) == (prev["runs_with_signal"], prev["runs_analyzed"])
        assert 0.0 <= r["lower"] <= r["estimate"] <= r["upper"] <= 1.0
        assert "independent Bernoulli" in r["source_notes"][0]  # the i.i.d. caveat travels with it
        assert a.sources["mode_id"] == m.id and st.verify(w.registry, w.store, a.id)["reproduced"]
        seen[m.id] = a.id
    assert len(set(seen.values())) == len(
        seen
    )  # distinct modes never collapse, whatever the counts
    with pytest.raises(Exception, match=r"not found|does not exist|no "):
        st.create(
            w.registry,
            w.store,
            "PROPORTION",
            {"kind": "failure_mode", "mode_id": "fmd_" + "0" * 32},
        )


def test_fault_trials_source_agrees_with_the_fault_engine_and_run_evaluations(
    design: Design,
) -> None:
    from experionyx.faults.report import analysis_run_id, read_artifact

    reg, store = design.w.registry, design.w.store
    fx = design.a.fault_experiment.id
    a, _ = st.create(
        reg, store, "COMPARE",
        {"kind": "fault_trials", "fault_experiment_id": fx, "metric": "accuracy", "point_index": 0,
         "reference_field": "baseline", "treatment_field": "faulted"},
        {"resamples": 300, "method": "bca"},
    )  # fmt: skip
    r = R(a)
    assert a.config["pairing"] == "PAIRED" and r["test"]["name"] == "paired_sign_flip"
    md = next(e for e in r["effects"] if e["name"] == "mean_difference")["value"]
    want = statistics.fmean(accuracy(design, x) for x in design.spec().a) - accuracy(
        design, design.baseline
    )
    assert md == pytest.approx(
        want, abs=1e-12
    )  # independent: run evaluations, not the fault engine
    agg: Any = read_artifact(reg, store, analysis_run_id(reg, fx), "fault/aggregate.json")
    det = agg["points"][0][
        "primary_deterioration"
    ]  # Phase 5 engine: deterioration = baseline - faulted
    assert md == pytest.approx(-det["mean"], abs=1e-12)
    dz = next(e for e in r["effects"] if e["name"] == "paired_standardized_difference_dz")["value"]
    assert dz == pytest.approx(-det["effect_size_dz"], abs=1e-9)  # same quantity, computed twice
    assert (
        r["reference"]["n"] == r["treatment"]["n"] == 4
        and st.verify(reg, store, a.id)["reproduced"]
    )
    with pytest.raises(ValidationError, match="no completed trial"):
        st.create(reg, store, "COMPARE", {"kind": "fault_trials", "fault_experiment_id": fx, "metric": "nope",
                                         "point_index": 0, "reference_field": "baseline", "treatment_field": "faulted"})  # fmt: skip


def test_artifact_source_reads_a_digest_verified_mapping(design: Design, ian: str) -> None:
    from experionyx.interactions.entities import InteractionAnalysis

    reg, store = design.w.registry, design.w.store
    run_id = reg.get(InteractionAnalysis, ian).run_id
    src_doc = {"kind": "artifact", "run_id": run_id, "path": "interaction/summary.json",
               "reference": ["trials_per_cell"], "treatment": ["trials_per_cell"]}  # fmt: skip
    a, _ = st.create(reg, store, "COMPARE", src_doc, {"resamples": 100})
    assert R(a)["reference"]["n"] == 4
    assert a.sources["reference"] == ("trials_per_cell",)  # a pointer into the artifact, no values
    with pytest.raises(ValidationError, match="not found"):
        st.create(reg, store, "COMPARE", {**src_doc, "reference": ["no", "such"]})


# -- CLI ---------------------------------------------------------------------------------------------------


def cli(d: Design, capsys: pytest.CaptureFixture[str], *args: str) -> tuple[int, str, str]:
    code = main(["--workspace", str(d.w.workspace), "stats", *args])
    out = capsys.readouterr()
    return code, out.out, out.err


def test_cli_compare_bootstrap_proportion_correct_list_inspect_verify(
    design: Design, ian: str, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    reg = design.w.registry
    reg.close()  # the CLI opens its own connection; this test must stay last (module fixture)
    code, out, _ = cli(
        design,
        capsys,
        "compare",
        "--reference-values",
        "1,2,3",
        "--treatment-values",
        "4,5,6",
        "--resamples",
        "200",
    )
    doc = json.loads(out)
    assert code == 0 and doc["kind"] == "COMPARE" and doc["new"] is True
    assert doc["result"]["test"]["p_value"] == pytest.approx(0.1) and doc["config"]["seed"] == 0
    aid = doc["id"]
    code, out, _ = cli(
        design,
        capsys,
        "compare",
        "--reference-values",
        "1,2,3",
        "--treatment-values",
        "4,5,6",
        "--resamples",
        "200",
    )
    assert json.loads(out)["new"] is False and json.loads(out)["id"] == aid

    code, out, _ = cli(
        design,
        capsys,
        "bootstrap",
        "--reference-values",
        "1,2,6",
        "--method",
        "bca",
        "--resamples",
        "300",
        "--seed",
        "4",
    )
    b = json.loads(out)["result"]
    assert code == 0 and b["method"] == "bca" and b["seed"] == 4 and b["pairing"] == "ONE_SAMPLE"
    assert b["acceleration"] == pytest.approx(
        18 / (6 * 14**1.5), abs=0.2
    )  # jackknife of a resample-free sample

    paired = tmp_path / "paired.json"
    paired.write_text(
        json.dumps({"reference": {"a": 1, "b": 2, "c": 3}, "treatment": {"c": 4, "b": 4, "a": 2}}),
        encoding="utf-8",
    )
    code, out, _ = cli(
        design,
        capsys,
        "compare",
        "--file",
        str(paired),
        "--pairing",
        "PAIRED",
        "--resamples",
        "100",
    )
    assert code == 0 and json.loads(out)["result"]["pairing"] == "PAIRED"
    positional = tmp_path / "pos.json"
    positional.write_text(
        json.dumps({"reference": [1, 2, 3], "treatment": [4, 5, 6]}), encoding="utf-8"
    )
    code, _, err = cli(design, capsys, "compare", "--file", str(positional), "--pairing", "PAIRED")
    assert code == 2 and "keyed" in err  # position is never identity
    code, _, err = cli(
        design, capsys, "compare", "--reference-values", "1,x", "--treatment-values", "2,3"
    )
    assert code == 2 and "comma-separated" in err

    code, out, _ = cli(design, capsys, "proportion", "--successes", "5", "--trials", "10")
    p = json.loads(out)["result"]
    assert code == 0 and (p["lower"], p["upper"]) == pytest.approx((0.2366, 0.7634), abs=5e-4)
    assert cli(design, capsys, "proportion", "--successes", "11", "--trials", "10")[0] == 2
    assert cli(design, capsys, "proportion")[0] == 2

    code, out, _ = cli(
        design,
        capsys,
        "correct",
        "--p",
        "a=0.01",
        "--p",
        "b=0.04",
        "--p",
        "c=0.03",
        "--p",
        "d=0.005",
        "--method",
        "BENJAMINI_HOCHBERG",
    )
    c = json.loads(out)["result"]
    assert code == 0 and c["adjusted"] == pytest.approx(
        {"a": 0.02, "b": 0.04, "c": 0.04, "d": 0.02}
    )
    code, out, _ = cli(design, capsys, "correct", "--analysis", aid)
    assert code == 0 and json.loads(out)["result"]["n_hypotheses"] == 1

    code, out, _ = cli(
        design,
        capsys,
        "compare",
        "--interaction",
        ian,
        "--measure",
        "accuracy",
        "--treatment-cell",
        "AB",
        "--resamples",
        "100",
    )
    inter = json.loads(out)
    assert (
        code == 0 and inter["source"] == "interaction" and inter["config"]["pairing"] == "UNPAIRED"
    )

    code, out, _ = cli(design, capsys, "list", "--kind", "COMPARE")
    assert code == 0 and aid in {r["id"] for r in json.loads(out)["analyses"]}
    assert cli(design, capsys, "list", "--format", "text")[1].count("\n") >= 4
    code, out, _ = cli(design, capsys, "inspect", aid)
    assert code == 0 and json.loads(out)["sources"]["kind"] == "inline"
    code, out, _ = cli(design, capsys, "verify", inter["id"])
    assert code == 0 and json.loads(out)["reproduced"] is True
    assert (
        cli(design, capsys, "inspect", "sta_" + "0" * 32)[0] == 2
    )  # unknown ID: an error, not a crash
