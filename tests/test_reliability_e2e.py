"""Reliability profiles built from REAL persisted evidence (fault laboratory, Phase 6 discovery,
Phase 7 interaction analysis), plus cross-phase integration checks. Every reported value is checked
against its original source, not against the profile's own code path."""

import json
from pathlib import Path
from typing import Any

import pytest

from eval_helpers import EvalWorld, load_evaluation
from experionyx.domain import Artifact, Claim, Evidence, Run, RunStatus
from experionyx.errors import DuplicateError, ProfileRefusal
from experionyx.evaluation.config import EvaluationConfig
from experionyx.failures.entities import FailureMode
from experionyx.failures.lifecycle import change_status as mode_change_status
from experionyx.failures.taxonomy import FailureStatus
from experionyx.faults.report import load_analysis
from experionyx.interactions.entities import InteractionAnalysis, InteractionEffect
from experionyx.provenance import Provenance as Prov
from experionyx.reliability import builder
from experionyx.reliability.engine import replay_check, run_profile
from experionyx.reliability.entities import ReliabilityProfile, ReliabilityReference
from experionyx.reliability.registry import ReliabilityProfileRegistry
from experionyx.reliability.spec import ProfileSpec
from experionyx.reliability.taxonomy import Dimension, RefKind, Scope
from interaction_helpers import EVAL, runs_of
from reliability_helpers import EvidenceSet, base_ids, full_evidence, launch_on, register_variant

VERDICT_WORDS = (
    "robust",
    "unreliable",
    "safer",
    "better",
    "worse",
    "best",
    "worst",
    "rank",
    "trustworthy",
)


@pytest.fixture(scope="module")
def ev(tmp_path_factory: pytest.TempPathFactory) -> EvidenceSet:
    return full_evidence(tmp_path_factory.mktemp("rel"))


def profile_of(e: EvidenceSet, spec: ProfileSpec | None = None):  # type: ignore[no-untyped-def]
    return run_profile(e.w.registry, e.w.store, e.w.executor, e.investigation, spec or e.spec())


def doc_of(e: EvidenceSet, pid: str) -> dict[str, Any]:
    return ReliabilityProfileRegistry(e.w.registry, e.w.store).document(pid)


# -- 1. the full profile ------------------------------------------------------------------------------------


def test_the_profile_reports_persisted_values_exactly_and_labels_each_dimension(
    ev: EvidenceSet,
) -> None:
    out = profile_of(ev)
    assert out.status is RunStatus.COMPLETED and out.profile_id
    doc = doc_of(ev, out.profile_id)
    st = doc["dimension_status"]
    assert set(st) == {
        d.value for d in Dimension
    }  # every dimension is present with an explicit status
    assert (
        st["BASELINE_PERFORMANCE"] == "OBSERVED"
        and st["FAULT_SENSITIVITY"] == "OBSERVED"
        and st["CALIBRATION"] == "OBSERVED"
    )
    assert (
        st["FAILURE_PREVALENCE"]
        == st["FAILURE_SEVERITY"]
        == st["INTERACTION_SENSITIVITY"]
        == "DERIVED"
    )

    # baseline values equal the stored evaluation, read independently
    w = ev.w
    stored = load_evaluation(w.store, w.registry.get(Run, ev.design.baseline))
    want = {m.metric_id: m.value for m in stored.metrics if m.structured is None}
    got = {
        m["metric_id"]: m["value"]
        for m in doc["dimensions"]["BASELINE_PERFORMANCE"]["observations"][0]["metrics"]
    }
    assert (
        got == want
        and doc["dimensions"]["BASELINE_PERFORMANCE"]["observations"][0]["n_samples"]
        == stored.n_samples
    )
    cal = doc["dimensions"]["CALIBRATION"]["observations"][0]
    assert cal["ece"] == stored.calibration.ece and cal["n_bins"] == stored.calibration.n_bins

    # fault responses equal the persisted fault analysis, one entry per experiment, none ranked
    faults = {o["source"]["id"]: o for o in doc["dimensions"]["FAULT_SENSITIVITY"]["observations"]}
    assert set(faults) == set(ev.fault_experiments)
    for fid, o in faults.items():
        res = load_analysis(w.registry, w.store, fid)
        (pt,) = o["points"]
        assert (
            pt["deterioration"]["mean"] == res.points[0].primary_deterioration.mean
            and pt["effect_classification"] == res.points[0].assessment.classification.value
        )
        assert (
            o["trials"]["completed"] == 4
            and o["fault"]["type"]
            and o["fault"]["version"]
            and o["baseline_value"] == res.baseline_value
        )
    assert {o["fault"]["type"] for o in faults.values()} == {
        "gaussian_noise",
        "feature_dropout",
        "compound",
    }
    assert [
        o["fault"]["components"] for o in faults.values() if o["fault"]["type"] == "compound"
    ] == [["gaussian_noise", "feature_dropout"]]

    # failure modes: lifecycle state comes from the registry, and nothing below CONFIRMED is 'established'
    modes = {o["source"]["id"]: o for o in doc["dimensions"]["FAILURE_PREVALENCE"]["observations"]}
    for mid, o in modes.items():
        assert o["lifecycle_state"] == w.registry.get(FailureMode, mid).status.value and o[
            "established"
        ] is (o["lifecycle_state"] == "CONFIRMED")
        assert (
            0 <= o["prevalence"]["fraction_of_runs"] <= 1 and o["prevalence"]["runs_analyzed"] > 0
        )  # prevalence keeps its denominator

    # the interaction summary equals the Phase 7 record
    (i,) = doc["dimensions"]["INTERACTION_SENSITIVITY"]["observations"]
    (eff,) = w.registry.find(InteractionEffect, analysis_id=ev.interaction_id, measure="accuracy")
    assert (
        i["interaction_contrast"] == eff.record["derived"]["interaction_contrast"]
        and i["lifecycle_state"]
        == w.registry.get(InteractionAnalysis, ev.interaction_id).status.value
    )
    assert i["components"] == ["gaussian_noise", "feature_dropout"]


def test_every_observation_has_a_resolvable_source_and_a_stored_reference(ev: EvidenceSet) -> None:
    out = profile_of(ev)
    doc = doc_of(ev, out.profile_id or "")
    w = ev.w
    kinds = {
        "RUN": Run,
        "FAULT_EXPERIMENT": __import__(
            "experionyx.faults.entities", fromlist=["FaultExperiment"]
        ).FaultExperiment,
        "FAILURE_MODE": FailureMode,
        "INTERACTION": InteractionAnalysis,
    }
    checked = 0
    for name, dim in doc["dimensions"].items():
        for o in dim["observations"]:
            src = o.get("source") if isinstance(o, dict) else None
            if name in ("REPRODUCIBILITY", "UNCERTAINTY"):
                continue  # aggregate views over the sourced observations
            assert (
                (src is not None and src["kind"] in kinds)
                or src == {}
                or name in ("SLICE_SENSITIVITY",)
            ), (name, o.keys())
            if src and src.get("kind") in kinds:
                assert w.registry.exists(kinds[src["kind"]], src["id"])
                checked += 1
    assert checked >= 8
    refs = ReliabilityProfileRegistry(w.registry, w.store).references(out.profile_id or "")
    by = {(r.dimension.value, r.ref_kind.value, r.ref_id) for r in refs}
    for fid in ev.fault_experiments:
        assert ("FAULT_SENSITIVITY", "FAULT_EXPERIMENT", fid) in by
    assert ("INTERACTION_SENSITIVITY", "INTERACTION", ev.interaction_id) in by and (
        "BASELINE_PERFORMANCE",
        "RUN",
        ev.design.baseline,
    ) in by
    assert any(
        r.ref_kind is RefKind.ARTIFACT and r.ref_id.startswith("art_") for r in refs
    )  # baseline artifacts are referenced by ID


def test_scientific_language_separates_observed_derived_interpreted_and_never_gives_a_verdict(
    ev: EvidenceSet,
) -> None:
    doc = doc_of(ev, profile_of(ev).profile_id or "")
    texts = [s["text"] for s in doc["interpreted"]["statements"]]
    assert texts and all(
        s["label"] == "INTERPRETED" and s["basis"] for s in doc["interpreted"]["statements"]
    )
    blob = " ".join(texts).lower()
    for word in VERDICT_WORDS:
        assert word not in blob, word
    assert any(
        "Under fault" in t and "mean deterioration" in t and "[" in t for t in texts
    )  # value + interval, not a judgement
    assert any("interaction contrast" in t and "not a causal claim" in t for t in texts)
    assert any(
        "a grouping of signals, not an established finding" in t for t in texts
    )  # modes below CONFIRMED are flagged
    assert (
        doc["no_score"].startswith("no overall score")
        and "score" not in json.dumps(doc["dimension_status"]).lower()
    )
    assert "robust" in " ".join(doc["interpreted"]["never_claimed"]) and doc["language"][
        "derived"
    ].startswith("values computed by earlier phases")
    assert (
        doc["dimensions"]["FAILURE_SEVERITY"]["note"]
        == "severity is a vector; no single severity score exists"
    )


def test_reproducibility_and_uncertainty_are_summarized_without_a_reliability_percentage(
    ev: EvidenceSet,
) -> None:
    doc = doc_of(ev, profile_of(ev).profile_id or "")
    rep = doc["dimensions"]["REPRODUCIBILITY"]["observations"][0]
    assert (
        rep["replay"]["status"] == "NOT_RUN"
        and rep["interactions"]["total"] == 1
        and rep["failure_modes"]["total"] == len(ev.mode_ids)
    )
    assert rep["failure_modes"]["with_passing_reproduction"] == 0 and any(
        "no passing reproduction check" in u for u in rep["unresolved_issues"]
    )
    assert any("not independently reproduced" in u for u in rep["unresolved_issues"]) and {
        r["distinct_seeds"] for r in rep["fault_experiments"]
    } == {4}
    unc = doc["dimensions"]["UNCERTAINTY"]["observations"][0]
    assert {"baseline", "interaction"} <= {
        i["source"].split()[0] for i in unc["intervals"]
    } and any("descriptive" in c for c in unc["caveats"])
    assert any(
        "control has one run" in c for c in unc["caveats"]
    )  # the Phase 7 limitation is carried, not hidden
    assert "percent" not in json.dumps(rep).lower() and "reliability_score" not in json.dumps(doc)


def test_slice_dimension_reuses_existing_class_breakdown_and_states_what_is_unavailable(
    ev: EvidenceSet,
) -> None:
    doc = doc_of(ev, profile_of(ev).profile_id or "")
    (s,) = doc["dimensions"]["SLICE_SENSITIVITY"]["observations"]
    assert {c["class"] for c in s["classes"]} == {"0", "1"} and all(
        c["support"] > 0 for c in s["classes"]
    )
    assert (
        s["slices"]["status"] == "UNAVAILABLE" and "no slices" in s["slices"]["reason"]
    )  # not assumed uniform
    assert isinstance(s["interaction_class_or_slice_effects"], list) and {
        e["level"] for e in s["interaction_class_or_slice_effects"]
    } == {"CLASS"}


# -- 2. minimal and unavailable evidence ---------------------------------------------------------------------------------


def test_missing_evidence_is_reported_unavailable_never_fabricated(ev: EvidenceSet) -> None:
    out = profile_of(ev, ProfileSpec(Scope.MODEL_DATASET, ev.design.baseline))
    doc = doc_of(ev, out.profile_id or "")
    st = doc["dimension_status"]
    for dim in (
        "FAULT_SENSITIVITY",
        "FAILURE_PREVALENCE",
        "FAILURE_SEVERITY",
        "INTERACTION_SENSITIVITY",
    ):
        assert (
            st[dim] == "UNAVAILABLE"
            and doc["dimensions"][dim]["reason"]
            and doc["dimensions"][dim]["observations"] == []
        )
    assert (
        st["REPRODUCIBILITY"] == "INSUFFICIENT_EVIDENCE"
        and st["BASELINE_PERFORMANCE"] == "OBSERVED"
    )
    assert doc["interpreted"]["statements"] and all(
        "fault" not in s["text"].lower() for s in doc["interpreted"]["statements"]
    )


def test_calibration_and_latency_are_unavailable_when_the_evaluation_has_none(
    tmp_path: Path,
) -> None:
    from eval_helpers import class_data, eval_world

    w = eval_world(
        tmp_path, model={"threshold": 5.0, "proba": False}, data=class_data(40), config=EVAL
    )
    res = w.run()
    spec = ProfileSpec(Scope.MODEL_DATASET, res.run.id)
    out = run_profile(w.registry, w.store, w.executor, w.experiment.investigation_id, spec)
    doc = ReliabilityProfileRegistry(w.registry, w.store).document(out.profile_id or "")
    assert (
        doc["dimension_status"]["CALIBRATION"] == "UNAVAILABLE"
        and doc["dimensions"]["CALIBRATION"]["reason"]
    )
    assert doc["dimensions"]["CALIBRATION"]["observations"] == []
    assert doc["dimension_status"]["UNCERTAINTY"] in ("DERIVED", "INSUFFICIENT_EVIDENCE")


# -- 3. scopes and refusals --------------------------------------------------------------------------------------------------


def codes(e: EvidenceSet, spec: ProfileSpec) -> set[str]:
    before = len(e.w.registry.find(Run))
    with pytest.raises(ProfileRefusal) as exc:
        run_profile(e.w.registry, e.w.store, e.w.executor, e.investigation, spec)
    assert len(e.w.registry.find(Run)) == before  # refused before any run was created
    assert all(i.required and i.found and i.why for i in exc.value.issues)  # type: ignore[attr-defined]
    return {i.code for i in exc.value.issues}  # type: ignore[attr-defined]


def test_scopes_decide_what_must_match_and_incompatible_sources_are_refused(
    ev: EvidenceSet,
) -> None:
    w = ev.w
    model, data = base_ids(w)
    other_split = launch_on(
        w, model, data, evaluation=EvaluationConfig(split=None), name="whole-dataset noise"
    )
    other_cfg = launch_on(
        w,
        model,
        data,
        evaluation=EvaluationConfig(split="test", batch_size=7),
        name="other batch size",
    )
    fx = other_split.fault_experiment.id
    assert (
        profile_of(
            ev,
            ev.spec(
                Scope.MODEL_DATASET, fault_experiments=(fx,), interactions=(), failure_modes=()
            ),
        ).status
        is RunStatus.COMPLETED
    )  # split/config may differ at the loosest scope
    assert codes(
        ev,
        ev.spec(
            Scope.MODEL_DATASET_SPLIT, fault_experiments=(fx,), interactions=(), failure_modes=()
        ),
    ) == {"INCOMPATIBLE_FAULT_EXPERIMENT"}
    assert (
        profile_of(
            ev,
            ev.spec(
                Scope.MODEL_DATASET_SPLIT,
                fault_experiments=(other_cfg.fault_experiment.id,),
                interactions=(),
                failure_modes=(),
            ),
        ).status
        is RunStatus.COMPLETED
    )  # same split, different batch size
    assert codes(
        ev,
        ev.spec(
            Scope.MODEL_DATASET_EVALUATION,
            fault_experiments=(other_cfg.fault_experiment.id,),
            interactions=(),
            failure_modes=(),
        ),
    ) == {"INCOMPATIBLE_FAULT_EXPERIMENT"}


def test_a_different_model_or_dataset_is_never_mixed_into_a_profile(ev: EvidenceSet) -> None:
    w = ev.w
    model, data = base_ids(w)
    m2, d2 = (
        register_variant(w, threshold=4.0, tag="m2"),
        register_variant(w, n=100, noise_every=5, tag="d2"),
    )
    other_model = launch_on(w, m2[0], data, name="other model")
    other_data = launch_on(w, model, d2[1], name="other dataset")
    for fx in (other_model.fault_experiment.id, other_data.fault_experiment.id):
        for scope in Scope:
            assert codes(
                ev, ev.spec(scope, fault_experiments=(fx,), interactions=(), failure_modes=())
            ) == {"INCOMPATIBLE_FAULT_EXPERIMENT"}
    disc = run_discovery_on(w, ev, other_data)
    assert codes(
        ev, ev.spec(Scope.MODEL_DATASET, fault_experiments=(), interactions=(), failure_modes=disc)
    ) == {"INCOMPATIBLE_FAILURE_MODE"}
    both = codes(
        ev,
        ev.spec(
            Scope.MODEL_DATASET,
            fault_experiments=(other_model.fault_experiment.id, other_data.fault_experiment.id),
            interactions=(),
            failure_modes=(),
        ),
    )
    assert both == {"INCOMPATIBLE_FAULT_EXPERIMENT"}  # every issue is listed, not just the first


def run_discovery_on(w: EvalWorld, ev: EvidenceSet, exp: Any) -> tuple[str, ...]:
    from experionyx.failures.config import DiscoveryConfig
    from experionyx.failures.engine import run_discovery

    # Only the modes THIS discovery created: everything already registered (from the original
    # design, and from earlier tests) belongs to other sources and must not be selected.
    before = {m.id for m in w.registry.find(FailureMode)}
    run_discovery(
        w.registry,
        w.store,
        w.executor,
        ev.investigation,
        [],
        [exp.fault_experiment.id],
        DiscoveryConfig(),
    )
    fresh = {m.id for m in w.registry.find(FailureMode)} - before
    assert fresh, "the other-dataset discovery should have produced new modes"
    return tuple(sorted(fresh))


def test_invalid_anchors_are_refused_with_explicit_reasons(ev: EvidenceSet) -> None:
    w = ev.w
    faulted = runs_of(w, ev.design.a)[0]
    assert codes(ev, ProfileSpec(Scope.MODEL_DATASET, faulted)) == {"BASELINE_IS_FAULTED"}
    assert codes(ev, ProfileSpec(Scope.MODEL_DATASET, "run_" + "0" * 32)) == {"BASELINE_MISSING"}
    from interaction_helpers import FR
    from interaction_helpers import launch as fault_launch

    bad = fault_launch(
        w,
        FR.make("missing_values", seed=0, probability=0.9),
        (51,),
        name="nan faults",
        baseline_run_id=ev.design.baseline,
    )
    failed = next(
        t.treatment_run_id
        for t in w.registry.find(
            __import__("experionyx.faults.entities", fromlist=["FaultTrial"]).FaultTrial,
            fault_experiment_id=bad.fault_experiment.id,
        )
        if t.treatment_run_id
    )
    assert w.registry.get(Run, failed).status is RunStatus.FAILED
    assert codes(ev, ProfileSpec(Scope.MODEL_DATASET, failed)) == {"BASELINE_NOT_COMPLETED"}


# -- 4. identity, provenance, idempotence, replay -------------------------------------------------------------------------------


def test_identity_is_deterministic_and_repeating_a_profile_is_idempotent(ev: EvidenceSet) -> None:
    first, second = profile_of(ev), profile_of(ev)
    assert first.profile_id == second.profile_id and first.run_id != second.run_id
    rr = ReliabilityProfileRegistry(ev.w.registry, ev.w.store)
    p = rr.get(first.profile_id or "")
    with pytest.raises(DuplicateError):
        rr.register(p)
    assert len(rr.references(p.id)) == len(
        {(r.dimension, r.ref_kind, r.ref_id) for r in rr.references(p.id)}
    )
    fewer = profile_of(ev, ev.spec(fault_experiments=ev.fault_experiments[:2]))
    assert (
        fewer.profile_id != first.profile_id
    )  # a different source set is a different logical profile
    assert rr.get(fewer.profile_id or "").provenance_fingerprint != p.provenance_fingerprint


def test_provenance_changes_when_an_included_lifecycle_state_or_run_changes(
    ev: EvidenceSet,
) -> None:
    w = ev.w
    spec = ev.spec()
    before = builder.build(w.registry, w.store, spec).provenance_fingerprint
    assert (
        builder.build(w.registry, w.store, spec).provenance_fingerprint == before
    )  # deterministic
    mode = next(
        m
        for m in w.registry.find(FailureMode)
        if m.id in ev.mode_ids and m.status in (FailureStatus.DISCOVERED, FailureStatus.CANDIDATE)
    )
    mode_change_status(
        w.registry, mode.id, FailureStatus.REJECTED, "reviewer", "test: lifecycle change"
    )
    assert (
        builder.build(w.registry, w.store, spec).provenance_fingerprint != before
    )  # the mode's state is part of the evidence
    doc = builder.build(w.registry, w.store, spec).document
    row = next(
        o
        for o in doc["dimensions"]["FAILURE_PREVALENCE"]["observations"]
        if o["source"]["id"] == mode.id
    )
    assert (
        row["lifecycle_state"] == "REJECTED"
    )  # and the profile reflects the registry, not a cached copy
    other_seed = builder.build(
        w.registry, w.store, ProfileSpec(Scope.MODEL_DATASET_EVALUATION, ev.design.baseline)
    ).provenance_fingerprint
    assert other_seed != before


def test_replay_reproduces_the_whole_profile_document(ev: EvidenceSet) -> None:
    stable = ev.spec(
        failure_modes=()
    )  # lifecycle states of modes may change between tests; this spec has none
    out = profile_of(ev, stable)
    rep = replay_check(ev.w.registry, ev.w.store, ev.w.executor, out.profile_id or "")
    assert (
        rep["deterministic"] is True
        and rep["sources_changed"] is False
        and rep["differences"] == []
        and rep["replay_run"] != rep["original_run"]
    )


def test_replay_reports_changed_evidence_instead_of_claiming_nondeterminism(
    ev: EvidenceSet,
) -> None:
    w = ev.w
    out = profile_of(ev)
    mode = next(
        m
        for m in w.registry.find(FailureMode)
        if m.id in ev.mode_ids and m.status in (FailureStatus.DISCOVERED, FailureStatus.CANDIDATE)
    )
    doc_before = doc_of(ev, out.profile_id or "")
    mode_change_status(
        w.registry,
        mode.id,
        FailureStatus.REJECTED,
        "reviewer",
        "test: evidence changes after the profile was built",
    )
    rep = replay_check(w.registry, w.store, w.executor, out.profile_id or "")
    assert (
        rep["deterministic"] is None
        and rep["sources_changed"] is True
        and rep["differences"]
        and "evidence changed" in str(rep["note"])
    )
    assert (
        doc_of(ev, out.profile_id or "") == doc_before
    )  # the registered profile is an immutable snapshot


def test_claim_and_evidence_link_to_the_baseline_run_and_profile_artifact(ev: EvidenceSet) -> None:
    out = profile_of(ev)
    w = ev.w
    claims = [c for c in w.registry.find(Claim) if f"[{out.run_id}]" in c.statement]
    assert len(claims) == 1 and "makes no assessment of overall reliability" in claims[0].statement
    targets = {
        (e.target_kind.value, e.target_id) for e in w.registry.find(Evidence, claim_id=claims[0].id)
    }
    assert ("RUN", ev.design.baseline) in targets and any(k == "ARTIFACT" for k, _ in targets)
    names = {
        a.path
        for a in ReliabilityProfileRegistry(w.registry, w.store).artifacts(out.profile_id or "")
    }
    assert names == {"reliability/profile.json", "reliability/sources.json"}


# -- 5. registry and comparison -------------------------------------------------------------------------------------------------


def test_registry_search_filters_evidence_and_provenance(ev: EvidenceSet) -> None:
    out = profile_of(ev)
    rr = ReliabilityProfileRegistry(ev.w.registry, ev.w.store)
    p = rr.get(out.profile_id or "")
    assert p in rr.search(
        model=p.model_fingerprint,
        dataset=p.dataset_fingerprint,
        scope="MODEL_DATASET_EVALUATION",
        split="test",
        evaluation=p.evaluation_config_hash,
    )
    assert (
        p in rr.search(ref=ev.interaction_id)
        and p in rr.search(ref=ev.fault_experiments[0])
        and p not in rr.search(ref="run_" + "0" * 32)
    )
    assert p in rr.search(
        dimension_status=("FAULT_SENSITIVITY", "OBSERVED")
    ) and p not in rr.search(dimension_status=("FAULT_SENSITIVITY", "UNAVAILABLE"))
    assert p not in rr.search(scope="MODEL_DATASET") and p not in rr.search(
        dataset="sha256:" + "0" * 64
    )
    assert (
        rr.references(p.id, dimension="INTERACTION_SENSITIVITY")
        and rr.provenance(p.id)["provenance_fingerprint"] == p.provenance_fingerprint
    )
    assert {a.path for a in rr.artifacts(p.id)} and p.summary["no_score"]


def test_two_real_models_compare_by_raw_differences_with_no_winner(ev: EvidenceSet) -> None:
    w = ev.w
    m2, dataset = register_variant(w, threshold=3.0, tag="cmp")[0], base_ids(w)[1]
    other = launch_on(
        w, m2, dataset, name="noise on model 2"
    )  # same fault definition as the strong design's A cell
    a = profile_of(
        ev, ev.spec(fault_experiments=(ev.fault_experiments[0],), interactions=(), failure_modes=())
    )
    b_spec = ProfileSpec(
        Scope.MODEL_DATASET_EVALUATION, other.baseline_run_id, (other.fault_experiment.id,)
    )
    b = run_profile(w.registry, w.store, w.executor, ev.investigation, b_spec)
    assert b.status is RunStatus.COMPLETED
    rr = ReliabilityProfileRegistry(w.registry, w.store)
    out = rr.compare(a.profile_id or "", b.profile_id or "")
    assert out["same_model"] is False
    acc = next(m for m in out["baseline"]["metrics"] if m["metric_id"] == "accuracy")
    assert acc["difference"] == pytest.approx(acc["b"] - acc["a"]) and acc["b"] != acc["a"]
    (f,) = out["fault_responses"]["matched"]
    assert (
        f["fault"]["type"] == "gaussian_noise"
        and f["points"][0]["deterioration_mean"]["a"] is not None
        and f["points"][0]["deterioration_mean"]["difference"]
        == pytest.approx(
            f["points"][0]["deterioration_mean"]["b"] - f["points"][0]["deterioration_mean"]["a"]
        )
    )
    assert set(out["dimension_status"]) == {d.value for d in Dimension} and out["note"].startswith(
        "raw differences"
    )
    blob = (
        json.dumps(out)
        .lower()
        .replace("higher_is_better", "")
        .replace("no overall ranking, winner or verdict is computed", "")
    )
    for word in ("winner", "better", "worse", "best", "worst", "rank"):
        assert word not in blob
    third = ReliabilityProfileRegistry(w.registry, w.store)
    assert (
        third.get(a.profile_id or "").id != third.get(b.profile_id or "").id
    )  # both remain independently inspectable


def test_profiles_of_different_datasets_cannot_be_compared(ev: EvidenceSet) -> None:
    w = ev.w
    model, _ = base_ids(w)
    _, d2 = register_variant(w, n=100, noise_every=5, tag="cmp_d")
    other = launch_on(w, model, d2, name="noise on dataset 2")
    b = run_profile(
        w.registry,
        w.store,
        w.executor,
        ev.investigation,
        ProfileSpec(Scope.MODEL_DATASET, other.baseline_run_id),
    )
    a = profile_of(ev)
    with pytest.raises(ProfileRefusal) as exc:
        ReliabilityProfileRegistry(w.registry, w.store).compare(
            a.profile_id or "", b.profile_id or ""
        )
    assert {i.code for i in exc.value.issues} >= {"DATASET_MISMATCH", "SCOPE_MISMATCH"}  # type: ignore[attr-defined]


# -- 6. cross-phase integration verification ---------------------------------------------------------------------------------------


def test_every_registered_artifact_still_matches_its_digest_across_all_phases(
    ev: EvidenceSet,
) -> None:
    profile_of(ev)
    w = ev.w
    checked = 0
    for a in w.registry.find(Artifact):
        w.store.verify(
            w.registry.get(Run, a.run_id), a
        )  # raises if any artifact of any phase changed
        checked += 1
    prefixes = {a.path.split("/")[0] for a in w.registry.find(Artifact)}
    assert (
        {"evaluation", "fault", "interaction", "reliability"} <= prefixes
        and any(p.startswith("failure") for p in prefixes)
        and checked > 50
    )


def test_provenance_records_recompute_and_every_phase_procedure_is_recorded(
    ev: EvidenceSet,
) -> None:
    profile_of(ev)
    w = ev.w
    procs = set()
    for p in w.registry.find(Prov):
        again = Prov.from_dict(p.to_dict())
        assert (
            again.fingerprint == p.fingerprint
        )  # the fingerprint is a pure function of the stored record
        procs.add(p.execution.procedure.rsplit(":", 1)[-1])
    assert {
        "run_evaluation",
        "run_fault_evaluation",
        "run_fault_analysis",
        "run_failure_discovery",
        "run_interaction_analysis",
        "run_reliability_profile",
    } <= procs


def test_every_phase_replays_deterministically_from_the_same_registry(ev: EvidenceSet) -> None:
    w = ev.w
    from experionyx.interactions.lifecycle import replay_check as interaction_replay

    assert (
        interaction_replay(w.registry, w.store, w.executor, ev.interaction_id)["deterministic"]
        is True
    )  # phase 7
    baseline = w.executor.replay(ev.design.baseline)  # phase 4
    assert baseline.status is RunStatus.COMPLETED
    old = load_evaluation(w.store, w.registry.get(Run, ev.design.baseline))
    new = load_evaluation(w.store, baseline.run)
    assert {m.metric_id: m.value for m in old.metrics if m.structured is None} == {
        m.metric_id: m.value for m in new.metrics if m.structured is None
    }
    treat = w.executor.replay(runs_of(w, ev.design.a)[0])  # phase 5
    assert treat.status is RunStatus.COMPLETED
    disc = w.executor.replay(ev.discovery.run_id)  # phase 6
    assert disc.status is RunStatus.COMPLETED and {
        m.id for m in w.registry.find(FailureMode)
    } >= set(ev.mode_ids)
    stable = profile_of(
        ev, ev.spec(failure_modes=())
    )  # a spec without modes: their lifecycle states may change
    assert (
        replay_check(w.registry, w.store, w.executor, stable.profile_id or "")["deterministic"]
        is True
    )  # phase 8
    assert w.registry.find(ReliabilityReference) and w.registry.find(ReliabilityProfile)
