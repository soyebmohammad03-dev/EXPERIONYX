"""Failure vocabulary, configuration, entities, similarity, clustering, criteria and persistence.
No ML framework needed: the registry world uses the pure-Python reference adapters."""

import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from eval_helpers import EvalWorld
from experionyx.domain import ExperimentStatus, Investigation
from experionyx.errors import (
    ConcurrentModificationError,
    DuplicateError,
    FailureError,
    MissingReferenceError,
    ValidationError,
)
from experionyx.failures import analysis as an
from experionyx.failures.config import (
    ClusteringAlgorithm,
    ClusteringConfig,
    DiscoveryConfig,
    EvidenceConfig,
    EvidenceCriteria,
    ExtractionConfig,
    SimilarityConfig,
    SimilarityWeights,
)
from experionyx.failures.discovery import analyze_signals, persist, transition
from experionyx.failures.entities import (
    FailureCluster,
    FailureEvidence,
    FailureMode,
    FailureRelationship,
    FailureSignal,
)
from experionyx.failures.extraction import SignalBuilder, extract_from_evaluation, label_key
from experionyx.failures.lifecycle import change_status, confirm, graph
from experionyx.failures.taxonomy import (
    FAILURE_TRANSITIONS,
    EvidenceKind,
    FailureCategory,
    FailureStatus,
    NodeKind,
    Predicate,
    SignalKind,
)
from failure_helpers import CFG, FP_A, FP_B, NOW, noisy_world, rid, sig


@pytest.fixture
def world(tmp_path: Path) -> EvalWorld:
    return noisy_world(tmp_path)


# -- vocabulary ----------------------------------------------------------------------------------


def test_taxonomy_contains_the_documented_categories() -> None:
    assert {c.value for c in FailureCategory} == {
        "INPUT_SENSITIVITY", "DISTRIBUTION_SHIFT", "LABEL_ERROR", "CLASS_SPECIFIC_ERROR", "FEATURE_DEPENDENCY",
        "CALIBRATION_FAILURE", "HIGH_CONFIDENCE_ERROR", "LOW_CONFIDENCE_ERROR", "SYSTEMATIC_MISCLASSIFICATION",
        "REGRESSION_ERROR", "PERFORMANCE_DEGRADATION", "RESOURCE_SENSITIVITY", "UNKNOWN",
    }  # fmt: skip


def test_lifecycle_transitions_forbid_shortcuts_and_terminal_states() -> None:
    assert set(FAILURE_TRANSITIONS) == set(FailureStatus)
    assert FailureStatus.CONFIRMED not in FAILURE_TRANSITIONS[FailureStatus.DISCOVERED]
    assert FailureStatus.CONFIRMED not in FAILURE_TRANSITIONS[FailureStatus.CANDIDATE]
    assert FAILURE_TRANSITIONS[FailureStatus.CONFIRMED] == {FailureStatus.DEPRECATED}
    assert not FAILURE_TRANSITIONS[FailureStatus.REJECTED]
    assert not FAILURE_TRANSITIONS[FailureStatus.DEPRECATED]


# -- configuration -------------------------------------------------------------------------------


def test_config_hash_is_stable_and_sensitive() -> None:
    a, b = DiscoveryConfig(), DiscoveryConfig()
    assert a.config_hash == b.config_hash
    assert DiscoveryConfig.from_dict(a.to_dict()) == a
    tweaked = replace(a, similarity=replace(a.similarity, threshold=0.7))
    assert tweaked.config_hash != a.config_hash
    assert DiscoveryConfig.from_dict(tweaked.to_dict()) == tweaked


@pytest.mark.parametrize(
    "make",
    [
        lambda: SimilarityConfig(threshold=1.5),
        lambda: SimilarityConfig(max_pairwise_comparisons=0),
        lambda: SimilarityWeights(error_type=-1),
        lambda: SimilarityWeights(*([0.0] * 9)),
        lambda: ExtractionConfig(severity_edges=(0.5, 0.1)),
        lambda: ExtractionConfig(class_recall_floor=2.0),
        lambda: ClusteringConfig(max_clusters=0),
        lambda: EvidenceCriteria(min_signals=0),
        lambda: EvidenceConfig(
            candidate=EvidenceCriteria(min_signals=9), supported=EvidenceCriteria(min_signals=2)
        ),
        lambda: DiscoveryConfig(max_source_runs=0),
        lambda: DiscoveryConfig(schema_version=99),
    ],
)
def test_config_rejects_invalid_values(make) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValidationError):
        make()


def test_config_from_dict_is_strict() -> None:
    data = DiscoveryConfig().to_dict()
    with pytest.raises(ValidationError):
        DiscoveryConfig.from_dict({**data, "surprise": 1})
    bad = {**data, "similarity": {**data["similarity"], "threshold": "high"}}  # type: ignore[dict-item]
    with pytest.raises(ValidationError):
        DiscoveryConfig.from_dict(bad)


# -- entities ------------------------------------------------------------------------------------


def test_signal_identity_ignores_time_but_not_configuration() -> None:
    a = sig(1)
    assert replace(a, created_at=NOW.replace(year=2030)).id == a.id
    assert sig(1, cfg_hash="sha256:" + "c" * 64).id != a.id
    assert FailureSignal.from_dict(a.to_dict()) == a
    assert a.to_dict()["kind"] == "failure_signal"  # the payload discriminator is not shadowed


def test_signal_validation_and_truncation_flag() -> None:
    with pytest.raises(ValidationError):
        sig(1, mag=-0.1)
    with pytest.raises(ValidationError):
        sig(1, ids=("1", "2", "3"), sample_count=2)
    s = sig(1, ids=("1", "2"), sample_count=50)
    assert s.truncated and not sig(2).truncated


def test_mode_transition_validation() -> None:
    m = FailureMode(rid("inv", 1), rid("fcl", 1), FailureCategory.UNKNOWN, "t", "d", {}, NOW)
    assert m.status is FailureStatus.DISCOVERED
    with pytest.raises(ValidationError):
        m.with_status(FailureStatus.CONFIRMED)
    assert m.with_status(FailureStatus.CANDIDATE).status is FailureStatus.CANDIDATE


def test_cluster_requires_sorted_unique_signal_ids() -> None:
    ids = (rid("fsg", 2), rid("fsg", 1))
    with pytest.raises(ValidationError):
        FailureCluster(rid("inv", 1), ids, "a", "1", CFG.config_hash, 0.8, {}, NOW)
    with pytest.raises(ValidationError):
        FailureCluster(rid("inv", 1), (), "a", "1", CFG.config_hash, 0.8, {}, NOW)


def test_relationship_only_allows_documented_edges_and_keeps_co_occurrence_distinct() -> None:
    kw = {"investigation_id": rid("inv", 1), "detail": {}, "created_at": NOW}
    ok = FailureRelationship(
        subject_kind=NodeKind.FAILURE_MODE,
        subject_id=rid("fmd", 1),
        predicate=Predicate.CO_OCCURS_WITH,
        object_kind=NodeKind.FAILURE_MODE,
        object_id=rid("fmd", 2),
        **kw,
    )  # type: ignore[arg-type]
    other = FailureRelationship(
        subject_kind=NodeKind.FAILURE_MODE,
        subject_id=rid("fmd", 1),
        predicate=Predicate.INTERACTS_WITH,
        object_kind=NodeKind.FAILURE_MODE,
        object_id=rid("fmd", 2),
        **kw,
    )  # type: ignore[arg-type]
    assert ok.id != other.id
    with pytest.raises(ValidationError):
        FailureRelationship(
            subject_kind=NodeKind.MODEL,
            subject_id="x",
            predicate=Predicate.CO_OCCURS_WITH,
            object_kind=NodeKind.CLASS,
            object_id="y",
            **kw,
        )  # type: ignore[arg-type]


def test_evidence_roundtrip() -> None:
    e = FailureEvidence(rid("fmd", 1), EvidenceKind.OBSERVATION, None, "s", {"a": [1, 2]}, NOW)
    assert FailureEvidence.from_dict(e.to_dict()) == e


# -- extraction on a REAL evaluation -------------------------------------------------------------


def test_class_keys_are_exact() -> None:
    assert (
        label_key(2) != label_key(20)
        and label_key(2) != label_key("2")
        and label_key(2) == label_key(2)
    )


def _evaluation(w: EvalWorld):  # type: ignore[no-untyped-def]
    from eval_helpers import load_evaluation

    res = w.run()
    return res.run, load_evaluation(w.store, res.run)


def test_extraction_from_a_real_evaluation_is_thresholded_deterministic_and_lineaged(
    world: EvalWorld,
) -> None:
    run, ev = _evaluation(world)
    from experionyx.failures.sources import load_errors

    errors = load_errors(world.registry, world.store, run.id)
    strict = ExtractionConfig(
        class_recall_floor=0.999,
        confusion_pair_rate_min=0.01,
        high_confidence_error_rate_min=0.0,
        ece_min=0.0,
    )
    b1 = SignalBuilder(run.id, run.experiment_id, ev, strict, CFG.config_hash, NOW)
    extract_from_evaluation(b1, errors)
    b2 = SignalBuilder(run.id, run.experiment_id, ev, strict, CFG.config_hash, NOW)
    extract_from_evaluation(b2, errors)
    assert [s.id for s in b1.out] == [s.id for s in b2.out]  # deterministic
    kinds = {s.signal_kind for s in b1.out}
    assert SignalKind.CONFUSION_PAIR in kinds
    for s in b1.out:
        assert s.run_id == run.id and s.detail["model_fingerprint"] == ev.context.model_fingerprint
        assert s.magnitude >= 0 and s.extractor_version
    pair = next(s for s in b1.out if s.signal_kind is SignalKind.CONFUSION_PAIR)
    assert pair.sample_ids and pair.sample_count == len(pair.sample_ids)  # IDs preserved
    lenient = ExtractionConfig(
        class_recall_floor=0.0,
        confusion_pair_rate_min=1.0,
        high_confidence_error_rate_min=1.0,
        ece_min=1.0,
        slice_deterioration_min=1.0,
    )
    b3 = SignalBuilder(run.id, run.experiment_id, ev, lenient, CFG.config_hash, NOW)
    extract_from_evaluation(b3, errors)
    assert not [
        s
        for s in b3.out
        if s.signal_kind in (SignalKind.PER_CLASS_RECALL_LOW, SignalKind.CONFUSION_PAIR)
    ]


def test_sample_id_truncation_is_recorded_not_silent(world: EvalWorld) -> None:
    run, ev = _evaluation(world)
    from experionyx.failures.sources import load_errors

    cfg = ExtractionConfig(confusion_pair_rate_min=0.01, max_sample_ids_per_signal=1)
    b = SignalBuilder(run.id, run.experiment_id, ev, cfg, CFG.config_hash, NOW)
    extract_from_evaluation(b, load_errors(world.registry, world.store, run.id))
    pairs = [s for s in b.out if s.signal_kind is SignalKind.CONFUSION_PAIR]
    assert pairs
    assert all(len(s.sample_ids) <= 1 for s in pairs)
    assert any(s.truncated and s.sample_count > 1 for s in pairs)


# -- similarity ----------------------------------------------------------------------------------


def test_class_two_never_matches_class_twenty() -> None:
    s = an.similarity(sig(1, cls="2"), sig(2, cls="20"), CFG.similarity)
    assert s.score == 0.0 and s.blocked == "different class_label"
    assert any(r.dimension == "class_label" and r.score == 0.0 for r in s.reasons)
    assert an.similarity(sig(1, cls="2"), sig(2, cls="2"), CFG.similarity).score > 0.9


def test_predicted_class_and_slice_are_hard_constraints() -> None:
    assert an.similarity(sig(1, pred="0"), sig(2, pred="2"), CFG.similarity).blocked
    assert an.similarity(sig(1, slice_name="a"), sig(2, slice_name="b"), CFG.similarity).blocked
    assert (
        an.similarity(
            sig(1, cls=None, slice_name="a"), sig(2, cls=None, slice_name="a"), CFG.similarity
        ).score
        > 0.9
    )


def test_similarity_is_symmetric_explained_and_a_weighted_mean() -> None:
    a, b = (
        sig(1, mag=0.1, params={"sigma": 1.0}),
        sig(2, mag=0.3, params={"sigma": 1.4}, dataset=FP_B),
    )
    ab, ba = an.similarity(a, b, CFG.similarity), an.similarity(b, a, CFG.similarity)
    assert ab.score == ba.score
    assert ab.score == pytest.approx(
        sum(r.weight * r.score for r in ab.reasons) / sum(r.weight for r in ab.reasons)
    )
    assert {r.dimension for r in ab.reasons} >= {
        "class_label",
        "error_type",
        "fault_type",
        "parameter_proximity",
        "severity",
        "dataset",
    }
    assert all(r.note for r in ab.reasons)


def test_dimensions_that_do_not_apply_are_excluded_not_scored_zero() -> None:
    e = an.similarity(sig(1, fault=None, cls=None), sig(2, fault=None, cls=None), CFG.similarity)
    assert {r.dimension for r in e.reasons}.isdisjoint(
        {"class_label", "fault_type", "parameter_proximity"}
    )
    assert e.score > 0.9


def test_different_fault_types_are_less_similar_but_can_still_cluster_when_configured() -> None:
    a, b = (
        sig(1, fault="gaussian_noise"),
        sig(2, fault="feature_dropout", params={"probability": 0.3}),
    )
    assert an.similarity(a, b, CFG.similarity).score < CFG.similarity.threshold
    relaxed = SimilarityConfig(
        weights=SimilarityWeights(fault_type=0.0, parameter_proximity=0.0, affected_fraction=0.0),
        threshold=0.8,
    )
    assert an.similarity(a, b, relaxed).score >= 0.8  # cross-fault reuse is a configuration choice


def test_blocked_comparison_only_compares_within_class_blocks_and_counts() -> None:
    signals = [sig(i, cls="2") for i in range(1, 4)] + [sig(i, cls="20") for i in range(4, 7)]
    links = an.compare_all(signals, CFG.similarity, 0.5)
    assert links.comparisons == 6  # 3 + 3, never the 9 cross-class pairs
    assert all((i < 3) == (j < 3) for i, j, _ in links.pairs)


def test_pairwise_budget_falls_back_to_exact_signatures_and_says_so() -> None:
    same = [replace(sig(i), signature="same") for i in range(1, 5)]
    other = sig(9, cls="7")
    links = an.compare_all([*same, other], SimilarityConfig(max_pairwise_comparisons=1), 0.5)
    assert links.fallback_blocks and links.comparisons <= 1
    assert {(i, j) for i, j, _ in links.pairs} >= {
        (0, 1),
        (0, 2),
        (2, 3),
    }  # identical signatures still linked


# -- clustering, stability, measurements ----------------------------------------------------------


def _tight(n0: int, cls: str, k: int = 4):  # type: ignore[no-untyped-def]
    return [sig(n0 + i, cls=cls, mag=0.2 + 0.001 * i, seed=i, dataset=FP_A) for i in range(k)]


def test_clusters_are_class_aware_deterministic_and_order_independent() -> None:
    signals = _tight(1, "2") + _tight(10, "20")
    a = analyze_signals(sorted(signals, key=lambda s: s.id), CFG, runs_analyzed=8)
    b = analyze_signals(
        sorted(signals, key=lambda s: s.id, reverse=True)[::-1], CFG, runs_analyzed=8
    )
    got = sorted(sorted(s.id for s in c.members) for c in a.clusters)
    assert got == sorted(sorted(s.id for s in c.members) for c in b.clusters)
    assert len(a.clusters) == 2
    for c in a.clusters:
        assert len({s.detail["class_label"] for s in c.members}) == 1  # 2 and 20 never mix
        assert c.metrics["stability"] == 1.0 and c.metrics["compactness"] > 0.95


def test_signature_grouping_algorithm_is_available_and_has_no_stability() -> None:
    cfg = replace(
        CFG, clustering=ClusteringConfig(algorithm=ClusteringAlgorithm.SIGNATURE_GROUPING)
    )
    same = [replace(sig(i), signature="s") for i in range(1, 4)] + [sig(9)]
    r = analyze_signals(same, cfg, runs_analyzed=4)
    assert sorted(len(c.members) for c in r.clusters) == [1, 3]
    assert all(c.metrics["stability"] is None for c in r.clusters)


def test_threshold_instability_is_reported() -> None:
    chain = [
        sig(i, cls="1", mag=0.05 * i, params={"sigma": 1.0 + 0.6 * i}, frac=0.1 * i)
        for i in range(1, 8)
    ]
    r = analyze_signals(
        chain, replace(CFG, similarity=replace(CFG.similarity, threshold=0.9)), runs_analyzed=7
    )
    stabilities = [float(str(c.metrics["stability"])) for c in r.clusters if len(c.members) > 1]
    assert stabilities
    assert min(stabilities) < 1.0


def test_measurements_keep_prevalence_impact_and_severity_separate_with_denominators() -> None:
    members = _tight(1, "1", 4)
    m = an.cluster_measurements(members, total_runs=10)
    assert m["prevalence"]["fraction_of_runs"] == 0.4 and m["prevalence"]["runs_analyzed"] == 10  # type: ignore[index]
    assert m["impact"]["max_magnitude"] >= m["impact"]["mean_magnitude"]  # type: ignore[index]
    sev = m["severity"]
    assert (
        isinstance(sev, dict) and "score" not in sev and len(sev) >= 5
    )  # a vector, never one number
    assert m["direction_consistency"] == 1.0 and m["seeds"] == [0, 1, 2, 3]


def test_sample_groups_preserve_ids_and_never_mix_datasets() -> None:
    a = sig(1, ids=("3", "4"), dataset=FP_A)
    b = sig(2, ids=("3",), dataset=FP_A)
    c = sig(3, ids=("3",), dataset=FP_B)
    g = an.sample_groups([a, b, c])
    top = {(t["dataset_fingerprint"], t["sample_id"]): t["signals"] for t in g["top"]}  # type: ignore[union-attr]
    assert top[(FP_A, "3")] == 2 and top[(FP_B, "3")] == 1 and top[(FP_A, "4")] == 1
    assert g["distinct_samples"] == 3


# -- evidence evaluation -------------------------------------------------------------------------


def test_criteria_report_required_vs_observed_and_fail_when_unmet() -> None:
    members = _tight(1, "1", 4)
    m = an.cluster_measurements(members, 4)
    m["compactness"], m["stability"] = 0.9, 1.0
    ok = an.evaluate_criteria(
        "candidate", EvidenceCriteria(min_signals=3, min_runs=2), members, m, 100, 0.95, 0
    )
    assert ok.passed and all(c.required is not None for c in ok.checks)
    bad = an.evaluate_criteria(
        "candidate", EvidenceCriteria(min_signals=9), members, m, 100, 0.95, 0
    )
    assert not bad.passed
    failed = next(c for c in bad.checks if c.name == "min_signals")
    assert (failed.required, failed.observed, failed.passed) == (9, 4, False)


def test_interval_requirement_is_seeded_and_needs_more_than_one_signal() -> None:
    members = _tight(1, "1", 5)
    m = an.cluster_measurements(members, 5)
    m["compactness"], m["stability"] = 0.9, 1.0
    crit = EvidenceCriteria(min_signals=2, require_interval_excludes_null=True, null_region=0.02)
    a = an.evaluate_criteria("supported", crit, members, m, 200, 0.95, 7)
    b = an.evaluate_criteria("supported", crit, members, m, 200, 0.95, 7)
    assert a.interval == b.interval and a.interval["seed"] == 7 and a.passed
    single = an.evaluate_criteria(
        "supported", crit, members[:1], an.cluster_measurements(members[:1], 1), 200, 0.95, 7
    )
    assert not single.passed and any(
        c.name == "interval_excludes_null" and not c.passed for c in single.checks
    )


def test_a_cluster_of_mixed_directions_fails_direction_consistency() -> None:
    members = [sig(1), sig(2), sig(3, direction=Direction.BETTER)]
    m = an.cluster_measurements(members, 3)
    m["compactness"], m["stability"] = 1.0, 1.0
    ev = an.evaluate_criteria("candidate", EvidenceCriteria(min_signals=3), members, m, 10, 0.9, 0)
    assert (
        not ev.passed and not next(c for c in ev.checks if c.name == "direction_consistency").passed
    )


from experionyx.failures.taxonomy import Direction  # noqa: E402  (kept next to its only users)

# -- persistence, integrity and lifecycle in a real registry --------------------------------------


def _signals_on(world: EvalWorld, k: int = 4) -> list[FailureSignal]:
    run = world.run().run
    return [
        sig(i, run=run.id, exp=run.experiment_id, cls="1", mag=0.2 + 0.001 * i, seed=i)
        for i in range(1, k + 1)
    ]


def _inv(world: EvalWorld) -> str:
    return world.registry.find(Investigation)[0].id


def test_registry_enforces_references_and_consistency(world: EvalWorld) -> None:
    reg = world.registry
    s = _signals_on(world)[0]
    with pytest.raises(MissingReferenceError):
        reg.add(sig(1))  # run does not exist
    other = replace(world.experiment, name="another experiment", status=ExperimentStatus.DRAFT)
    reg.add(other)
    with pytest.raises(ValidationError):
        reg.add(replace(s, experiment_id=other.id))  # a real experiment, but not the run's own
    reg.add(s)
    with pytest.raises(DuplicateError):
        reg.add(s)
    assert reg.get(FailureSignal, s.id) == s
    cl_bad = FailureCluster(_inv(world), (rid("fsg", 1),), "a", "1", CFG.config_hash, 0.8, {}, NOW)
    with pytest.raises(MissingReferenceError):
        reg.add(cl_bad)


def test_failure_tables_are_immutable_and_undeletable(world: EvalWorld) -> None:
    reg = world.registry
    _persisted(world)  # every failure table now has rows, so the row triggers can fire
    path = world.workspace / "registry.sqlite"
    reg.close()
    with sqlite3.connect(path) as raw:
        for table in (
            "failure_signals",
            "failure_clusters",
            "failure_modes",
            "failure_evidence",
            "failure_relationships",
        ):
            assert raw.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] > 0  # noqa: S608
            with pytest.raises(sqlite3.DatabaseError):
                raw.execute(f"DELETE FROM {table}")  # noqa: S608
        for table in (
            "failure_signals",
            "failure_clusters",
            "failure_evidence",
            "failure_relationships",
        ):
            with pytest.raises(sqlite3.DatabaseError):
                raw.execute(f"UPDATE {table} SET payload = '{{}}'")  # noqa: S608
        with pytest.raises(sqlite3.DatabaseError):
            raw.execute(
                "UPDATE failure_modes SET cluster_id = 'fcl_x'"
            )  # identity columns are guarded


def _persisted(world: EvalWorld, k: int = 4):  # type: ignore[no-untyped-def]
    signals = _signals_on(world, k)
    res = analyze_signals(signals, CFG, runs_analyzed=1)
    return signals, persist(world.registry, _inv(world), signals, res, CFG, NOW)


def test_persist_registers_modes_with_evidence_transitions_and_graph(world: EvalWorld) -> None:
    reg = world.registry
    signals, p = _persisted(world)
    (mode,) = p.modes
    assert mode.status in (
        FailureStatus.DISCOVERED,
        FailureStatus.CANDIDATE,
    )  # never CONFIRMED by machine
    assert p.new_signals == len(signals) and p.new_modes == 1 and p.relationships > 0
    ev = reg.find(FailureEvidence, failure_mode_id=mode.id)
    kinds = {e.evidence_kind for e in ev}
    assert EvidenceKind.SIGNAL in kinds and EvidenceKind.OBSERVATION in kinds
    trans = [e for e in ev if e.evidence_kind is EvidenceKind.TRANSITION]
    assert all(e.detail["actor"] and e.detail["reason"] for e in trans)
    g = graph(reg, mode.id)
    assert {e["predicate"] for e in g["edges"]} >= {
        "EXHIBITS",
        "EXPOSES",
        "AFFECTS",
        "SUPPORTED_BY",
    }  # type: ignore[union-attr]
    assert mode.structured["discovered_by"].startswith("deterministic rules")  # type: ignore[union-attr]


def test_rediscovery_is_idempotent_and_never_resets_a_decision(world: EvalWorld) -> None:
    reg = world.registry
    signals, first = _persisted(world)
    mode = first.modes[0]
    if mode.status is FailureStatus.DISCOVERED:
        mode = transition(reg, mode, FailureStatus.CANDIDATE, "tester", "manual", NOW)
    rejected = change_status(
        reg, mode.id, FailureStatus.REJECTED, "tester", "not a real pattern", NOW
    )
    again = persist(reg, _inv(world), signals, analyze_signals(signals, CFG, 1), CFG, NOW)
    assert again.new_signals == 0 and again.new_modes == 0
    assert reg.get(FailureMode, mode.id).status is FailureStatus.REJECTED == rejected.status
    evidence_after = reg.find(FailureEvidence, failure_mode_id=mode.id)
    assert any(e.detail.get("reason") == "not a real pattern" for e in evidence_after)  # retained


def test_status_updates_are_validated_and_compare_and_swap(world: EvalWorld) -> None:
    reg = world.registry
    _, p = _persisted(world)
    mode = reg.get(FailureMode, p.modes[0].id)
    with pytest.raises(ValidationError):
        reg.update_status(replace(mode, status=FailureStatus.CONFIRMED))  # illegal jump
    with pytest.raises(ValidationError):
        reg.update_status(
            replace(mode.with_status(FailureStatus.REJECTED), title="renamed")
        )  # more than status
    stale = mode.with_status(FailureStatus.REJECTED)
    reg.update_status(stale)
    with pytest.raises((ValidationError, ConcurrentModificationError)):
        reg.update_status(stale)


def test_evidence_and_relationship_references_are_checked(world: EvalWorld) -> None:
    reg = world.registry
    _, p = _persisted(world)
    mode = p.modes[0]
    with pytest.raises(MissingReferenceError):
        reg.add(FailureEvidence(mode.id, EvidenceKind.SIGNAL, rid("fsg", 4242), "s", {}, NOW))
    with pytest.raises(MissingReferenceError):
        reg.add(FailureEvidence(rid("fmd", 4242), EvidenceKind.OBSERVATION, None, "s", {}, NOW))
    with pytest.raises(MissingReferenceError):
        reg.add(
            FailureRelationship(
                mode.investigation_id,
                NodeKind.FAILURE_MODE,
                rid("fmd", 4242),
                Predicate.CO_OCCURS_WITH,
                NodeKind.FAILURE_MODE,
                mode.id,
                {},
                NOW,
            )
        )


def test_transition_needs_actor_and_reason_and_confirmation_needs_reproduction(
    world: EvalWorld,
) -> None:
    reg = world.registry
    _, p = _persisted(world)
    mode = reg.get(FailureMode, p.modes[0].id)
    with pytest.raises(ValidationError):
        change_status(reg, mode.id, FailureStatus.REJECTED, " ", "why", NOW)
    with pytest.raises(ValidationError):
        change_status(reg, mode.id, FailureStatus.REJECTED, "me", "", NOW)
    with pytest.raises(FailureError):
        change_status(reg, mode.id, FailureStatus.CONFIRMED, "me", "shortcut", NOW)
    with pytest.raises(ValidationError, match="SUPPORTED"):
        confirm(reg, mode.id, "me", "wish", NOW)


def test_confirmation_requires_a_passing_reproduction_record(world: EvalWorld) -> None:
    reg = world.registry
    _, p = _persisted(world)
    mode = reg.get(FailureMode, p.modes[0].id)
    if mode.status is FailureStatus.DISCOVERED:
        mode = transition(reg, mode, FailureStatus.CANDIDATE, "t", "r", NOW)
    mode = transition(reg, mode, FailureStatus.SUPPORTED, "t", "r", NOW)
    with pytest.raises(ValidationError, match="reproduction"):
        confirm(reg, mode.id, "me", "no reproduction yet", NOW)
    failing = FailureEvidence(
        mode.id, EvidenceKind.REPRODUCTION, None, "failed", {"summary": True, "passed": False}, NOW
    )
    reg.add(failing)
    with pytest.raises(ValidationError, match="reproduction"):
        confirm(reg, mode.id, "me", "failed reproduction", NOW)
    reg.add(
        FailureEvidence(
            mode.id, EvidenceKind.REPRODUCTION, None, "ok", {"summary": True, "passed": True}, NOW
        )
    )
    done = confirm(reg, mode.id, "me", "reproduced and reviewed", NOW)
    assert done.status is FailureStatus.CONFIRMED
    last = [
        e
        for e in reg.find(FailureEvidence, failure_mode_id=mode.id)
        if e.evidence_kind is EvidenceKind.TRANSITION
    ][-1]
    assert last.detail["actor"] in ("me", "t")
    with pytest.raises(ValidationError):
        change_status(
            reg,
            mode.id,
            FailureStatus.REJECTED,
            "me",
            "a confirmed mode can only be deprecated",
            NOW,
        )
