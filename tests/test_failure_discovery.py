"""Failure discovery end to end on REAL fault experiments and evaluations (five scenarios plus
bounds, claims and failure handling). Nothing here is mocked except one deliberately wrong replay."""

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from eval_helpers import EvalWorld
from experionyx.domain import (
    Claim,
    ConfigurationRef,
    Evidence,
    EvidenceTarget,
    Observation,
    Run,
    RunStatus,
)
from experionyx.errors import FailureLimitError, MissingReferenceError, ValidationError
from experionyx.execution import Executor
from experionyx.failures.config import (
    ClusteringConfig,
    DiscoveryConfig,
    EvidenceConfig,
    EvidenceCriteria,
    ExtractionConfig,
    SimilarityConfig,
    SimilarityWeights,
)
from experionyx.failures.engine import PROCEDURE, DiscoveryRunResult, run_discovery
from experionyx.failures.entities import (
    FailureCluster,
    FailureEvidence,
    FailureMode,
    FailureRelationship,
    FailureSignal,
)
from experionyx.failures.lifecycle import change_status, confirm, reproduce
from experionyx.failures.taxonomy import EvidenceKind, FailureStatus, SignalKind
from experionyx.faults.entities import FaultTrial, TrialStatus
from experionyx.faults.report import read_artifact
from experionyx.provenance import Provenance
from failure_helpers import home_of, launch, noise_sweep, noisy_world

EASY = DiscoveryConfig(
    evidence=EvidenceConfig(
        candidate=EvidenceCriteria(
            min_signals=2,
            min_runs=1,
            min_experiments=1,
            min_seeds=1,
            min_mean_effect=0.01,
            min_compactness=0.0,
            min_threshold_stability=0.0,
        ),
        supported=EvidenceCriteria(
            min_signals=3,
            min_runs=3,
            min_experiments=1,
            min_seeds=1,
            min_mean_effect=0.01,
            min_compactness=0.0,
            min_threshold_stability=0.0,
        ),
    )
)


@pytest.fixture
def w(tmp_path: Path) -> EvalWorld:
    return noisy_world(tmp_path)


def discover(
    w: EvalWorld,
    home: str,
    faults: list[str],
    runs: list[str] | None = None,
    cfg: DiscoveryConfig | None = None,
) -> DiscoveryRunResult:
    return run_discovery(w.registry, w.store, w.executor, home, runs or [], faults, cfg)


def modes(w: EvalWorld) -> list[FailureMode]:
    return w.registry.find(FailureMode)


def art(w: EvalWorld, run_id: str, name: str) -> Any:
    return read_artifact(w.registry, w.store, run_id, f"{name}.json")


# -- scenario 1: noise sweep -> discovery -> candidates, with full lineage -----------------------


def test_scenario_1_noise_sweep_discovery_records_lineage_artifacts_claims_and_never_confirms(
    w: EvalWorld,
) -> None:
    res = noise_sweep(w)
    out = discover(w, home_of(w, res), [res.fault_experiment.id])
    reg = w.registry
    assert out.status is RunStatus.COMPLETED
    found = modes(w)
    assert found
    assert not any(m.status is FailureStatus.CONFIRMED for m in found)  # nothing is auto-confirmed
    assert any(m.status is FailureStatus.CANDIDATE for m in found)

    trial_runs = {
        t.treatment_run_id
        for t in reg.find(FaultTrial, fault_experiment_id=res.fault_experiment.id)
        if t.status is TrialStatus.COMPLETED
    }
    signals = reg.find(FailureSignal)
    assert signals
    for s in signals:  # full lineage: every signal points at a real treatment run
        assert s.run_id in trial_runs and reg.get(Run, s.run_id).status is RunStatus.COMPLETED
        assert s.detail["fault"]["fault_experiment_id"] == res.fault_experiment.id
        assert s.detail["fault"]["baseline_run_id"] == res.baseline_run_id
        assert s.detail["fault"]["type"] == "gaussian_noise" and s.detail["fault"]["seed"] in (
            1,
            2,
            3,
        )
        assert s.direction.value == "WORSE" and s.magnitude > 0
    assert {c.id for c in reg.find(FailureCluster)} == {m.cluster_id for m in found} | {
        c.id for c in reg.find(FailureCluster) if len(c.signal_ids) == 1
    }

    names = {
        a.name
        for a in reg.find(
            __import__("experionyx.domain", fromlist=["Artifact"]).Artifact, run_id=out.run_id
        )
    }
    assert {
        "failure-signals",
        "failure-clusters",
        "failure-candidates",
        "failure-analysis",
    } <= names
    analysis = art(w, out.run_id, "failure-analysis")
    assert (
        analysis["counts"]["signals"] == len(signals)
        and analysis["config_hash"] == DiscoveryConfig().config_hash
    )
    assert "cannot_conclude" in analysis and "causal" in analysis["cannot_conclude"]
    assert (
        analysis["limits"]["max_pairwise_comparisons"]
        == DiscoveryConfig().similarity.max_pairwise_comparisons
    )
    cands = art(w, out.run_id, "failure-candidates")["modes"]
    assert {c["id"] for c in cands} == {m.id for m in found}

    # claims are bounded statements with run + artifact evidence, never causal
    claims = [c for c in reg.find(Claim) if "failure signals of kinds" in c.statement.lower()]
    assert claims and all("no causal claim" in c.statement for c in claims)
    for c in claims:
        targets = {e.target_kind for e in reg.find(Evidence, claim_id=c.id)}
        assert {EvidenceTarget.RUN, EvidenceTarget.ARTIFACT} <= targets
    assert {o.name for o in reg.find(Observation, run_id=out.run_id)} >= {
        "failures.signals",
        "failures.clusters",
        "failures.modes",
    }
    (prov,) = reg.find(Provenance, run_id=out.run_id)
    assert prov.execution.procedure == PROCEDURE
    conf = reg.get(
        ConfigurationRef, reg.get(type(w.experiment), out.experiment_id).configuration_id
    )
    assert (
        DiscoveryConfig.from_dict(conf.parameters["failure_discovery"]["config"])
        == DiscoveryConfig()
    )  # config recorded


def test_measurements_have_denominators_and_class_specific_modes_are_class_aware(
    w: EvalWorld,
) -> None:
    res = noise_sweep(w)
    discover(w, home_of(w, res), [res.fault_experiment.id])
    for m in modes(w):
        meas = m.structured["measurements"]
        assert meas["prevalence"]["runs_analyzed"] >= meas["prevalence"]["runs_with_signal"] >= 1
        assert "score" not in meas["severity"] and meas["impact"]["unit"]
        assert m.structured["provenance"]["config_hash"] == DiscoveryConfig().config_hash
        assert set(m.structured["criteria"]) == {"candidate", "supported"}
        for check in m.structured["criteria"]["candidate"]["checks"]:
            assert {"name", "required", "observed", "passed"} <= set(check)
        if len(meas["classes"]) > 0:
            assert len(meas["classes"]) == 1  # a mode never mixes classes ("2" vs "20" stays apart)


# -- scenario 2: cross-experiment / cross-fault reuse ---------------------------------------------


def test_scenario_2_cross_experiment_discovery_and_configurable_cross_fault_clustering(
    w: EvalWorld,
) -> None:
    a = noise_sweep(w)
    b = launch(
        w,
        "uniform_noise",
        {"half_width": 1.0},
        ("half_width", (3.0, 6.0, 12.0)),
        (1, 2, 3),
        baseline_run_id=a.baseline_run_id,
    )
    home = home_of(w, a)
    both = [a.fault_experiment.id, b.fault_experiment.id]
    default_out = discover(w, home, both)
    assert default_out.status is RunStatus.COMPLETED
    analysis = art(w, default_out.run_id, "failure-analysis")
    assert {
        s["fault_experiment_id"] for s in analysis["sources"] if s["kind"] == "fault-experiment"
    } == set(both)
    assert all(
        s["investigation_id"] for s in analysis["sources"]
    )  # source investigations are lineage
    # by default a Gaussian and a uniform noise fault do not merge (fault type is a similarity dimension)
    assert all(len(m.structured["measurements"]["fault_types"]) == 1 for m in modes(w))

    relaxed = replace(
        DiscoveryConfig(),
        similarity=SimilarityConfig(
            weights=SimilarityWeights(
                fault_type=0.0, parameter_proximity=0.0, affected_fraction=0.0, severity=0.0
            ),
            threshold=0.8,
        ),
    )
    discover(w, home, both, cfg=relaxed)
    merged = [
        m
        for m in modes(w)
        if m.structured["provenance"]["config_hash"] == relaxed.config_hash
        and len(m.structured["measurements"]["fault_types"]) == 2
    ]
    assert merged  # cross-fault reuse: one mode built from signals of two different fault types...
    assert all(
        len(m.structured["measurements"]["fault_families"]) >= 2 for m in merged
    )  # ...and both experiments
    assert {m.structured["measurements"]["fault_targets"][0] for m in merged} == {"INPUT_OR_MIXED"}


def test_home_investigation_must_exist(w: EvalWorld) -> None:
    res = noise_sweep(w)
    with pytest.raises(MissingReferenceError):
        run_discovery(
            w.registry, w.store, w.executor, "inv_" + "0" * 32, [], [res.fault_experiment.id]
        )
    # a FAILED discovery run leaves no partial modes behind
    assert not modes(w)


# -- scenario 3: reproduction and explicit confirmation --------------------------------------------


def test_scenario_3_reproduce_then_confirm_with_full_evidence_trail(w: EvalWorld) -> None:
    res = noise_sweep(w)
    home = home_of(w, res)
    discover(w, home, [res.fault_experiment.id], cfg=EASY)
    supported = [m for m in modes(w) if m.status is FailureStatus.SUPPORTED]
    assert supported, "the easy configuration should support at least one mode"
    mode = supported[0]
    with pytest.raises(ValidationError, match="reproduction"):
        confirm(w.registry, mode.id, "reviewer", "premature")
    rep = reproduce(w.registry, w.store, w.executor, mode.id, EASY)
    assert rep.passed and rep.attempted >= 1 and rep.passed_count == rep.attempted
    for r in rep.records:  # original result, reproduction result, tolerance, pass/fail, provenance
        assert {
            "original_magnitude",
            "reproduced_magnitude",
            "tolerance",
            "passed",
            "provenance",
            "replay_run_id",
            "original_run_id",
        } <= set(r)
        assert r["replay_run_id"] != r["original_run_id"]  # a NEW run; the original is untouched
        assert w.registry.get(Run, r["original_run_id"]).status is RunStatus.COMPLETED
        assert r["reproduced_magnitude"] == pytest.approx(r["original_magnitude"], abs=1e-9)
    assert (
        w.registry.get(FailureMode, mode.id).status is FailureStatus.SUPPORTED
    )  # reproduce never confirms
    done = confirm(w.registry, mode.id, "reviewer", "reproduced on replay; reviewed evidence")
    assert done.status is FailureStatus.CONFIRMED
    kinds = [e.evidence_kind for e in w.registry.find(FailureEvidence, failure_mode_id=mode.id)]
    assert (
        kinds.count(EvidenceKind.REPRODUCTION) == rep.attempted + 1
        and EvidenceKind.TRANSITION in kinds
    )
    edges = [
        e
        for e in w.registry.find(FailureRelationship)
        if e.subject_id == mode.id and e.predicate.value == "REPRODUCED_BY"
    ]
    assert len(edges) == rep.attempted
    change_status(w.registry, mode.id, FailureStatus.DEPRECATED, "reviewer", "superseded")
    retained = [
        e
        for e in w.registry.find(FailureEvidence, failure_mode_id=mode.id)
        if e.evidence_kind is EvidenceKind.TRANSITION
    ]
    assert any(
        e.detail["reason"] == "reproduced on replay; reviewed evidence" for e in retained
    )  # evidence survives deprecation


class _WrongReplay:
    """Replays a DIFFERENT run than asked (the strongest one): a reproduction that must fail."""

    def __init__(self, real: Executor, target: str) -> None:
        self.real, self.target = real, target

    def replay(self, run_id: str):  # type: ignore[no-untyped-def]
        return self.real.replay(self.target)


def test_a_reproduction_outside_tolerance_fails_and_blocks_confirmation(w: EvalWorld) -> None:
    res = noise_sweep(w)
    discover(w, home_of(w, res), [res.fault_experiment.id], cfg=EASY)
    mode = next(m for m in modes(w) if m.status is FailureStatus.SUPPORTED)
    weakest = min(
        w.registry.get(FailureCluster, mode.cluster_id).signal_ids,
        key=lambda i: w.registry.get(FailureSignal, i).magnitude,
    )
    with pytest.raises(
        ValidationError, match="config hash"
    ):  # reproduction must use the discovery configuration
        reproduce(
            w.registry,
            w.store,
            w.executor,
            mode.id,
            replace(EASY, max_source_runs=EASY.max_source_runs + 1),
        )
    other_run = w.registry.get(FailureSignal, weakest).run_id
    rep = reproduce(w.registry, w.store, _WrongReplay(w.executor, other_run), mode.id, EASY)  # type: ignore[arg-type]
    assert rep.attempted >= 2
    assert not rep.passed
    assert any(not r["passed"] for r in rep.records)
    with pytest.raises(ValidationError, match="reproduction"):
        confirm(w.registry, mode.id, "reviewer", "should be refused")
    assert w.registry.get(FailureMode, mode.id).status is FailureStatus.SUPPORTED


# -- scenario 4: determinism, idempotence and replay of the discovery itself ------------------------


def test_scenario_4_discovery_is_deterministic_idempotent_and_replayable(w: EvalWorld) -> None:
    res = noise_sweep(w)
    home = home_of(w, res)
    first = discover(w, home, [res.fault_experiment.id])
    snapshot = {m.id: (m.status, m.cluster_id) for m in modes(w)}
    signal_ids = {s.id for s in w.registry.find(FailureSignal)}
    second = discover(w, home, [res.fault_experiment.id])
    assert (
        second.experiment_id == first.experiment_id and second.run_id != first.run_id
    )  # same design, new Run
    assert {m.id: (m.status, m.cluster_id) for m in modes(w)} == snapshot
    assert {s.id for s in w.registry.find(FailureSignal)} == signal_ids
    counts = art(w, second.run_id, "failure-analysis")["counts"]
    assert (
        counts["new_signals"] == 0 and counts["new_modes"] == 0 and counts["modes"] == len(snapshot)
    )
    a, b = art(w, first.run_id, "failure-clusters"), art(w, second.run_id, "failure-clusters")
    assert [c["id"] for c in a["clusters"]] == [c["id"] for c in b["clusters"]]
    assert [c["metrics"] for c in a["clusters"]] == [c["metrics"] for c in b["clusters"]]
    replay = w.executor.replay(first.run_id)  # the discovery is itself a replayable Run
    assert replay.status is RunStatus.COMPLETED
    assert {m.id for m in modes(w)} == set(snapshot)


def test_a_manual_decision_survives_rediscovery(w: EvalWorld) -> None:
    res = noise_sweep(w)
    home = home_of(w, res)
    discover(w, home, [res.fault_experiment.id])
    mode = next(m for m in modes(w) if m.status is FailureStatus.CANDIDATE)
    change_status(
        w.registry, mode.id, FailureStatus.REJECTED, "reviewer", "artifact of the test data"
    )
    discover(w, home, [res.fault_experiment.id])
    assert w.registry.get(FailureMode, mode.id).status is FailureStatus.REJECTED


# -- scenario 5: failed and skipped trials, and explicit bounds -------------------------------------


def test_scenario_5_failed_and_skipped_trials_are_recorded_not_hidden(w: EvalWorld) -> None:
    from experionyx.faults.design import FaultLimits

    bad = launch(
        w,
        "missing_values",
        {"probability": 0.9},
        None,
        (1, 2, 3, 4),
        limits=FaultLimits(max_failed_trials=1),
        name="nan faults",
    )
    good = noise_sweep(w, baseline_run_id=bad.baseline_run_id)
    out = discover(w, home_of(w, good), [bad.fault_experiment.id, good.fault_experiment.id])
    assert out.status is RunStatus.COMPLETED
    analysis = art(w, out.run_id, "failure-analysis")
    reasons = [s["reason"] for s in analysis["skipped"]]
    assert any("FAILED" in r for r in reasons) and any(
        "SKIPPED" in r for r in reasons
    )  # nothing silently dropped
    assert all(
        s.detail["fault"]["type"] == "gaussian_noise" for s in w.registry.find(FailureSignal)
    )  # no signals from failed runs


def test_a_discovery_over_nothing_usable_completes_with_zero_modes(w: EvalWorld) -> None:
    bad = launch(w, "missing_values", {"probability": 0.9}, None, (1, 2), name="only failures")
    out = discover(w, home_of(w, bad), [bad.fault_experiment.id])
    assert out.status is RunStatus.COMPLETED and not modes(w)
    assert art(w, out.run_id, "failure-analysis")["counts"]["signals"] == 0


def test_bounds_fail_explicitly_or_record_what_was_limited(w: EvalWorld) -> None:
    res = noise_sweep(w)
    home = home_of(w, res)
    tiny = DiscoveryConfig(max_source_runs=2)
    out = discover(w, home, [res.fault_experiment.id], cfg=tiny)
    assert out.status is RunStatus.FAILED and not modes(
        w
    )  # too many source runs: nothing is analyzed silently
    from experionyx.failures.sources import collect_signals
    from failure_helpers import NOW

    with pytest.raises(FailureLimitError, match="max_source_runs"):
        collect_signals(w.registry, w.store, home, [], [res.fault_experiment.id], tiny, NOW)
    few = DiscoveryConfig(extraction=ExtractionConfig(max_signals=3))
    with pytest.raises(FailureLimitError, match="max_signals"):
        collect_signals(w.registry, w.store, home, [], [res.fault_experiment.id], few, NOW)

    capped = DiscoveryConfig(
        clustering=ClusteringConfig(max_clusters=2),
        similarity=SimilarityConfig(max_pairwise_comparisons=5),
    )
    out = discover(w, home, [res.fault_experiment.id], cfg=capped)
    assert out.status is RunStatus.COMPLETED
    limits = art(w, out.run_id, "failure-analysis")["limits"]
    assert limits["clusters_dropped_over_limit"]  # explicit truncation record
    assert limits["blocks_compared_by_exact_signature_only"]
    assert len(w.registry.find(FailureCluster)) == 2


def test_sample_ids_are_bounded_and_the_truncation_is_visible(w: EvalWorld) -> None:
    res = noise_sweep(w)
    cfg = DiscoveryConfig(extraction=ExtractionConfig(max_sample_ids_per_signal=1))
    out = discover(w, home_of(w, res), [res.fault_experiment.id], cfg=cfg)
    sigs = w.registry.find(FailureSignal)
    assert all(len(s.sample_ids) <= 1 for s in sigs)
    assert any(s.truncated for s in sigs)  # the true count is kept when the IDs are cut
    assert art(w, out.run_id, "failure-analysis")["limits"][
        "signals_with_truncated_sample_ids"
    ] == sum(s.truncated for s in sigs)


# -- evaluation-only discovery (no fault) -------------------------------------------------------------


def test_discovery_over_a_plain_evaluation_run_finds_its_own_weaknesses(w: EvalWorld) -> None:
    res = noise_sweep(w)
    home = home_of(w, res)
    strict = DiscoveryConfig(
        extraction=ExtractionConfig(
            class_recall_floor=0.999,
            confusion_pair_rate_min=0.01,
            ece_min=0.0,
            high_confidence_error_rate_min=0.0,
        )
    )
    out = discover(w, home, [], runs=[res.baseline_run_id], cfg=strict)
    assert out.status is RunStatus.COMPLETED
    sigs = w.registry.find(FailureSignal)
    assert sigs and all(s.run_id == res.baseline_run_id for s in sigs)
    kinds = {s.signal_kind for s in sigs}
    assert kinds & {
        SignalKind.PER_CLASS_RECALL_LOW,
        SignalKind.CONFUSION_PAIR,
        SignalKind.CALIBRATION_ECE,
        SignalKind.HIGH_CONFIDENCE_ERRORS,
    }
    assert all(
        "fault" not in s.detail for s in sigs
    )  # baseline weaknesses are not attributed to a fault
    with_ids = [s for s in sigs if s.sample_ids]
    assert with_ids and all(i.isdigit() for s in with_ids for i in s.sample_ids)
