"""Interaction analysis end to end on REAL fault experiments run through the fault laboratory.
Expected values are recomputed independently from the stored evaluation artifacts of each run.
Nothing is mocked except where a test says so explicitly (adversarial sample alignment)."""

import statistics
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from eval_helpers import EvalWorld, load_evaluation
from experionyx.domain import Claim, Evidence, EvidenceTarget, Observation, Run, RunStatus
from experionyx.errors import DesignRefusal, DuplicateError, ValidationError
from experionyx.failures.config import DiscoveryConfig
from experionyx.failures.engine import run_discovery
from experionyx.failures.entities import FailureRelationship
from experionyx.faults.report import read_artifact
from experionyx.interactions import samples as samples_mod
from experionyx.interactions.config import InteractionConfig, InteractionSpec
from experionyx.interactions.design import validate, validate_report
from experionyx.interactions.engine import build_result, run_interaction
from experionyx.interactions.entities import (
    InteractionAnalysis,
    InteractionEffect,
    InteractionEvidence,
)
from experionyx.interactions.lifecycle import (
    change_status,
    check_reproduction,
    confirm_by_review,
    replay_check,
)
from experionyx.interactions.registry import InteractionRegistry
from experionyx.interactions.taxonomy import (
    EvidenceKind,
    InteractionClass,
    InteractionStatus,
    Normalization,
    Pairing,
)
from interaction_helpers import Design, dropout, four_cell, launch, noise, runs_of, world

CFG = InteractionConfig(bootstrap_resamples=400)
STRONG = dict(sigma=6.0, p=0.5)


@pytest.fixture(scope="module")
def strong(tmp_path_factory: pytest.TempPathFactory) -> Design:
    """One real design with an observable interaction (six seeds, both orders)."""
    return four_cell(world(tmp_path_factory.mktemp("strong")), seeds=(1, 2, 3, 4, 5, 6), **STRONG)


@pytest.fixture(scope="module")
def replicate(strong: Design) -> Design:
    """An independent replicate: same structure, different seeds, its own control run."""
    return four_cell(strong.w, seeds=(11, 12, 13, 14, 15, 16), order=False, **STRONG)


def accuracy(w: EvalWorld, run_id: str) -> float:
    ev = load_evaluation(w.store, w.registry.get(Run, run_id))
    return next(m.value for m in ev.metrics if m.metric_id == "accuracy" and m.value is not None)


def independent_contrast(d: Design, cfg: InteractionConfig = CFG) -> dict[str, float]:
    spec = d.spec(cfg)
    y = {
        cell: statistics.fmean(accuracy(d.w, r) for r in runs)
        for cell, runs in spec.cells().items()
    }
    out = {"y0": y["CONTROL"], "ya": y["A"], "yb": y["B"], "yab": y["AB"]}
    out["contrast"] = y["AB"] - y["A"] - y["B"] + y["CONTROL"]
    if "BA" in y:
        out["order_effect"] = y["AB"] - y["BA"]
    return out


def analyze(d: Design, cfg: InteractionConfig = CFG, **kw):  # type: ignore[no-untyped-def]
    return run_interaction(
        d.w.registry, d.w.store, d.w.executor, d.investigation, d.spec(cfg, **kw)
    )


def effect(w: EvalWorld, analysis_id: str, name: str = "accuracy") -> InteractionEffect:
    (e,) = w.registry.find(InteractionEffect, analysis_id=analysis_id, measure=name)
    return e


# -- 1. the real four-cell design, end to end ------------------------------------------------------------------


def test_the_stored_contrast_equals_an_independent_recomputation_from_the_run_evaluations(
    strong: Design,
) -> None:
    out = analyze(strong)
    assert out.status is RunStatus.COMPLETED and out.analysis_id
    want = independent_contrast(strong)
    rec = effect(strong.w, out.analysis_id).record
    d = rec["derived"]
    assert d["interaction_contrast"] == pytest.approx(want["contrast"], abs=1e-12)
    assert (d["y0"], d["ya"], d["yb"], d["yab"]) == pytest.approx(
        (want["y0"], want["ya"], want["yb"], want["yab"])
    )
    assert d["effect_a"] == pytest.approx(want["ya"] - want["y0"]) and d[
        "combined_effect"
    ] == pytest.approx(want["yab"] - want["y0"])
    assert d["expected_additive_effect"] == pytest.approx(d["effect_a"] + d["effect_b"])
    assert d["interaction_contrast"] == pytest.approx(
        d["combined_effect"] - d["expected_additive_effect"]
    )
    assert (
        rec["interpreted"]["class"] == InteractionClass.OBSERVED_INTERACTION.value
    )  # a real, sizeable, stable contrast
    # raw contrast kept as is; the direction-aware reading is separate and explicit
    i = rec["interpreted"]
    assert i["raw_contrast"] == d["interaction_contrast"] and i[
        "deterioration_contrast"
    ] == pytest.approx(-d["interaction_contrast"])
    assert (
        i["deterioration_relation"] == "SUB_ADDITIVE_DETERIORATION"
        and rec["direction"] == "HIGHER_IS_BETTER"
    )


def test_raw_trials_artifacts_claims_evidence_and_language(strong: Design) -> None:
    w = strong.w
    out = analyze(strong, replace(CFG, bootstrap_seed=1))
    reg = InteractionRegistry(w.registry, w.store)
    a = reg.get(out.analysis_id)
    names = {x.path for x in reg.artifacts(a.id)}
    assert {
        f"interaction/{n}.json"
        for n in (
            "spec",
            "design_validation",
            "trials",
            "effects",
            "bootstrap",
            "per_sample",
            "summary",
        )
    } == names  # no failure_modes without a discovery
    trials = reg.raw_trials(a.id)
    assert {c: len(v) for c, v in trials.items()} == {
        "CONTROL": 1,
        "A": 6,
        "B": 6,
        "AB": 6,
    }  # every raw trial retained
    assert all(
        "accuracy" in t["values"] and t["provenance_fingerprint"]
        for ts in trials.values()
        for t in ts
    )
    assert [t["seed"] for t in trials["A"]] == sorted(t["seed"] for t in trials["A"])
    claims = [c for c in w.registry.find(Claim) if str(a.summary["statement"]) in c.statement]
    assert claims and "not a causal or mechanistic claim" in claims[0].statement
    ev = w.registry.find(Evidence, claim_id=claims[0].id)
    assert {e.target_kind for e in ev} >= {EvidenceTarget.RUN, EvidenceTarget.ARTIFACT} and len(
        [e for e in ev if e.target_kind is EvidenceTarget.RUN]
    ) == 19
    assert {o.name for o in w.registry.find(Observation, run_id=out.run_id)} >= {
        "interaction.new_record",
        "interaction.hypotheses_tested",
        "interaction.accuracy.contrast",
    }
    text = " ".join(str(c.statement) for c in claims).lower()
    for banned in ("causes", "responsible", "vulnerable", "mechanistic interaction"):
        assert banned not in text.replace("not a causal or mechanistic claim", "")
    summary = reg.artifact(a.id, "summary")
    assert set(summary["language"]) == {"measures", "infers", "human_may_conclude", "never_claimed"}
    assert (
        summary["multiplicity"]["adjustment"] == "none"
        and summary["multiplicity"]["hypotheses_tested"] > 3
    )


def test_a_supported_interaction_has_lifecycle_evidence_with_actor_and_reason(
    strong: Design,
) -> None:
    out = analyze(strong)
    reg = InteractionRegistry(strong.w.registry)
    a = reg.get(out.analysis_id)
    assert (
        a.status is InteractionStatus.SUPPORTED
        and a.primary_class is InteractionClass.OBSERVED_INTERACTION
    )
    (t,) = [e for e in reg.evidence(a.id) if e.evidence_kind is EvidenceKind.TRANSITION]
    assert (t.detail["from"], t.detail["to"], t.detail["actor"]) == (
        "DISCOVERED",
        "SUPPORTED",
        "experionyx-interaction",
    )
    assert (
        t.detail["reason"]
        and t.detail["at"]
        and t.detail["automatic"] is True
        and t.detail["rule"]["rule"]
    )


def test_uncertainty_and_multiplicity_are_exposed_not_hidden(strong: Design) -> None:
    out = analyze(strong)
    rec = effect(strong.w, out.analysis_id).record
    b = rec["bootstrap"]
    assert (
        b["status"] == "COMPUTED"
        and b["method"] == "percentile"
        and b["resamples"] == 400
        and b["seed"] == 0
        and b["confidence"] == 0.95
    )
    assert b["n_trials"] == {"CONTROL": 1, "A": 6, "B": 6, "AB": 6}
    assert any("not a significance test" in x for x in b["warnings"]) and any(
        "control has one run" in x for x in b["warnings"]
    )
    iv = b["intervals"]["interaction_contrast"]
    assert iv["lower"] < rec["derived"]["interaction_contrast"] < iv["upper"] and (
        iv["lower"] > 0 or iv["upper"] < 0
    )


# -- 2. idempotence, duplicates, replay, provenance ---------------------------------------------------------


def test_repeating_an_analysis_is_idempotent_and_duplicate_registration_is_refused(
    strong: Design,
) -> None:
    first, second = analyze(strong), analyze(strong)
    assert first.analysis_id == second.analysis_id and first.run_id != second.run_id
    reg = InteractionRegistry(strong.w.registry)
    assert len(reg.search(investigation=strong.investigation)) == len(
        {a.id for a in reg.search(investigation=strong.investigation)}
    )
    a = reg.get(first.analysis_id)
    with pytest.raises(DuplicateError):
        reg.register(a)
    assert (
        strong.w.registry.find(
            InteractionEvidence, analysis_id=a.id, evidence_kind="DESIGN"
        ).__len__()
        == 1
    )


def test_replay_reproduces_specification_contrast_bootstrap_and_per_sample_results(
    strong: Design,
) -> None:
    out = analyze(strong, replace(CFG, bootstrap_seed=3))
    rep = replay_check(strong.w.registry, strong.w.store, strong.w.executor, out.analysis_id)
    assert (
        rep["deterministic"] is True
        and rep["differences"] == []
        and rep["replay_run"] != rep["original_run"]
    )
    again = read_artifact(
        strong.w.registry, strong.w.store, str(rep["replay_run"]), "interaction/spec.json"
    )
    orig = read_artifact(
        strong.w.registry,
        strong.w.store,
        strong.w.registry.get(InteractionAnalysis, out.analysis_id).run_id,
        "interaction/spec.json",
    )
    assert (
        again["provenance_fingerprint"] == orig["provenance_fingerprint"]
        and again["spec_id"] == orig["spec_id"]
    )


def test_provenance_changes_with_every_scientifically_meaningful_input(strong: Design) -> None:
    w, reg = strong.w, strong.w.registry
    base = build_result(reg, w.store, strong.spec(CFG))
    assert (
        build_result(reg, w.store, strong.spec(CFG)).provenance_fingerprint
        == base.provenance_fingerprint
    )  # identical inputs

    def fp(spec: InteractionSpec) -> str:
        return build_result(reg, w.store, spec).provenance_fingerprint

    variants = {
        "bootstrap seed": strong.spec(replace(CFG, bootstrap_seed=9)),
        "bootstrap count": strong.spec(replace(CFG, bootstrap_resamples=401)),
        "confidence": strong.spec(replace(CFG, confidence=0.9)),
        "normalization": strong.spec(replace(CFG, normalization=Normalization.BASELINE_MAGNITUDE)),
        "metric configuration": strong.spec(replace(CFG, primary_metric="f1")),
        "aggregation": strong.spec(replace(CFG, aggregation=CFG.aggregation.__class__.MEDIAN)),
        "one fewer trial seed in A": strong.spec(CFG, a=runs_of(w, strong.a)[:-1]),
        "fault order (A/B swapped, BA as compound)": InteractionSpec(
            control=(strong.baseline,),
            a=runs_of(w, strong.b),
            b=runs_of(w, strong.a),
            ab=runs_of(w, strong.ba),
            config=CFG,
        ),  # type: ignore[arg-type]
        "order analysis on": strong.spec(replace(CFG, order_analysis=True)),
    }
    seen = {base.provenance_fingerprint}
    for name, spec in variants.items():
        f = fp(spec)
        assert f not in seen, f"{name} did not change the provenance fingerprint"
        seen.add(f)


def test_a_different_fault_seed_or_parameter_is_a_different_design_with_different_provenance(
    strong: Design, replicate: Design
) -> None:
    w = strong.w
    a = build_result(w.registry, w.store, strong.spec(CFG))
    other_seeds = replicate.spec(CFG)  # same structure, different seeds and control
    b = build_result(w.registry, w.store, other_seeds)
    assert (
        a.provenance_fingerprint != b.provenance_fingerprint
        and a.design.spec.spec_id != other_seeds.spec_id
    )
    assert (
        a.structural_key == b.structural_key
    )  # ...but the STRUCTURE is the same (this is what 'related' uses)
    weak = four_cell(
        world(Path(w.workspace.parent / "weak")), seeds=(1, 2, 3), sigma=2.0, p=0.6, order=False
    )  # different fault parameters
    c = build_result(weak.w.registry, weak.w.store, weak.spec(CFG))
    assert c.structural_key != a.structural_key and c.fault_a != a.fault_a


# -- 3. order effects --------------------------------------------------------------------------------------


def test_order_effect_is_computed_from_both_orders_and_preserved(strong: Design) -> None:
    cfg = replace(CFG, order_analysis=True)
    out = analyze(strong, cfg)
    rec = effect(strong.w, out.analysis_id).record
    want = independent_contrast(strong, cfg)
    d = rec["derived"]
    assert d["order_effect"] == pytest.approx(want["order_effect"]) and d[
        "order_effect"
    ] == pytest.approx(d["interaction_contrast"] - d["interaction_contrast_ba"])
    assert (
        "order_effect" in rec["bootstrap"]["intervals"]
        and rec["interpreted"]["rule"]["order_effect"] == d["order_effect"]
    )
    trials = InteractionRegistry(strong.w.registry, strong.w.store).raw_trials(out.analysis_id)
    assert len(trials["BA"]) == 6 and all(
        t["fault"]["components"][0]["fault_type"] == "feature_dropout" for t in trials["BA"]
    )  # BA really is B then A
    assert all(t["fault"]["components"][0]["fault_type"] == "gaussian_noise" for t in trials["AB"])
    no_order = effect(strong.w, analyze(strong).analysis_id).record["derived"]
    assert (
        "order_effect" not in no_order and no_order["order_note"] == "order analysis not requested"
    )  # not invented when not requested


# -- 4. pairing ---------------------------------------------------------------------------------------------


def test_paired_analysis_pairs_by_seed_key_and_is_refused_when_seeds_differ(strong: Design) -> None:
    out = analyze(strong, replace(CFG, pairing=Pairing.PAIRED))
    rec = effect(strong.w, out.analysis_id).record
    assert rec["bootstrap"]["pairing"] == "PAIRED" and rec["pairing"] == "PAIRED"
    assert (
        sorted(rec["observed"]["A"]["trials"])
        == sorted(rec["observed"]["AB"]["trials"])
        == ["1", "2", "3", "4", "5", "6"]
    )  # keyed by seed, not position
    mismatched = strong.spec(
        replace(CFG, pairing=Pairing.PAIRED), a=runs_of(strong.w, strong.a)[:-1]
    )
    with pytest.raises(DesignRefusal) as exc:
        validate(strong.w.registry, strong.w.store, mismatched)
    assert any(i.code == "PAIRING_KEYS" for i in exc.value.issues)  # type: ignore[attr-defined]
    unknown = analyze(strong, replace(CFG, pairing=Pairing.UNKNOWN))
    assert (
        effect(strong.w, unknown.analysis_id)
        .record["pairing"]
        .startswith("UNPAIRED (declared UNKNOWN)")
    )


# -- 5. per-sample and class/slice analysis ---------------------------------------------------------------------


def test_per_sample_contrasts_are_aligned_and_consistent_with_the_metric_contrast(
    strong: Design,
) -> None:
    out = analyze(strong)
    ps = InteractionRegistry(strong.w.registry, strong.w.store).artifact(
        out.analysis_id, "per_sample"
    )
    assert (
        ps["status"] == "COMPUTED" and ps["alignment"]["verified"] is True and ps["n_samples"] == 40
    )
    err = ps["summary"]["error"]
    acc = independent_contrast(strong)["contrast"]
    assert err["mean_contrast"] == pytest.approx(
        -acc, abs=1e-12
    )  # accuracy = 1 - mean(error), so the mean per-sample error contrast is -I_accuracy
    r = ps["records"][0]
    f = r["fields"]["error"]
    assert (
        f["interaction_contrast"] == pytest.approx(f["yab"] - f["ya"] - f["yb"] + f["y0"])
        and r["sample_id"] == 40
        and r["class"] in ("0", "1")
        and f["status"] == "VALID"
    )
    assert set(ps["by_class"]) == {"0", "1"} and {"true_class_probability", "confidence"} <= set(
        ps["fields"]
    )
    ids = [r["sample_id"] for r in ps["records"]]
    assert ids == sorted(set(ids))  # unique, stable sample IDs


def test_class_level_and_metric_level_effects_are_reported_separately(strong: Design) -> None:
    out = analyze(strong)
    levels = {
        (e.level.value, e.measure)
        for e in strong.w.registry.find(InteractionEffect, analysis_id=out.analysis_id)
    }
    assert (
        ("METRIC", "accuracy") in levels
        and ("CLASS", "class:0:recall") in levels
        and ("CLASS", "class:1:recall") in levels
    )
    for e in strong.w.registry.find(InteractionEffect, analysis_id=out.analysis_id, level="CLASS"):
        assert e.record["derived"]["y0"] is not None and e.record["interpreted"]["class"] in {
            c.value for c in InteractionClass
        }


@pytest.fixture
def rows(monkeypatch: pytest.MonkeyPatch, strong: Design):  # type: ignore[no-untyped-def]
    """Adversarial control of what the per-sample loader reads (the files themselves are digest-protected)."""
    real = samples_mod._rows
    state = {"mutate": None}

    def fake(reg, store, run_id):  # type: ignore[no-untyped-def]
        data = list(real(reg, store, run_id))
        m = state["mutate"]
        return iter(m(run_id, data) if m else data)

    monkeypatch.setattr(samples_mod, "_rows", fake)
    return state


def per_sample_of(d: Design) -> dict:  # type: ignore[type-arg]
    return build_result(d.w.registry, d.w.store, d.spec(CFG)).samples  # type: ignore[return-value]


def test_row_order_never_matters_because_samples_are_keyed_by_id_not_position(
    strong: Design, rows: dict[str, Any]
) -> None:
    base = per_sample_of(strong)
    rows["mutate"] = lambda run_id, data: (
        data[::-1] if run_id == runs_of(strong.w, strong.ab)[0] else data
    )
    shuffled = per_sample_of(strong)
    assert shuffled["status"] == "COMPUTED" and shuffled["records"] == base["records"]


@pytest.mark.parametrize("attack", ["shifted_ids", "duplicate_id", "swapped_truth", "dropped_rows"])
def test_misaligned_samples_are_refused_never_approximated(
    strong: Design, rows: dict[str, Any], attack: str
) -> None:  # type: ignore[no-untyped-def]
    victim = runs_of(strong.w, strong.a)[0]

    def mutate(run_id: str, data: list[dict]) -> list[dict]:  # type: ignore[type-arg]
        if run_id != victim:
            return data
        if attack == "shifted_ids":
            return [{**r, "index": r["index"] + 1} for r in data]
        if attack == "duplicate_id":
            return [*data, data[0]]
        if attack == "dropped_rows":
            return data[:-3]
        return [{**r, "true": 1 - r["true"]} if i == 4 else r for i, r in enumerate(data)]

    rows["mutate"] = mutate
    ps = per_sample_of(strong)
    assert (
        ps["status"] == "REFUSED"
        and ps["reason"]
        and ps["alignment"]["verified"] is False
        and "records" not in ps
    )
    assert {
        "shifted_ids": "different ground-truth",
        "duplicate_id": "more than once",
        "swapped_truth": "different ground-truth",
        "dropped_rows": "different sample set",
    }[attack] in ps["reason"]


def test_per_sample_is_refused_explicitly_beyond_the_size_bound(strong: Design) -> None:
    ps = build_result(
        strong.w.registry, strong.w.store, strong.spec(replace(CFG, max_samples=10))
    ).samples
    assert ps is not None and ps["status"] == "REFUSED" and "max_samples=10" in ps["reason"]  # type: ignore[operator]
    assert (
        build_result(
            strong.w.registry, strong.w.store, strong.spec(replace(CFG, per_sample=False))
        ).samples
        is None
    )


# -- 6. design refusals ------------------------------------------------------------------------------------------


def codes(d: Design, spec: InteractionSpec) -> set[str]:
    with pytest.raises(DesignRefusal) as exc:
        validate(d.w.registry, d.w.store, spec)
    return {i.code for i in exc.value.issues}  # type: ignore[attr-defined]


def test_structurally_invalid_specs_are_refused_before_any_run_is_created(strong: Design) -> None:
    before = len(strong.w.registry.find(Run))
    with pytest.raises(ValidationError):
        InteractionSpec(
            control=(),
            a=runs_of(strong.w, strong.a),
            b=runs_of(strong.w, strong.b),
            ab=runs_of(strong.w, strong.ab),
        )  # missing control
    for cell in ("a", "b", "ab"):
        with pytest.raises(ValidationError):
            strong.spec(CFG, **{cell: ()})
    with pytest.raises(ValidationError):
        strong.spec(replace(CFG, order_analysis=True), ba=())  # order analysis without BA
    assert len(strong.w.registry.find(Run)) == before


def test_incompatible_cells_are_refused_with_every_issue_listed(
    strong: Design, replicate: Design
) -> None:
    w = strong.w
    a, b, ab, ba = (runs_of(w, x) for x in (strong.a, strong.b, strong.ab, strong.ba))  # type: ignore[arg-type]
    assert codes(strong, strong.spec(CFG, control=(a[0],), a=a[1:])) == {"CONTROL_HAS_FAULT"}
    assert codes(strong, strong.spec(CFG, a=(strong.baseline,) if False else a, ab=ba)) == {
        "COMPOUND_MISMATCH"
    }  # BA runs offered as AB
    assert "TREATMENT_WITHOUT_FAULT" in codes(strong, strong.spec(CFG, a=(replicate.baseline,)))
    assert "RUN_MISSING" in codes(strong, strong.spec(CFG, control=("run_" + "0" * 32,)))
    same = launch(w, noise(6.0), (21, 22, 23), name="A again", baseline_run_id=strong.baseline)
    assert "SAME_FAULT" in codes(strong, strong.spec(CFG, b=runs_of(w, same)))
    other = launch(
        w, dropout(0.1), (1, 2, 3), name="B other params", baseline_run_id=strong.baseline
    )
    assert "COMPOUND_MISMATCH" in codes(
        strong, strong.spec(CFG, b=runs_of(w, other))
    )  # AB was built with p=0.5
    mixed = strong.spec(CFG, a=(*a[:2], *runs_of(w, other)[:1]))
    assert "CELL_MIXES_FAULTS" in codes(strong, mixed)
    assert len(ab) == 6 and len(b) == 6


def test_a_different_dataset_model_split_or_evaluation_config_is_refused(
    strong: Design, tmp_path: Path
) -> None:
    import json
    from dataclasses import replace as dc_replace

    from eval_helpers import NOW, class_data
    from experionyx.adapters.capabilities import DeviceKind
    from experionyx.adapters.records import RegisteredDataset, RegisteredModel
    from experionyx.evaluation.config import EvaluationConfig
    from experionyx.faults.design import FaultDesign
    from experionyx.faults.lab import run_fault_experiment
    from interaction_helpers import FR
    from pure_adapters import ListDatasetAdapter, ThresholdClassifierAdapter

    w = strong.w
    (w.workspace / "m" / "data2.json").write_text(
        json.dumps(class_data(80, noise_every=5)), encoding="utf-8"
    )
    (w.workspace / "m" / "model2.json").write_text(
        json.dumps({"threshold": 4.0, "proba": True}), encoding="utf-8"
    )
    d2 = ListDatasetAdapter.load(w.workspace / "m" / "data2.json", version="1", options={})
    m2 = ThresholdClassifierAdapter.load(
        w.workspace / "m" / "model2.json", version="1", device=DeviceKind.CPU, options={}
    )
    rec_d, rec_m = (
        RegisteredDataset("data2", d2.metadata(), NOW, "m/data2.json"),
        RegisteredModel("model2", m2.metadata(), NOW, "m/model2.json"),
    )
    w.registry.add(rec_d)
    w.registry.add(rec_m)
    a_runs = runs_of(w, strong.a)

    def launched(model_id: str, dataset_id: str, evaluation: EvaluationConfig, seeds=(1, 2, 3)):  # type: ignore[no-untyped-def]
        spec = noise(6.0)
        return run_fault_experiment(
            w.registry,
            w.store,
            w.executor,
            model_id=model_id,
            dataset_id=dataset_id,
            base_spec=spec,
            fault_registry=FR,
            design=FaultDesign(spec.to_dict(), evaluation, seeds=seeds),
            name="other",
            source_root=w.workspace.parent,
        )

    orig_m, orig_d = (
        w.registry.find(RegisteredModel)[0].id,
        w.registry.find(RegisteredDataset)[0].id,
    )
    other_dataset = launched(orig_m, rec_d.id, EvaluationConfig(split="test"))
    other_model = launched(rec_m.id, orig_d, EvaluationConfig(split="test"))
    other_config = launched(orig_m, orig_d, EvaluationConfig(split="test", batch_size=7))
    other_split = launched(orig_m, orig_d, EvaluationConfig(split=None))
    for res, expected in (
        (other_dataset, "DATASET_MISMATCH"),
        (other_model, "MODEL_MISMATCH"),
        (other_config, "EVALUATION_CONFIG_MISMATCH"),
        (other_split, "SPLIT_MISMATCH"),
    ):
        got = codes(strong, strong.spec(CFG, a=runs_of(w, res)))
        assert expected in got, (expected, got)
    assert a_runs and dc_replace(CFG) == CFG


def test_failed_runs_are_excluded_and_listed_and_too_few_trials_gives_no_interval(
    strong: Design,
) -> None:
    w = strong.w
    from experionyx.faults.design import (
        FaultLimits,  # noqa: F401  (documenting: a failing fault below)
    )

    bad = launch(
        w,
        __import__("interaction_helpers").FR.make("missing_values", seed=0, probability=0.9),
        (31, 32),
        name="nan faults",
        baseline_run_id=strong.baseline,
    )
    failed = tuple(
        t.treatment_run_id
        for t in w.registry.find(
            __import__("experionyx.faults.entities", fromlist=["FaultTrial"]).FaultTrial,
            fault_experiment_id=bad.fault_experiment.id,
        )
        if t.treatment_run_id
    )
    assert failed and all(w.registry.get(Run, r).status is RunStatus.FAILED for r in failed)
    d = validate(w.registry, w.store, strong.spec(CFG, a=(*runs_of(w, strong.a), *failed)))
    assert {e["run_id"] for e in d.excluded} == set(failed) and all(
        "FAILED" in str(e["reason"]) for e in d.excluded
    )
    assert len(d.trials["A"]) == 6 and any(
        "excluded" in x for x in d.warnings
    )  # mixed failed + successful: the failed ones are listed, not hidden
    assert "CELL_EMPTY" in codes(strong, strong.spec(CFG, a=failed))
    report = validate_report(w.registry, w.store, strong.spec(CFG, a=failed))
    assert (
        report["valid"] is False
        and report["issues"][0]["required"]
        and report["issues"][0]["found"]
        and report["issues"][0]["why"]
    )


def test_one_or_two_trials_are_inconclusive_with_no_interval_and_no_fake_precision(
    tmp_path: Path,
) -> None:
    d = four_cell(world(tmp_path), seeds=(1, 2), order=False, **STRONG)
    out = analyze(d)
    rec = effect(d.w, out.analysis_id).record
    assert (
        out.status is RunStatus.COMPLETED
        and rec["status"] == "INSUFFICIENT_DATA"
        and rec["bootstrap"]["status"] == "INSUFFICIENT_DATA"
    )
    assert (
        rec["bootstrap"]["intervals"] == {}
        and rec["interpreted"]["class"] == "INCONCLUSIVE"
        and "min_trials" in rec["undefined_reason"]
    )
    assert (
        rec["derived"]["interaction_contrast"] is not None
    )  # the point value is still reported, clearly without an interval
    a = d.w.registry.get(InteractionAnalysis, out.analysis_id)
    assert (
        a.status is InteractionStatus.DISCOVERED
        and a.primary_class is InteractionClass.INCONCLUSIVE
    )
    one = four_cell(world(tmp_path / "one"), seeds=(1,), order=False, **STRONG)
    assert effect(one.w, analyze(one).analysis_id).record["bootstrap"]["n_trials"]["A"] == 1


def test_analyze_refuses_an_invalid_design_before_creating_any_record(strong: Design) -> None:
    before = (len(strong.w.registry.find(Run)), len(strong.w.registry.find(InteractionAnalysis)))
    with pytest.raises(DesignRefusal):
        analyze(
            strong,
            CFG,
            control=(runs_of(strong.w, strong.a)[0],),
            a=runs_of(strong.w, strong.a)[1:],
        )
    assert (
        len(strong.w.registry.find(Run)),
        len(strong.w.registry.find(InteractionAnalysis)),
    ) == before


# -- 7. failure-mode interaction ---------------------------------------------------------------------------------------


def test_failure_modes_are_compared_only_when_one_discovery_covers_every_cell(
    strong: Design,
) -> None:
    w = strong.w
    faults = [
        strong.a.fault_experiment.id,
        strong.b.fault_experiment.id,
        strong.ab.fault_experiment.id,
    ]
    partial = run_discovery(
        w.registry, w.store, w.executor, strong.investigation, [], faults[:2], DiscoveryConfig()
    )
    res = build_result(w.registry, w.store, strong.spec(CFG, discovery_run_id=partial.run_id))
    assert (
        res.modes["status"] == "NOT_AVAILABLE"
        and "did not cover" in res.modes["reason"]
        and res.modes["uncovered_runs"]
    )  # type: ignore[index]
    assert (
        build_result(w.registry, w.store, strong.spec(CFG)).modes is None
    )  # not requested: not applicable
    full = run_discovery(
        w.registry, w.store, w.executor, strong.investigation, [], faults, DiscoveryConfig()
    )
    out = analyze(strong, CFG, discovery_run_id=full.run_id)
    fm = InteractionRegistry(w.registry, w.store).artifact(out.analysis_id, "failure_modes")
    assert fm["status"] == "COMPUTED" and fm["modes"] and fm["discovery_run_id"] == full.run_id
    for row in fm["modes"]:
        assert set(row["prevalence"]) == {"A", "B", "AB"} and set(row["states"]) <= {
            "NEWLY_OBSERVED",
            "OBSERVED_UNDER_COMPOUND_TREATMENT",
            "PERSISTING",
            "INCREASED_PREVALENCE",
            "REDUCED_PREVALENCE",
            "INSUFFICIENT_EVIDENCE",
        }
        assert row["evidence"] and all(0 <= p["fraction"] <= 1 for p in row["prevalence"].values())
        if row["mode_status"] == "DISCOVERED":
            assert (
                "INSUFFICIENT_EVIDENCE" in row["states"]
            )  # a mode below CANDIDATE is never treated as established
    edges = {
        (r.predicate.value, r.object_kind.value)
        for r in w.registry.find(FailureRelationship)
        if r.object_id == out.analysis_id or r.subject_id == out.analysis_id
    }
    assert ("OBSERVED_WITH", "INTERACTION") in edges and ("SUPPORTED_BY", "RUN") in edges
    assert not any(
        p in ("CAUSES", "CAUSED_BY") for p, _ in edges
    )  # the graph has no causal predicate


# -- 8. lifecycle, reproduction, review, registry, matching ------------------------------------------------------------------


def test_reproduction_by_independent_replicate_then_explicit_human_confirmation(
    strong: Design, replicate: Design
) -> None:
    w = strong.w
    cfg = replace(CFG, bootstrap_seed=21)
    orig = analyze(strong, cfg)
    rep = run_interaction(
        w.registry, w.store, w.executor, replicate.investigation, replicate.spec(cfg)
    )
    assert (
        w.registry.get(InteractionAnalysis, rep.analysis_id).primary_class
        is InteractionClass.OBSERVED_INTERACTION
    )
    with pytest.raises(ValidationError, match="REPRODUCIBLE"):
        confirm_by_review(
            w.registry, orig.analysis_id, "reviewer", "premature"
        )  # SUPPORTED cannot be confirmed
    check = check_reproduction(w.registry, orig.analysis_id, rep.analysis_id, rel_tol=1.0)
    assert check.passed and {c["name"] for c in check.checks} >= {
        "same_structure",
        "independent_treatment_runs",
        "same_sign",
        "magnitude_within_tolerance",
    }
    a = w.registry.get(InteractionAnalysis, orig.analysis_id)
    assert a.status is InteractionStatus.REPRODUCIBLE
    with pytest.raises(ValidationError, match="named person"):
        confirm_by_review(w.registry, a.id, "experionyx-interaction", "self-confirmation")
    done = confirm_by_review(
        w.registry, a.id, "reviewer", "replicate agrees; reviewed trial-level evidence"
    )
    assert done.status is InteractionStatus.CONFIRMED_BY_REVIEW
    trail = [
        (e.detail["from"], e.detail["to"], e.detail["actor"])
        for e in w.registry.find(InteractionEvidence, analysis_id=a.id, evidence_kind="TRANSITION")
    ]
    assert sorted(trail) == sorted(
        [
            ("DISCOVERED", "SUPPORTED", "experionyx-interaction"),
            ("SUPPORTED", "REPRODUCIBLE", "experionyx-interaction"),
            ("REPRODUCIBLE", "CONFIRMED_BY_REVIEW", "reviewer"),
        ]
    )
    change_status(
        w.registry, a.id, InteractionStatus.DEPRECATED, "reviewer", "superseded by a larger design"
    )
    kept = [
        e
        for e in w.registry.find(InteractionEvidence, analysis_id=a.id)
        if e.evidence_kind is EvidenceKind.REPRODUCTION
    ]
    assert (
        kept
        and kept[0].detail["passed"] is True
        and kept[0].detail["replicate_id"] == rep.analysis_id
    )  # evidence survives deprecation
    with pytest.raises(ValidationError):
        change_status(
            w.registry, a.id, InteractionStatus.REJECTED, "reviewer", "terminal states do not move"
        )


def test_reproduction_fails_for_dependent_runs_or_disagreeing_replicates_and_is_recorded(
    strong: Design, replicate: Design
) -> None:
    w = strong.w
    cfg = replace(CFG, bootstrap_seed=22)
    orig = analyze(strong, cfg)
    spare = analyze(strong, replace(CFG, bootstrap_seed=23)).analysis_id
    change_status(
        w.registry, spare, InteractionStatus.REJECTED, "reviewer", "test: no longer SUPPORTED"
    )
    with pytest.raises(ValidationError, match="SUPPORTED"):
        check_reproduction(w.registry, spare, orig.analysis_id)
    self_check = check_reproduction(w.registry, orig.analysis_id, orig.analysis_id)
    assert not self_check.passed and any(
        c["name"] == "independent_treatment_runs" and not c["passed"] for c in self_check.checks
    )
    assert (
        w.registry.get(InteractionAnalysis, orig.analysis_id).status is InteractionStatus.SUPPORTED
    )  # failure does not move it
    tight = check_reproduction(
        w.registry,
        orig.analysis_id,
        replicate_analysis(replicate, cfg, w),
        abs_tol=0.0,
        rel_tol=0.0,
    )
    assert not tight.passed and any(
        c["name"] == "magnitude_within_tolerance" and not c["passed"] for c in tight.checks
    )
    evidence = w.registry.find(
        InteractionEvidence, analysis_id=orig.analysis_id, evidence_kind="REPRODUCTION"
    )
    assert len(evidence) == 2 and all(e.detail["passed"] is False for e in evidence)
    with pytest.raises(Exception, match="own checks"):
        change_status(
            w.registry, orig.analysis_id, InteractionStatus.CONFIRMED_BY_REVIEW, "me", "shortcut"
        )


def replicate_analysis(replicate: Design, cfg: InteractionConfig, w: EvalWorld) -> str:
    out = run_interaction(
        w.registry, w.store, w.executor, replicate.investigation, replicate.spec(cfg)
    )
    assert out.analysis_id
    return out.analysis_id


def test_registry_search_filters_and_retrieval(strong: Design) -> None:
    w = strong.w
    out = analyze(strong, replace(CFG, bootstrap_seed=41))
    reg = InteractionRegistry(w.registry, w.store)
    a = reg.get(out.analysis_id)
    assert (
        a in reg.search(metric="accuracy")
        and a in reg.search(fault="gaussian_noise")
        and a in reg.search(fault=a.fault_b)
    )
    assert a in reg.search(
        status="SUPPORTED",
        primary_class="OBSERVED_INTERACTION",
        model=a.model_fingerprint,
        dataset=a.dataset_fingerprint,
    )
    assert a in reg.search(
        experiment=w.registry.get(Run, strong.baseline).experiment_id
    ) and a not in reg.search(fault="salt_and_pepper")
    assert a not in reg.search(status="REJECTED") and a not in reg.search(metric="not_a_metric")
    assert reg.provenance(a.id)[
        "provenance_fingerprint"
    ] == a.provenance_fingerprint and reg.effects(a.id, level="METRIC")
    assert {e.evidence_kind for e in reg.evidence(a.id)} >= {
        EvidenceKind.DESIGN,
        EvidenceKind.OBSERVATION,
        EvidenceKind.TRANSITION,
    }


def test_related_interactions_are_pointers_with_visible_differences_never_merges(
    strong: Design, replicate: Design
) -> None:
    w = strong.w
    a = analyze(strong, replace(CFG, bootstrap_seed=51)).analysis_id
    same = run_interaction(
        w.registry,
        w.store,
        w.executor,
        replicate.investigation,
        replicate.spec(replace(CFG, bootstrap_seed=51)),
    ).analysis_id
    weak = four_cell(
        world(w.workspace.parent / "weak2"), seeds=(1, 2, 3), sigma=2.0, p=0.5, order=False
    )  # same faults, different sigma
    weak_id = run_interaction(
        weak.w.registry, weak.w.store, weak.w.executor, weak.investigation, weak.spec(CFG)
    ).analysis_id
    reg = InteractionRegistry(w.registry, w.store)
    rel = {r.analysis_id: r for r in reg.related(a)}
    assert rel[same].relation == "SAME_STRUCTURE" and reg.get(same).id != a
    assert weak_id not in rel  # a different registry: nothing links across registries
    other = InteractionRegistry(weak.w.registry, weak.w.store)
    assert other.related(weak_id) == []  # nothing similar in that registry
    # same registry, different parameters: register a weaker design next to the strong one
    w2 = four_cell(w, seeds=(41, 42, 43), sigma=2.0, p=0.5, order=False)
    diff_id = run_interaction(
        w.registry, w.store, w.executor, w2.investigation, w2.spec(CFG)
    ).analysis_id
    r2 = {r.analysis_id: r for r in reg.related(a)}
    assert r2[diff_id].relation == "POTENTIALLY_RELATED" and any(
        "sigma" in x and "6.0" in x and "2.0" in x for x in r2[diff_id].differences
    )
