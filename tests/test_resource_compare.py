"""Comparing resource analyses: Phase 10 statistics on the PRESERVED raw trials, pairing declared not
assumed, multiple-comparison correction across a sweep, and refusal of comparisons that would mislead.
Timing is simulated, so the expected medians, ratios and paired differences are computed by hand from
the stored trials, independently of `experionyx.resources.compare`."""

import statistics
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from experionyx.errors import ValidationError
from experionyx.resources import compare as rc
from experionyx.resources import measure as ms
from experionyx.resources.registry import ResourceRegistry
from resource_helpers import RWorld, Sim, SimLinear, View, resource_world


@pytest.fixture
def rig(tmp_path: Path) -> Iterator[tuple[RWorld, Sim]]:
    # a deterministic wobble so that trials differ (a zero-variance sample has no defined test)
    sim = Sim(duration=lambda i, n: 0.001 * (n / 10) * (1 + 0.07 * ((i * 7) % 5)))
    SimLinear.SIM = sim
    with ms.use_probe(sim.probe()):
        yield resource_world(tmp_path), sim
    SimLinear.SIM = None


def raw(r: RWorld, a: View, key: str = "throughput") -> list[float]:
    ts = [t for t in ResourceRegistry(r.w.registry, r.w.store).document(a.id, "trials")["trials"] if t["phase"] == "MEASURED" and t["status"] == "COMPLETED"]  # fmt: skip
    return [t["throughput"]["samples_per_second"] if key == "throughput" else t["wall_seconds"] for t in ts]  # fmt: skip


def cmp(r: RWorld, ids: list[str], **kw: Any) -> dict[str, Any]:
    return rc.compare_analyses(r.w.registry, r.w.store, ids, resamples=200, permutations=300, **kw)


def test_an_unpaired_throughput_comparison_matches_the_stored_trials(rig: tuple[RWorld, Sim]) -> None:  # fmt: skip
    r, _ = rig
    ref = r.run(r.spec(batch_size=30, repeats=8, warmup_trials=0))
    tre = r.run(r.spec(batch_size=15, repeats=8, warmup_trials=0))
    out = cmp(r, [ref.id, tre.id])
    (row,) = out["comparisons"]
    assert out["pairing"] == "UNPAIRED" and out["reference"] == ref.id and out["metric"] == "throughput" and out["environment_match"] is True  # fmt: skip
    x, y = raw(r, ref), raw(r, tre)
    assert row["median_ratio_vs_reference"] == pytest.approx(statistics.median(y) / statistics.median(x))  # fmt: skip
    c = row["comparison"]
    assert c["pairing"] == "UNPAIRED" and c["reference"]["mean"]["value"] == pytest.approx(statistics.fmean(x)) and c["treatment"]["mean"]["value"] == pytest.approx(statistics.fmean(y))  # fmt: skip
    assert c["interval"]["status"] == "DERIVED" and c["interval"]["lower"] < c["interval"]["estimate"] < c["interval"]["upper"]  # fmt: skip
    assert c["effects"] and {e["name"] for e in c["effects"]} and c["test"]["p_value"] is not None
    assert row["differs_from_reference"] == [
        "batch_size"
    ]  # what actually differed in the definition
    assert any("not independent samples of a hardware population" in x for x in out["caveats"])
    assert any("not a causal claim" in x for x in out["caveats"])
    assert cmp(r, [ref.id, tre.id]) == out  # deterministic given the seed


def test_a_sweep_is_corrected_for_multiple_comparisons(rig: tuple[RWorld, Sim]) -> None:
    r, _ = rig
    units = rc.run_sweep(r.w.registry, r.w.store, r.w.executor, r.investigation, r.spec(repeats=8, warmup_trials=0), [30, 15, 10, 6])  # fmt: skip
    ids = [u["analysis_id"] for u in units]
    raw_p = {row["analysis_id"]: row["comparison"]["test"]["p_value"] for row in cmp(r, ids, correction="NONE")["comparisons"]}  # fmt: skip
    bon = cmp(r, ids, correction="BONFERRONI")
    assert bon["correction"]["n_hypotheses"] == 3 and bon["correction"]["method"] == "BONFERRONI"  # the reference is not a hypothesis  # fmt: skip
    for row in bon["comparisons"]:
        assert row["adjusted_p_value"] == pytest.approx(min(1.0, raw_p[row["analysis_id"]] * 3))
        assert row["significant_after_correction"] == (row["adjusted_p_value"] <= 0.05)
    none = cmp(r, ids, correction="NONE")
    assert all(row["adjusted_p_value"] == pytest.approx(raw_p[row["analysis_id"]]) for row in none["comparisons"])  # fmt: skip
    bh = cmp(r, ids, correction="BENJAMINI_HOCHBERG")
    assert "false discovery rate" in bh["correction"]["controls"]
    assert all(b["adjusted_p_value"] <= o["adjusted_p_value"] + 1e-12 for b, o in zip(bh["comparisons"], bon["comparisons"], strict=True))  # fmt: skip


def test_per_sample_latency_at_batch_size_one_is_paired_by_sample_id(rig: tuple[RWorld, Sim]) -> None:  # fmt: skip
    r, sim = rig
    sim.duration = lambda i, n: 0.001 * (1 + (i % 4) * 0.1)
    a = r.run(r.spec(batch_size=1, n_samples=16, repeats=5, warmup_trials=0))
    sim.duration = lambda i, n: 0.0013 * (1 + (i % 3) * 0.2)
    b = r.run(r.spec(batch_size=1, n_samples=16, repeats=5, warmup_trials=0, seed=1))
    out = cmp(r, [a.id, b.id], metric="per_sample_seconds")
    assert out["pairing"] == "PAIRED" and out["comparisons"][0]["comparison"]["pairing"] == "PAIRED"
    rr = ResourceRegistry(r.w.registry, r.w.store)

    def per_sample_means(x: View) -> dict[str, float]:
        acc: dict[str, list[float]] = {}
        for t in rr.document(x.id, "trials")["trials"]:
            if t["phase"] == "MEASURED":
                for sid, sec in t["per_sample_seconds"]:
                    acc.setdefault(str(sid), []).append(sec)
        return {k: statistics.fmean(v) for k, v in acc.items()}

    ma, mb = per_sample_means(a), per_sample_means(b)
    diff = statistics.fmean(mb[k] - ma[k] for k in ma)
    assert out["comparisons"][0]["comparison"]["interval"]["estimate"] == pytest.approx(diff)
    assert (
        out["comparisons"][0]["comparison"]["reference"]["n"] == 16
    )  # 16 sample IDs, not 80 draws


def test_comparisons_that_would_mislead_are_refused(rig: tuple[RWorld, Sim]) -> None:
    r, _ = rig
    a = r.run(r.spec(batch_size=1, n_samples=12, repeats=5, warmup_trials=0))
    b = r.run(r.spec(batch_size=1, n_samples=8, repeats=5, warmup_trials=0))  # other sample IDs
    c = r.run(
        r.spec(batch_size=6, n_samples=12, repeats=5, warmup_trials=0)
    )  # no per-sample latency
    with pytest.raises(ValidationError, match="identical sample IDs"):
        cmp(r, [a.id, b.id], metric="per_sample_seconds")
    with pytest.raises(ValidationError, match="no per-sample latency"):
        cmp(r, [a.id, c.id], metric="per_sample_seconds")
    with pytest.raises(ValidationError, match="same number of samples"):
        cmp(r, [a.id, b.id], metric="trial_seconds")
    assert (
        cmp(r, [a.id, b.id], metric="throughput")["pairing"] == "UNPAIRED"
    )  # throughput is comparable
    with pytest.raises(ValidationError, match="at least two distinct"):
        cmp(r, [a.id])
    with pytest.raises(ValidationError, match="at least two distinct"):
        cmp(r, [a.id, a.id])
    with pytest.raises(ValidationError, match="metric"):
        cmp(r, [a.id, b.id], metric="speed")


def test_an_analysis_without_completed_trials_cannot_be_compared(rig: tuple[RWorld, Sim]) -> None:
    r, _ = rig
    good = r.run(r.spec(repeats=5, warmup_trials=0))
    timed_out = r.run(r.spec(repeats=5, warmup_trials=0, timeout_seconds=0.001))
    assert timed_out.summary["trials"]["used"] == 0
    with pytest.raises(ValidationError, match="no completed measured trials"):
        cmp(r, [good.id, timed_out.id])


def test_analyses_from_different_environments_are_refused_unless_explicitly_allowed_and_then_labelled(tmp_path: Path) -> None:  # fmt: skip
    sim = Sim(duration=lambda i, n: 0.001 * (1 + 0.07 * ((i * 7) % 5)))
    SimLinear.SIM = sim
    try:
        r = resource_world(tmp_path)
        with ms.use_probe(sim.probe()):
            a = r.run(r.spec(repeats=5, warmup_trials=0))
        with ms.use_probe(
            sim.probe(memory=False)
        ):  # another measurement backend = another environment
            b = r.run(r.spec(repeats=5, warmup_trials=0))
        with pytest.raises(ValidationError, match="different environments"):
            cmp(r, [a.id, b.id])
        out = cmp(r, [a.id, b.id], allow_environment_mismatch=True)
        assert out["environment_match"] is False and any("DIFFERENT environments" in x for x in out["caveats"])  # fmt: skip
        assert out["environments"][a.id] != out["environments"][b.id]
    finally:
        SimLinear.SIM = None


def test_nothing_is_stored_by_a_comparison(rig: tuple[RWorld, Sim]) -> None:
    from experionyx.domain import Run
    from experionyx.resources.entities import ResourceAnalysis

    r, _ = rig
    a, b = r.run(r.spec(repeats=5, warmup_trials=0)), r.run(r.spec(repeats=5, warmup_trials=0, batch_size=15))  # fmt: skip
    runs, analyses = len(r.w.registry.find(Run)), len(r.w.registry.find(ResourceAnalysis))
    cmp(r, [a.id, b.id])
    assert len(r.w.registry.find(Run)) == runs and len(r.w.registry.find(ResourceAnalysis)) == analyses  # fmt: skip
