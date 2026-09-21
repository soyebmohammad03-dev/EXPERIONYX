"""End-to-end resource measurements on a controlled world. Timing here is SIMULATED (`Sim`), so every
latency, throughput, CPU, memory, timeout and statistics value is checked against numbers computed by
hand from the script. Real-clock behaviour (real threads, the system probe) is checked separately and
never against a timing value."""

import json
import sqlite3
import sys
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import experionyx.capture as capture_module
from experionyx.artifacts import LocalArtifactStore
from experionyx.domain import Observation, Run, to_jsonable
from experionyx.errors import (  # fmt: skip
    ArtifactIntegrityError,
    CorruptRecordError,
    DuplicateError,
    ExperionyxError,
    ValidationError,
)
from experionyx.resources import compare as rc
from experionyx.resources import engine as re_
from experionyx.resources import measure as ms
from experionyx.resources.entities import ResourceAnalysis, ResourceTrial
from experionyx.resources.registry import ResourceRegistry
from resource_helpers import (  # fmt: skip
    MB,
    N,
    RWorld,
    SafeSimLinear,
    Sim,
    SimLinear,
    SleepLinear,
    View,
    resource_world,
)


@pytest.fixture
def rig(tmp_path: Path) -> Iterator[tuple[RWorld, Sim]]:
    sim = Sim()
    SimLinear.SIM = sim
    with ms.use_probe(sim.probe()):
        yield resource_world(tmp_path), sim
    SimLinear.SIM = None


def docs(r: RWorld, a: View) -> dict[str, Any]:
    rr = ResourceRegistry(r.w.registry, r.w.store)
    return {n: rr.document(a.id, n) for n in re_.DOCUMENTS}


def n_runs(r: RWorld) -> int:
    return len(r.w.registry.find(Run))


# -- exact measurements -----------------------------------------------------------------------------------------


def test_fixture_measurements_are_exactly_the_scripted_ones(rig: tuple[RWorld, Sim]) -> None:
    r, _ = rig
    a = r.run(r.spec(batch_size=30, repeats=5, warmup_trials=1))
    d = docs(r, a)
    ts = d["trials"]["trials"]
    assert [(t["phase"], t["trial_index"]) for t in ts] == [("WARMUP", 0)] + [("MEASURED", i) for i in range(5)]  # fmt: skip
    for t in ts:
        assert t["status"] == "COMPLETED" and t["completed_samples"] == N and t["failed_samples"] == 0  # fmt: skip
        assert t["wall_seconds"] == pytest.approx(0.04) and t["batch_seconds"] == pytest.approx([0.01] * 4)  # fmt: skip
        assert t["batch_sizes"] == [30] * 4 and t["cpu_seconds"] == pytest.approx(0.032)
        assert t["cpu_utilization"] == pytest.approx(0.8) and t["adapter_inference_seconds"] == pytest.approx(4e-6)  # fmt: skip
        assert t["throughput"]["samples_per_second"] == pytest.approx(3000.0) and t["throughput"]["batches_per_second"] == pytest.approx(100.0)  # fmt: skip
    st = d["statistics"]
    assert st["trials"] == {"planned": 5, "missing": [], "measured": 5, "used": 5, "failed": 0, "timed_out": 0, "warmup": 1}  # fmt: skip
    assert st["trial_seconds"]["mean"] == pytest.approx(0.04) and st["trial_seconds"]["std"] == pytest.approx(0.0, abs=1e-12)  # fmt: skip
    assert st["batch_seconds"]["n"] == 20 and st["batch_seconds"]["percentiles"]["p99"] == pytest.approx(0.01)  # fmt: skip
    assert a.analysis_status == "COMPLETE" and a.summary["evidence_status"] == "MEASURED"


def test_warmup_is_recorded_but_excluded_from_steady_state(rig: tuple[RWorld, Sim]) -> None:
    r, sim = rig
    sim.duration = lambda i, n: 0.05 if i < 4 else 0.01  # the first pass (the warmup) is slow
    a = r.run(r.spec(batch_size=30, repeats=5, warmup_trials=1))
    d = docs(r, a)
    st = d["statistics"]
    assert st["trial_seconds"]["n"] == 5 and st["trial_seconds"]["mean"] == pytest.approx(0.04)
    assert st["warmup_seconds"]["n"] == 1 and st["warmup_seconds"]["mean"] == pytest.approx(0.2)
    s = a.summary
    assert s["warmup"] == {"trials": 1, "seconds": pytest.approx(0.2), "excluded_from_steady_state": True}  # fmt: skip
    assert s["steady_state"]["trial_seconds"]["max"] == pytest.approx(
        0.04
    )  # the slow pass is not in it
    assert d["trials"]["trials"][0]["wall_seconds"] == pytest.approx(
        0.2
    )  # ...but it is preserved raw
    # the four quantities are separate: initialization is not warmup, and end-to-end spans both
    assert s["end_to_end"]["procedure_seconds"] == pytest.approx(0.2 + 0.2)
    assert s["end_to_end"]["measured_seconds"] == pytest.approx(0.2)
    assert set(s["initialization"]) == {"model_load_seconds", "model_load_note", "stress_build_seconds", "data_materialization_seconds"}  # fmt: skip
    assert s["initialization"]["stress_build_seconds"] is None and s["initialization"]["model_load_seconds"] == 0.0  # fmt: skip
    zero = r.run(r.spec(batch_size=30, repeats=2, warmup_trials=0))
    assert zero.summary["warmup"]["trials"] == 0 and docs(r, zero)["statistics"]["warmup_seconds"]["status"] == "UNAVAILABLE"  # fmt: skip


def test_repeated_trials_give_hand_computed_percentiles_and_a_bootstrap_interval(rig: tuple[RWorld, Sim]) -> None:  # fmt: skip
    r, sim = rig
    sim.duration = lambda i, n: 0.01 * (1 + i // 4)  # trial j costs 0.04 * (j + 1)
    a = r.run(r.spec(batch_size=30, repeats=5, warmup_trials=0, measurement={"percentiles": [50, 90], "resamples": 300, "seed": 4}))  # fmt: skip
    ts = docs(r, a)["statistics"]["trial_seconds"]
    assert [round(t["wall_seconds"], 6) for t in docs(r, a)["trials"]["trials"]] == [0.04, 0.08, 0.12, 0.16, 0.2]  # fmt: skip
    assert ts["mean"] == pytest.approx(0.12) and ts["median"] == pytest.approx(0.12) and ts["min"] == pytest.approx(0.04) and ts["max"] == pytest.approx(0.2)  # fmt: skip
    assert ts["std"] == pytest.approx(0.0632455532) and ts["percentiles"] == {"p50": pytest.approx(0.12), "p90": pytest.approx(0.184)}  # fmt: skip
    ci = ts["interval_mean"]
    assert ci["status"] == "MEASURED" and ci["lower"] < 0.12 < ci["upper"] and ci["seed"] == 4 and ci["method"] == "percentile"  # fmt: skip
    again = r.run(r.spec(batch_size=30, repeats=5, warmup_trials=0, measurement={"percentiles": [50, 90], "resamples": 300, "seed": 4}))  # fmt: skip
    assert again.id != a.id  # a repeated measurement is a NEW analysis, never an overwrite


def test_too_few_trials_give_descriptive_statistics_and_no_interval(
    rig: tuple[RWorld, Sim],
) -> None:
    r, _ = rig
    a = r.run(r.spec(repeats=3, warmup_trials=0))
    ts = docs(r, a)["statistics"]["trial_seconds"]
    assert ts["mean"] is not None and ts["interval_mean"]["status"] == "INSUFFICIENT_EVIDENCE"
    assert (
        "lower" not in ts["interval_mean"]
        and a.summary["evidence_status"] == "INSUFFICIENT_EVIDENCE"
    )
    assert a.analysis_status == "PARTIAL" and a.summary["steady_state"]["interval_status"] == "INSUFFICIENT_EVIDENCE"  # fmt: skip


def test_throughput_comes_from_completed_samples_and_measured_time(rig: tuple[RWorld, Sim]) -> None:
    r, sim = rig
    sim.duration = 0.02
    a = r.run(
        r.spec(batch_size=40, repeats=5, warmup_trials=0, n_samples=100)
    )  # 3 batches: 40, 40, 20
    st = docs(r, a)["statistics"]
    assert a.summary["workload"]["samples"] == 100 and a.summary["workload"]["requested_samples"] == 100  # fmt: skip
    assert st["throughput_samples_per_second"]["mean"] == pytest.approx(100 / 0.06)
    assert docs(r, a)["trials"]["trials"][0]["throughput"]["batches_per_second"] == pytest.approx(3 / 0.06)  # fmt: skip
    assert st["amortized_per_sample_seconds"]["mean"] == pytest.approx(0.06 / 100) and "NOT a per-sample latency" in st["amortized_per_sample_seconds"]["note"]  # fmt: skip


# -- batches -----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("batch", "batches", "eff_min", "eff_max", "final"),
    [
        (1, 120, 1, 1, 1),
        (7, 18, 1, 7, 1),
        (30, 4, 30, 30, 30),
        (50, 3, 20, 50, 20),
        (120, 1, 120, 120, 120),
        (500, 1, 120, 120, 120),
    ],
)
def test_requested_and_effective_batch_sizes_including_the_final_partial_batch(
    rig: tuple[RWorld, Sim], batch: int, batches: int, eff_min: int, eff_max: int, final: int
) -> None:
    r, _ = rig
    a = r.run(r.spec(batch_size=batch, repeats=2, warmup_trials=0, n_samples=None))
    b = a.summary["batch_size"]
    assert b["requested"] == batch and b["effective_min"] == eff_min and b["effective_max"] == eff_max and b["final_batch"] == final  # fmt: skip
    assert b["effective_mean"] == pytest.approx(N / batches) and a.summary["workload"]["batches"] == batches  # fmt: skip
    t = docs(r, a)["trials"]["trials"][0]
    assert t["batches_planned"] == batches == len(t["batch_seconds"]) and sum(t["batch_sizes"]) == N  # fmt: skip
    assert t["wall_seconds"] == pytest.approx(0.01 * batches)  # one simulated call per batch
    assert t["throughput"]["samples_per_second"] == pytest.approx(N / (0.01 * batches))


def test_n_samples_limits_the_workload_and_a_larger_request_is_capped_and_says_so(rig: tuple[RWorld, Sim]) -> None:  # fmt: skip
    r, _ = rig
    a = r.run(r.spec(batch_size=32, n_samples=50, repeats=2, warmup_trials=0))
    assert a.summary["workload"]["samples"] == 50 and a.summary["batch_size"]["final_batch"] == 18
    big = r.run(r.spec(batch_size=32, n_samples=5000, repeats=2, warmup_trials=0))
    assert big.summary["workload"]["samples"] == N and big.summary["workload"]["requested_samples"] == 5000  # fmt: skip


def test_each_batch_size_of_a_sweep_is_its_own_persisted_run_and_analysis(rig: tuple[RWorld, Sim]) -> None:  # fmt: skip
    r, _ = rig
    before = n_runs(r)
    units = rc.run_sweep(r.w.registry, r.w.store, r.w.executor, r.investigation, r.spec(repeats=3, warmup_trials=0), [1, 8, 30])  # fmt: skip
    assert n_runs(r) == before + 3 and [u["batch_size"] for u in units] == [1, 8, 30]
    assert len({u["run_id"] for u in units}) == len({u["analysis_id"] for u in units}) == len({u["spec_id"] for u in units}) == 3  # fmt: skip
    for u in units:
        a = View(r.w.registry.get(ResourceAnalysis, u["analysis_id"]))
        assert u["status"] == "COMPLETED" and a.spec["batch_size"] == u["batch_size"] and a.run_id == u["run_id"]  # fmt: skip
        assert a.summary["batch_size"]["requested"] == u["batch_size"]


def test_an_invalid_sweep_executes_nothing(rig: tuple[RWorld, Sim]) -> None:
    r, sim = rig
    before, calls = n_runs(r), sim.calls
    for sizes, match in (([8, 0], "batch_size"), ([8, -1], "batch_size"), ([8, 8], "distinct"), ([], "distinct"), ([True], "batch_size"), ([10**7], "batch_size")):  # fmt: skip
        with pytest.raises(ValidationError, match=match):
            rc.run_sweep(r.w.registry, r.w.store, r.w.executor, r.investigation, r.spec(), sizes)  # fmt: skip
    with pytest.raises(ValidationError, match="distinct"):
        rc.run_sweep(r.w.registry, r.w.store, r.w.executor, r.investigation, r.spec(), list(range(1, 60)))  # fmt: skip
    assert n_runs(r) == before and sim.calls == calls and not r.w.registry.find(ResourceAnalysis)


def test_per_sample_latency_exists_only_at_batch_size_one(rig: tuple[RWorld, Sim]) -> None:
    r, sim = rig
    sim.duration = lambda i, n: 0.001 * (1 + i % 3)
    one = r.run(r.spec(batch_size=1, n_samples=12, repeats=3, warmup_trials=0))
    ps = docs(r, one)["statistics"]["per_sample_seconds"]
    assert ps["status"] == "MEASURED" and ps["n"] == 36 and ps["min"] == pytest.approx(0.001) and ps["max"] == pytest.approx(0.003)  # fmt: skip
    ids = [s[0] for s in docs(r, one)["trials"]["trials"][0]["per_sample_seconds"]]
    assert ids == list(range(12))
    many = r.run(r.spec(batch_size=6, n_samples=12, repeats=3, warmup_trials=0))
    un = docs(r, many)["statistics"]["per_sample_seconds"]
    assert un["status"] == "UNAVAILABLE" and "batch size 1" in un["reason"]


def test_predict_proba_is_timed_when_the_model_supports_it(rig: tuple[RWorld, Sim]) -> None:
    r, sim = rig
    a = r.run(r.spec(operation="PREDICT_PROBA", repeats=2, warmup_trials=0))
    assert a.summary["workload"]["operation"] == "PREDICT_PROBA" and sim.calls == 8


# -- failures and timeouts ---------------------------------------------------------------------------------------


def test_a_failed_batch_is_recorded_with_its_partial_work_and_never_counts_as_success(rig: tuple[RWorld, Sim]) -> None:  # fmt: skip
    r, sim = rig
    sim.fail_on = {5}  # the 2nd batch of the first measured pass (calls 0-3 are the warmup pass)
    a = r.run(r.spec(batch_size=30, repeats=4, warmup_trials=1))
    d = docs(r, a)
    t0 = d["trials"]["trials"][1]
    assert t0["phase"] == "MEASURED" and t0["status"] == "FAILED"
    assert (t0["completed_samples"], t0["failed_samples"], t0["not_attempted_samples"]) == (
        30,
        30,
        60,
    )
    assert t0["batches_completed"] == 1 and t0["batches_failed"] == 1 and t0["throughput"]["samples_per_second"] is None  # fmt: skip
    assert t0["error"] == {"type": "RuntimeError", "message": "simulated inference failure on call 5", "batch_index": 1}  # fmt: skip
    assert t0["outputs_digest"] is None and t0["per_sample_seconds"] is None
    st = d["statistics"]
    assert (
        st["trials"]["used"] == 3 and st["trials"]["failed"] == 1 and st["trial_seconds"]["n"] == 3
    )
    assert a.analysis_status == "PARTIAL" and a.summary["failures"]["failed"] == 1 and a.summary["failures"]["failed_samples"] == 30  # fmt: skip
    rec = [t for t in r.w.registry.find(ResourceTrial, analysis_id=a.id) if t.status == "FAILED"]
    assert len(rec) == 1 and rec[0].failed_samples == 30 and rec[0].completed_samples == 30


def test_a_timeout_is_recorded_as_a_timeout_with_partial_work(rig: tuple[RWorld, Sim]) -> None:
    r, _ = rig
    a = r.run(r.spec(batch_size=30, repeats=3, warmup_trials=0, timeout_seconds=0.025))
    t = docs(r, a)["trials"]["trials"][0]
    assert t["status"] == "TIMED_OUT" and t["completed_samples"] == 60 and t["late_samples"] == 30  # fmt: skip
    assert t["not_attempted_samples"] == 30 and t["batches_completed"] == 2 and t["batches_failed"] == 0  # fmt: skip
    assert t["overrun_seconds"] == pytest.approx(0.005) and t["throughput"]["samples_per_second"] is None  # fmt: skip
    assert t["error"] is None and t["outputs_digest"] is None
    st = docs(r, a)["statistics"]
    assert st["trials"]["used"] == 0 and st["trials"]["timed_out"] == 3 and st["trial_seconds"]["status"] == "UNAVAILABLE"  # fmt: skip
    assert (
        a.analysis_status == "PARTIAL" and a.summary["evidence_status"] == "INSUFFICIENT_EVIDENCE"
    )
    assert (
        a.summary["steady_state"]["status"] == "UNAVAILABLE"
    )  # nothing completed: nothing to report
    assert "cooperative" in a.summary["failures"]["timeout_semantics"]
    assert {t.status for t in r.w.registry.find(ResourceTrial, analysis_id=a.id)} == {"TIMED_OUT"}


def test_a_generous_timeout_succeeds_and_a_tight_one_fails_at_the_boundary(rig: tuple[RWorld, Sim]) -> None:  # fmt: skip
    r, _ = rig
    ok = r.run(r.spec(repeats=2, warmup_trials=0, timeout_seconds=1.0))
    assert {t["status"] for t in docs(r, ok)["trials"]["trials"]} == {"COMPLETED"} and ok.spec["timeout_seconds"] == 1.0  # fmt: skip
    just = r.run(r.spec(repeats=2, warmup_trials=0, timeout_seconds=0.0401))  # a pass costs 0.04
    assert {t["status"] for t in docs(r, just)["trials"]["trials"]} == {"COMPLETED"}
    late = r.run(
        r.spec(repeats=2, warmup_trials=0, timeout_seconds=0.0399)
    )  # the LAST batch finishes late
    lt = docs(r, late)["trials"]["trials"][0]
    assert lt["status"] == "TIMED_OUT" and lt["completed_samples"] == 90 and lt["late_samples"] == 30 and lt["not_attempted_samples"] == 0  # fmt: skip


def test_a_timeout_in_a_warmup_pass_is_recorded_and_does_not_hide_the_measured_passes(rig: tuple[RWorld, Sim]) -> None:  # fmt: skip
    r, sim = rig
    sim.duration = lambda i, n: 0.05 if i < 4 else 0.005
    a = r.run(r.spec(batch_size=30, repeats=5, warmup_trials=1, timeout_seconds=0.1))
    ts = docs(r, a)["trials"]["trials"]
    assert ts[0]["phase"] == "WARMUP" and ts[0]["status"] == "TIMED_OUT"
    assert {t["status"] for t in ts[1:]} == {"COMPLETED"} and a.analysis_status == "COMPLETE"  # fmt: skip
    assert a.summary["failures"]["timed_out"] == 0  # warmup outcomes are not steady-state failures


# -- memory and CPU ------------------------------------------------------------------------------------------------


def test_memory_reports_the_process_high_water_mark_before_and_after_each_phase(rig: tuple[RWorld, Sim]) -> None:  # fmt: skip
    r, sim = rig
    sim.rss_step = MB  # every inference call pushes the peak up by 1 MB
    a = r.run(r.spec(batch_size=30, repeats=5, warmup_trials=1))
    m = a.summary["memory"]
    assert m["status"] == "MEASURED" and m["high_water_at_procedure_start_bytes"] == 200 * MB
    assert m["high_water_after_initialization_bytes"] == 200 * MB and m["high_water_after_warmup_bytes"] == 204 * MB  # fmt: skip
    assert m["high_water_after_measured_bytes"] == 224 * MB and m["peak_bytes"] == 224 * MB
    assert m["growth_during_measured_bytes"] == 20 * MB and m["gpu"].startswith("UNAVAILABLE")
    assert "model-load memory delta is UNAVAILABLE" in m["baseline_note"] and "never decreases" in m["high_water_note"]  # fmt: skip
    t = docs(r, a)["trials"]["trials"][2]
    assert t["rss_before_bytes"] == 208 * MB and t["rss_after_bytes"] == 212 * MB and t["rss_growth_bytes"] == 4 * MB  # fmt: skip
    assert docs(r, a)["statistics"]["memory_growth_bytes"]["mean"] == pytest.approx(4 * MB)


def test_zero_growth_means_the_earlier_peak_was_not_exceeded_not_that_no_memory_was_used(rig: tuple[RWorld, Sim]) -> None:  # fmt: skip
    r, _ = rig
    a = r.run(r.spec(repeats=3, warmup_trials=1))
    m = a.summary["memory"]
    assert m["growth_during_measured_bytes"] == 0 and m["peak_bytes"] == 200 * MB
    assert docs(r, a)["statistics"]["memory_growth_bytes"]["mean"] == 0.0
    assert "not that no memory was used" in m["high_water_note"]


def test_memory_is_unavailable_when_the_backend_is_missing_or_not_requested(tmp_path: Path) -> None:
    sim = Sim()
    SimLinear.SIM = sim
    try:
        r = resource_world(tmp_path)
        with ms.use_probe(sim.probe(memory=False)):
            a = r.run(r.spec(repeats=2, warmup_trials=0))
            t = docs(r, a)["trials"]["trials"][0]
            assert a.summary["memory"]["status"] == "UNAVAILABLE" and "peak_bytes" not in a.summary["memory"]  # fmt: skip
            assert t["rss_before_bytes"] is None and t["rss_growth_bytes"] is None
            assert docs(r, a)["statistics"]["memory_growth_bytes"]["status"] == "UNAVAILABLE"
        with ms.use_probe(sim.probe()):
            off = r.run(r.spec(repeats=2, warmup_trials=0, measurement={"memory": False}))
            assert off.summary["memory"]["status"] == "UNAVAILABLE"  # a backend exists but was not requested  # fmt: skip
    finally:
        SimLinear.SIM = None


def test_cpu_is_process_cpu_time_and_utilization_is_never_inferred_from_wall_time(tmp_path: Path) -> None:  # fmt: skip
    sim = Sim(duration=0.01, cpu_per_call=0.004)
    SimLinear.SIM = sim
    try:
        r = resource_world(tmp_path)
        with ms.use_probe(sim.probe()):
            a = r.run(r.spec(repeats=3, warmup_trials=0))
            c = a.summary["cpu"]
            assert c["status"] == "MEASURED" and c["cpu_seconds"]["mean"] == pytest.approx(0.016)
            assert c["utilization"]["mean"] == pytest.approx(0.4) and "not machine utilization" in c["utilization"]["definition"]  # fmt: skip
            assert [t.cpu_seconds for t in r.w.registry.find(ResourceTrial, analysis_id=a.id)] == [pytest.approx(0.016)] * 3  # fmt: skip
        with ms.use_probe(sim.probe(cpu=False)):
            n = r.run(r.spec(repeats=3, warmup_trials=0))
            assert n.summary["cpu"]["status"] == "UNAVAILABLE" and n.summary["cpu"]["cpu_seconds"]["status"] == "UNAVAILABLE"  # fmt: skip
            t = docs(r, n)["trials"]["trials"][0]
            assert t["cpu_seconds"] is None and t["cpu_utilization"] is None  # not wall/wall
            assert all(x.cpu_seconds is None for x in r.w.registry.find(ResourceTrial, analysis_id=n.id))  # fmt: skip
        with ms.use_probe(sim.probe()):
            off = r.run(r.spec(repeats=3, warmup_trials=0, measurement={"cpu": False}))
            assert off.summary["cpu"]["status"] == "UNAVAILABLE"
    finally:
        SimLinear.SIM = None


class _Usage:
    def __init__(self, kib: int) -> None:
        self.ru_maxrss = kib


@pytest.mark.parametrize(("platform_name", "expected"), [("linux", 1_024_000), ("darwin", 1000), ("freebsd14", None), ("win32", None)])  # fmt: skip
def test_the_system_memory_backend_knows_its_platforms_units_and_refuses_the_rest(
    monkeypatch: pytest.MonkeyPatch, platform_name: str, expected: int | None
) -> None:
    import resource

    monkeypatch.setattr(sys, "platform", platform_name)
    monkeypatch.setattr(resource, "getrusage", lambda _who: _Usage(1000))
    p = ms._system_probe()
    if expected is None:
        assert p.rss is None and p.describe()["memory"] == "UNAVAILABLE"
    else:
        assert p.rss is not None and p.rss() == expected and "high-water" in str(p.describe()["memory"])  # fmt: skip


# -- refused before execution ------------------------------------------------------------------------------------


def test_unsupported_requests_are_refused_as_unavailable_and_execute_nothing(rig: tuple[RWorld, Sim]) -> None:  # fmt: skip
    r, sim = rig
    before, calls = n_runs(r), sim.calls
    for over, match in (
        ({"memory_limit_mb": 512}, "memory_limit"), ({"workers": 2}, "concurrency"),
        ({"stress": [{"family": "THRESHOLD", "parameters": {"threshold": 0.4}}], "workers": 3}, "concurrency"),
    ):  # fmt: skip
        with pytest.raises(re_.ResourceUnavailable, match=f"UNAVAILABLE.*{match}"):
            r.run(r.spec(**over))
    with pytest.raises(ValidationError, match="unknown split"):
        r.run(r.spec(split="nope"))
    with pytest.raises(ExperionyxError):
        r.run(r.spec(model_id="mdl_" + "0" * 32))
    assert n_runs(r) == before and sim.calls == calls and not r.w.registry.find(ResourceAnalysis)


def test_an_impossible_timeout_is_refused(tmp_path: Path) -> None:
    sim = Sim()
    SimLinear.SIM = sim
    try:
        r = resource_world(tmp_path)
        with ms.use_probe(sim.probe(resolution=0.5)), pytest.raises(re_.ResourceUnavailable, match="timer resolution"):  # fmt: skip
            r.run(r.spec(timeout_seconds=0.4))
    finally:
        SimLinear.SIM = None


def test_the_capability_report_names_what_is_and_is_not_available(rig: tuple[RWorld, Sim]) -> None:
    r, sim = rig
    model = SimLinear([1.0, 0.5], 0.0, "f" * 64, "1")
    rep = {c["capability"]: c for c in re_.capability_report(r.spec(workers=2, memory_limit_mb=64, timeout_seconds=1.0), model, sim.probe(cpu=False, memory=False))}  # fmt: skip
    assert (
        rep["operation:PREDICT"]["status"] == "SUPPORTED"
        and rep["timeout"]["status"] == "SUPPORTED"
    )
    assert rep["concurrency"]["status"] == rep["memory_limit"]["status"] == "UNAVAILABLE" and rep["concurrency"]["blocking"]  # fmt: skip
    assert rep["cpu_accounting"]["status"] == rep["memory_accounting"]["status"] == "UNAVAILABLE"
    assert not rep["cpu_accounting"]["blocking"] and not rep["memory_accounting"]["blocking"]  # reported, not refused  # fmt: skip
    with pytest.raises(re_.ResourceUnavailable):
        re_.refuse_unavailable(list(rep.values()))


# -- concurrency (real threads, real clock) ---------------------------------------------------------------------


def _real_world(tmp_path: Path, cls: type = SleepLinear) -> RWorld:
    SleepLinear.ACTIVE[:] = [0, 0]
    return resource_world(tmp_path, cls)


def test_concurrent_workers_really_overlap_preserve_outputs_and_are_measured(
    tmp_path: Path,
) -> None:
    r = _real_world(tmp_path)
    seq = r.run(r.spec(batch_size=8, repeats=5, warmup_trials=0, workers=1))
    assert SleepLinear.ACTIVE[1] == 1  # one at a time
    SleepLinear.ACTIVE[:] = [0, 0]
    par = r.run(r.spec(batch_size=8, repeats=5, warmup_trials=0, workers=4))
    assert (
        SleepLinear.ACTIVE[1] >= 2 and SleepLinear.ACTIVE[0] == 0
    )  # threads overlapped and all finished
    assert par.summary["concurrency"]["workers"] == 4 and par.summary["concurrency"]["adapter_declared_thread_safe"] is True  # fmt: skip
    assert par.summary["outputs"]["consistent_across_trials"] is True
    assert (
        par.summary["outputs"]["digest"] == seq.summary["outputs"]["digest"]
    )  # batch order is preserved
    assert par.analysis_status == "COMPLETE" and par.spec["workers"] == 4

    def med(x: View) -> float:
        return float(x.summary["steady_state"]["trial_seconds"]["median"])

    assert med(par) < med(
        seq
    )  # sleep-bound work: four workers finish sooner (a robust, coarse check)
    assert not [t for t in threading.enumerate() if t.name.startswith("experionyx-resource")]  # cleaned up  # fmt: skip


def test_a_concurrent_timeout_cancels_pending_work_drains_running_work_and_leaves_no_threads(tmp_path: Path) -> None:  # fmt: skip
    r = _real_world(tmp_path)
    a = r.run(r.spec(batch_size=4, repeats=2, warmup_trials=0, workers=2, timeout_seconds=0.02))
    ts = docs(r, a)["trials"]["trials"]
    assert {t["status"] for t in ts} == {"TIMED_OUT"}
    for t in ts:
        assert t["not_attempted_samples"] > 0 and t["completed_samples"] + t["late_samples"] + t["not_attempted_samples"] == N  # fmt: skip
        assert t["overrun_seconds"] >= 0.0 and t["throughput"]["samples_per_second"] is None
    assert SleepLinear.ACTIVE[0] == 0  # no call is still running after the trial was recorded
    assert not [t for t in threading.enumerate() if t.name.startswith("experionyx-resource")]
    assert a.summary["evidence_status"] == "INSUFFICIENT_EVIDENCE"


def test_a_failure_under_concurrency_is_a_failed_trial_with_the_other_batches_counted(tmp_path: Path) -> None:  # fmt: skip
    sim = Sim(duration=0.0)
    sim.fail_on = {2}
    SimLinear.SIM = sim
    try:
        r = resource_world(tmp_path, SafeSimLinear)
        with ms.use_probe(ms.current_probe()):
            a = r.run(r.spec(batch_size=30, repeats=1, warmup_trials=0, workers=3))
        t = docs(r, a)["trials"]["trials"][0]
        assert t["status"] == "FAILED" and t["batches_failed"] == 1 and t["failed_samples"] == 30
        assert t["completed_samples"] == 90 and t["error"]["type"] == "RuntimeError"
    finally:
        SimLinear.SIM = None


def test_a_model_that_does_not_declare_thread_safe_inference_reports_unavailable(tmp_path: Path) -> None:  # fmt: skip
    r = resource_world(tmp_path)  # SimLinear does not declare it
    with pytest.raises(re_.ResourceUnavailable, match="does not declare thread-safe inference"):
        r.run(r.spec(workers=2))
    assert not r.w.registry.find(ResourceAnalysis)


# -- provenance and replay -----------------------------------------------------------------------------------


def test_provenance_records_the_whole_definition_and_the_environment(
    rig: tuple[RWorld, Sim],
) -> None:
    r, _ = rig
    a = r.run(r.spec(batch_size=15, repeats=3, warmup_trials=2, workers=1, timeout_seconds=9.0, seed=5, environment_note="plugged in"))  # fmt: skip
    p = ResourceRegistry(r.w.registry, r.w.store).provenance(a.id)
    assert p["model"]["model_id"] == r.model_id and p["dataset"]["dataset_id"] == r.dataset_id and p["split"] == "test"  # fmt: skip
    assert p["model"]["fingerprint"] and p["dataset"]["fingerprint"] and p["seed"] == 5 and p["stress_ids"] == []  # fmt: skip
    rc_ = p["resource_configuration"]
    assert (rc_["batch_size"], rc_["warmup_trials"], rc_["workers"], rc_["timeout_seconds"]) == (15, 2, 1, 9.0)  # fmt: skip
    assert rc_["measurement"]["percentiles"] == [50.0, 90.0, 95.0, 99.0] and rc_["operation"] == "PREDICT"  # fmt: skip
    assert p["run_provenance"]["seed"] == 5 and p["run_provenance"]["source_revision"] and p["run_provenance"]["environment_id"] == p["environment_id"]  # fmt: skip
    assert p["run_provenance"]["execution"]["resources"]["timeout_seconds"] == 9.0 and p["run_provenance"]["execution"]["resources"]["max_workers"] == 1  # fmt: skip
    assert set(p["artifact_digests"]) == {f"resources/{n}.json" for n in re_.DOCUMENTS}
    assert all(d.startswith("sha256:") for d in p["artifact_digests"].values())
    assert p["environment"]["measurement_backends"]["probe"] == "simulated" and p["environment"]["python_version"]  # fmt: skip
    assert d_env(r, a)["load_average_1min_at_start"] is None or d_env(r, a)["load_average_1min_at_start"] >= 0  # fmt: skip
    assert p["provenance_fingerprint"] == a.provenance_fingerprint and a.summary["environment"]["environment_note"] == "plugged in"  # fmt: skip


def d_env(r: RWorld, a: View) -> dict[str, Any]:
    return dict(ResourceRegistry(r.w.registry, r.w.store).document(a.id, "environment"))


def test_the_same_definition_in_the_same_environment_has_one_fingerprint_and_new_records(rig: tuple[RWorld, Sim]) -> None:  # fmt: skip
    r, _ = rig
    a, b = r.run(r.spec()), r.run(r.spec())
    assert a.provenance_fingerprint == b.provenance_fingerprint and a.spec_id == b.spec_id
    assert a.id != b.id and a.run_id != b.run_id


@pytest.mark.parametrize(
    "change",
    [
        {"batch_size": 16},
        {"workers": 1, "timeout_seconds": 3.0},
        {"repeats": 6},
        {"warmup_trials": 0},
        {"n_samples": 60},
        {"seed": 3},
        {"measurement": {"percentiles": [50.0]}},
        {"environment_note": "hot"},
        {"stress": [{"family": "PARAMETER_SCALE", "parameters": {"factor": 1.05}}]},
    ],
)
def test_a_configuration_change_alters_the_provenance_fingerprint(rig: tuple[RWorld, Sim], change: dict[str, Any]) -> None:  # fmt: skip
    r, _ = rig
    assert r.run(r.spec(**change)).provenance_fingerprint != r.run(r.spec()).provenance_fingerprint


def test_a_different_environment_or_measurement_backend_alters_the_provenance_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sim = Sim()
    SimLinear.SIM = sim
    try:
        r = resource_world(tmp_path)
        with ms.use_probe(sim.probe()):
            base = r.run(r.spec())
        with ms.use_probe(
            sim.probe(memory=False)
        ):  # another measurement backend is another environment
            assert r.run(r.spec()).provenance_fingerprint != base.provenance_fingerprint
        with ms.use_probe(sim.probe(resolution=1e-6)):
            assert r.run(r.spec()).provenance_fingerprint != base.provenance_fingerprint
        real = capture_module.installed_packages

        def other() -> Any:
            pk, issues = real()
            return {**pk, "an-extra-package": "1.0"}, issues

        monkeypatch.setattr(capture_module, "installed_packages", other)
        with ms.use_probe(sim.probe()):
            changed = r.run(r.spec())
        assert changed.provenance_fingerprint != base.provenance_fingerprint
        assert d_env(r, changed)["environment_id"] != d_env(r, base)["environment_id"]
    finally:
        SimLinear.SIM = None


def test_replay_reproduces_the_definition_and_the_outputs_but_never_promises_the_timing(rig: tuple[RWorld, Sim]) -> None:  # fmt: skip
    r, sim = rig
    a = r.run(r.spec(repeats=3, warmup_trials=0))
    sim.duration = 0.05  # the "machine" is now five times slower
    out = re_.replay_check(r.w.registry, r.w.store, r.w.executor, a.id)
    assert out["definition_reproduced"] is True and out["outputs_reproduced"] is True and out["differences"] == []  # fmt: skip
    assert out["timing"] == "NOT_EXPECTED_TO_REPRODUCE" and out["compared"] == ["spec", "outputs"]
    med = out["informational_median_trial_seconds"]
    assert med["original"] == pytest.approx(0.04) and med["replay"] == pytest.approx(0.2)  # different, and fine  # fmt: skip
    new = r.w.registry.get(ResourceAnalysis, str(out["replay_analysis"]))
    assert new.id != a.id and new.run_id == out["replay_run"] and new.spec_id == a.spec_id
    assert new.provenance_fingerprint == a.provenance_fingerprint


def test_replay_of_a_changed_model_is_refused_at_binding_time(rig: tuple[RWorld, Sim]) -> None:
    r, _ = rig
    a = r.run(r.spec(repeats=2, warmup_trials=0))
    (r.w.workspace / "m" / "model.json").write_text(json.dumps({"coef": [9.0, 9.0], "intercept": 1.0}), encoding="utf-8")  # fmt: skip
    with pytest.raises(ExperionyxError, match=r"fingerprint|changed"):
        re_.replay_check(r.w.registry, r.w.store, r.w.executor, a.id)
    with pytest.raises(ExperionyxError):
        r.run(
            r.spec(repeats=2, warmup_trials=0)
        )  # ...and so is a fresh measurement of the changed model


# -- persistence and artifacts ----------------------------------------------------------------------------------


def test_the_analysis_trials_observations_and_six_artifacts_round_trip(rig: tuple[RWorld, Sim]) -> None:  # fmt: skip
    r, _ = rig
    a = r.run(r.spec(repeats=4, warmup_trials=1))
    rr = ResourceRegistry(r.w.registry, r.w.store)
    assert r.w.registry.get(ResourceAnalysis, a.id) == a.entity and rr.analysis(a.id) == a.entity
    assert [x.id for x in rr.analyses(spec_id=a.spec_id)] == [a.id]
    trials = rr.trials(a.id)
    assert [(t.phase, t.trial_index) for t in trials] == [("MEASURED", i) for i in range(4)] + [("WARMUP", 0)]  # fmt: skip
    assert all(ResourceTrial.from_dict(t.to_dict()) == t for t in trials)
    assert {x.path for x in rr.artifacts(a.id)} == {f"resources/{n}.json" for n in re_.DOCUMENTS}
    d = docs(r, a)
    assert set(d) == set(re_.DOCUMENTS) and d["spec"]["spec_id"] == a.spec_id and d["spec"]["spec"] == to_jsonable(a.spec)  # fmt: skip
    assert (
        len(d["trials"]["trials"]) == 5 and d["trials"]["warmup_excluded_from_statistics"] is True
    )
    assert [o["value"] for o in d["observations"]["observations"] if o["name"] == "resources.trial_wall_seconds"] == pytest.approx([0.04] * 4)  # fmt: skip
    assert d["summary"]["provenance_fingerprint"] == a.provenance_fingerprint and set(d["summary"]["artifacts"]) == set(re_.DOCUMENTS) - {"summary"} and set(a.summary["artifacts"]) == set(re_.DOCUMENTS)  # fmt: skip
    obs = [o for o in r.w.registry.find(Observation, run_id=a.run_id) if o.name == "resources.trial_wall_seconds"]  # fmt: skip
    assert len(obs) == 4 and all(
        o.unit == "s" and o.epistemic_kind.value == "OBSERVATION" for o in obs
    )
    assert any("environment-specific" in x for x in d["summary"]["limitations"])


def _keys(x: Any) -> Iterator[str]:
    if isinstance(x, dict):
        for k, v in x.items():
            yield str(k)
            yield from _keys(v)
    elif isinstance(x, list):
        for v in x:
            yield from _keys(v)


def test_summary_and_every_document_state_that_the_measurement_is_environment_specific(rig: tuple[RWorld, Sim]) -> None:  # fmt: skip
    r, _ = rig
    a = r.run(r.spec(repeats=2, warmup_trials=0))
    d = docs(r, a)
    assert "environment-specific" in d["environment"]["note"] and any("not a universal hardware benchmark" in x for x in d["summary"]["limitations"])  # fmt: skip
    assert any("not reproducible" in x or "vary between executions" in x for x in d["summary"]["limitations"])  # fmt: skip
    assert (
        d["statistics"]["variability"]["environment"]["status"] == "UNAVAILABLE"
    )  # one machine only
    assert not [
        k for k in _keys(d["summary"]) if "score" in k.lower() or "composite" in k.lower()
    ]  # no composite score


# -- adversarial ------------------------------------------------------------------------------------------------


def test_a_tampered_artifact_is_detected_when_it_is_read(rig: tuple[RWorld, Sim]) -> None:
    r, _ = rig
    a = r.run(r.spec(repeats=2, warmup_trials=0))
    store = r.w.store
    assert isinstance(store, LocalArtifactStore)
    target = (
        store.run_dir(r.w.registry.get(Run, a.run_id)) / "artifacts" / "resources" / "trials.json"
    )
    target.write_text(target.read_text().replace('"status": "COMPLETED"', '"status": "FAILED"', 1), encoding="utf-8")  # fmt: skip
    rr = ResourceRegistry(r.w.registry, r.w.store)
    with pytest.raises(ArtifactIntegrityError):
        rr.document(a.id, "trials")
    assert (
        rr.document(a.id, "statistics")["trials"]["used"] == 2
    )  # only the tampered document is refused
    with pytest.raises(ArtifactIntegrityError):
        rc.compare_analyses(r.w.registry, r.w.store, [a.id, r.run(r.spec(repeats=2, warmup_trials=0)).id])  # fmt: skip


def test_a_corrupted_resource_record_is_detected(rig: tuple[RWorld, Sim]) -> None:
    r, _ = rig
    a = r.run(r.spec(repeats=2, warmup_trials=0))
    raw = r.w.registry._conn
    raw.execute("DROP TRIGGER resource_analyses_immutable")
    payload = a.to_dict() | {"model_id": "mdl_" + "0" * 32}
    raw.execute(
        "UPDATE resource_analyses SET payload = ? WHERE id = ?", (json.dumps(payload), a.id)
    )
    with pytest.raises(CorruptRecordError):
        r.w.registry.get(ResourceAnalysis, a.id)


def test_resource_records_are_immutable_undeletable_and_reject_duplicate_identities(rig: tuple[RWorld, Sim]) -> None:  # fmt: skip
    r, _ = rig
    a = r.run(r.spec(repeats=2, warmup_trials=0))
    (t,) = [x for x in r.w.registry.find(ResourceTrial, analysis_id=a.id) if x.trial_index == 0]
    with pytest.raises(DuplicateError):
        r.w.registry.add(t)  # the same trial again
    other = ResourceTrial(t.analysis_id, t.phase, t.trial_index, "COMPLETED", t.planned_samples, t.completed_samples, 0, 99.0, None, t.created_at)  # fmt: skip
    assert other.id == t.id
    with pytest.raises(DuplicateError):
        r.w.registry.add(other)  # the same identity with different content is not an overwrite
    assert r.w.registry.get(ResourceTrial, t.id).wall_seconds == pytest.approx(0.04)
    raw = r.w.registry._conn
    with pytest.raises(sqlite3.DatabaseError, match="immutable"):
        raw.execute("UPDATE resource_trials SET payload = '{}'")
    with pytest.raises(sqlite3.DatabaseError, match="cannot be deleted"):
        raw.execute("DELETE FROM resource_analyses")


def _trial(phase: str, i: int, wall: float = 0.04) -> dict[str, Any]:
    return {"phase": phase, "trial_index": i, "status": "COMPLETED", "planned_samples": 10, "completed_samples": 10, "failed_samples": 0, "wall_seconds": wall, "cpu_seconds": None, "cpu_utilization": None, "batch_seconds": [wall], "batch_sizes": [10], "rss_growth_bytes": None, "throughput": {"samples_per_second": 10 / wall, "batches_per_second": 1 / wall}, "per_sample_seconds": None}  # fmt: skip


def test_statistics_do_not_depend_on_the_order_the_measurements_are_listed_in(rig: tuple[RWorld, Sim]) -> None:  # fmt: skip
    r, _ = rig
    spec = r.spec(repeats=6, warmup_trials=0)
    trials = [_trial("MEASURED", i, w) for i, w in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])]
    fwd = re_.statistics_doc(spec, trials, ms.current_probe())
    assert fwd == re_.statistics_doc(spec, list(reversed(trials)), ms.current_probe())
    assert fwd == re_.statistics_doc(
        spec, [trials[i] for i in (3, 0, 5, 1, 4, 2)], ms.current_probe()
    )
    assert fwd["variability"]["measurement"]["lag1_autocorrelation_trial_seconds"] == pytest.approx(0.5)  # fmt: skip


def test_the_same_values_in_a_different_trial_order_change_only_the_drift_diagnostic(rig: tuple[RWorld, Sim]) -> None:  # fmt: skip
    r, _ = rig
    spec = r.spec(repeats=6, warmup_trials=0)
    drifting = [_trial("MEASURED", i, w) for i, w in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])]
    shuffled = [_trial("MEASURED", i, w) for i, w in enumerate([0.3, 0.1, 0.5, 0.2, 0.6, 0.4])]
    a, b = re_.statistics_doc(spec, drifting, ms.current_probe()), re_.statistics_doc(spec, shuffled, ms.current_probe())  # fmt: skip
    for k in ("trial_seconds", "throughput_samples_per_second"):
        assert a[k] == b[k]  # a bootstrap over exchangeable trials does not see the order
    assert a["variability"]["measurement"]["lag1_autocorrelation_trial_seconds"] != b["variability"]["measurement"]["lag1_autocorrelation_trial_seconds"]  # fmt: skip


def test_a_missing_trial_is_reported_and_a_duplicate_trial_identity_is_refused(rig: tuple[RWorld, Sim]) -> None:  # fmt: skip
    r, _ = rig
    spec = r.spec(repeats=4, warmup_trials=0)
    trials = [_trial("MEASURED", i) for i in (0, 1, 3)]
    st = re_.statistics_doc(spec, trials, ms.current_probe())
    assert (
        st["trials"]["missing"] == [2]
        and st["trials"]["used"] == 3
        and st["trials"]["planned"] == 4
    )
    with pytest.raises(re_.ResourceError, match="duplicate trial identity"):
        re_.statistics_doc(spec, [*trials, _trial("MEASURED", 1, 0.09)], ms.current_probe())
    assert re_.check_trials([_trial("WARMUP", 0), _trial("MEASURED", 0)], 1) == []  # a warmup 0 is not measured 0  # fmt: skip


def test_a_measurement_of_only_failures_reports_no_statistics_not_zeros(rig: tuple[RWorld, Sim]) -> None:  # fmt: skip
    r, _ = rig
    bad = {**_trial("MEASURED", 0), "status": "FAILED", "completed_samples": 0, "failed_samples": 10, "throughput": {"samples_per_second": None, "batches_per_second": None}}  # fmt: skip
    st = re_.statistics_doc(r.spec(repeats=1, warmup_trials=0), [bad], ms.current_probe())
    assert st["trials"]["used"] == 0 and st["trial_seconds"]["status"] == "UNAVAILABLE" and st["trial_seconds"]["mean"] is None  # fmt: skip
    assert st["throughput_samples_per_second"]["status"] == "UNAVAILABLE"
