"""The resource domain and its pure measurement code: deterministic identity, normalization, refusal of
invalid definitions, the probe abstraction (simulated, unavailable and the real system probe), and
percentile / throughput / interval arithmetic checked against values computed by hand."""

import math
import sys
from datetime import UTC, datetime
from typing import Any

import pytest

from experionyx.errors import ValidationError
from experionyx.resources import measure as ms
from experionyx.resources.entities import ResourceAnalysis, ResourceTrial
from experionyx.resources.spec import ResourceSpec, Subset
from resource_helpers import Sim

M = "mdl_" + "a" * 32
D = "dst_" + "b" * 32
BASE: dict[str, Any] = {"model_id": M, "dataset_id": D, "split": "test"}
NOW = datetime(2026, 9, 21, tzinfo=UTC)


def spec(**over: Any) -> ResourceSpec:
    return ResourceSpec.from_dict({**BASE, **over})


# -- identity and normalization ---------------------------------------------------------------------------------


def test_identity_is_deterministic_order_independent_and_round_trips() -> None:
    a = spec(batch_size=8, repeats=3)
    assert a.spec_id.startswith("rsp_") and len(a.spec_id) == 4 + 32
    assert spec(repeats=3, batch_size=8).spec_id == a.spec_id  # key order is irrelevant
    assert ResourceSpec.from_dict(a.to_dict()) == a and ResourceSpec.from_dict(a.to_dict()).spec_id == a.spec_id  # fmt: skip
    assert spec().to_dict().keys() == ResourceSpec.from_dict(spec().to_dict()).to_dict().keys()
    d = spec().to_dict()
    assert not {"n_samples", "timeout_seconds", "memory_limit_mb", "subset", "stress", "environment_note", "calibration_analyses"} & set(d)  # optional blocks are omitted  # fmt: skip


@pytest.mark.parametrize(
    "change",
    [
        {"batch_size": 16},
        {"n_samples": 50},
        {"repeats": 11},
        {"warmup_trials": 0},
        {"workers": 2},
        {"timeout_seconds": 5.0},
        {"operation": "PREDICT_PROBA"},
        {"split": "train"},
        {"seed": 1},
        {"device": "MPS"},
        {"environment_note": "on battery"},
        {"model_id": "mdl_" + "c" * 32},
        {"dataset_id": "dst_" + "c" * 32},
        {"measurement": {"percentiles": [50.0, 99.0]}},
        {"measurement": {"resamples": 501}},
        {"measurement": {"cpu": False}},
        {"measurement": {"memory": False}},
        {"measurement": {"min_trials": 3}},
        {"measurement": {"seed": 7}},
        {"measurement": {"confidence": 0.9}},
        {"stress": [{"family": "PARAMETER_SCALE", "parameters": {"factor": 1.1}}]},
        {"stress_analyses": ["sxa_" + "d" * 32]},
        {"calibration_analyses": ["cba_" + "d" * 32]},
        {"quality_analyses": ["qan_" + "d" * 32]},
        {
            "subset": {
                "kind": "SLICE",
                "baseline_run": "run_" + "e" * 32,
                "slice": {
                    "name": "s",
                    "condition": {"op": "range", "field": "feature:x0", "low": 0.0, "high": 5.0},
                },
            }
        },
    ],
)
def test_every_meaningful_configuration_change_alters_the_identity(change: dict[str, Any]) -> None:
    assert spec(**change).spec_id != spec().spec_id


def test_normalization_makes_equivalent_definitions_one_identity() -> None:
    assert spec(measurement={"percentiles": [50, 95]}).spec_id == spec(measurement={"percentiles": [50.0, 95.0]}).spec_id  # fmt: skip
    a, b = "cba_" + "1" * 32, "cba_" + "2" * 32
    assert spec(calibration_analyses=[b, a, a]).spec_id == spec(calibration_analyses=[a, b]).spec_id  # fmt: skip
    assert spec(timeout_seconds=5).spec_id == spec(timeout_seconds=5.0).spec_id


# -- refusal of invalid definitions --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("over", "match"),
    [
        ({"batch_size": 0}, "batch_size"), ({"batch_size": -4}, "batch_size"), ({"batch_size": True}, "batch_size"),
        ({"batch_size": 1.5}, "batch_size"), ({"batch_size": "8"}, "batch_size"), ({"batch_size": 10**7}, "batch_size"),
        ({"repeats": 0}, "repeats"), ({"repeats": 501}, "repeats"), ({"warmup_trials": -1}, "warmup_trials"),
        ({"n_samples": 0}, "n_samples"), ({"workers": 0}, "workers"), ({"workers": 65}, "workers"),
        ({"workers": 2.0}, "workers"), ({"timeout_seconds": 0}, "timeout_seconds"),
        ({"timeout_seconds": -1.0}, "timeout_seconds"), ({"timeout_seconds": math.nan}, "timeout_seconds"),
        ({"timeout_seconds": math.inf}, "timeout_seconds"), ({"timeout_seconds": "soon"}, "timeout_seconds"),
        ({"timeout_seconds": 10**9}, "timeout_seconds"), ({"memory_limit_mb": 0}, "memory_limit_mb"),
        ({"operation": "TRAIN"}, "operation"), ({"device": "TPU"}, "device"), ({"split": ""}, "split"),
        ({"model_id": "nope"}, "model_id"), ({"dataset_id": M}, "dataset_id"), ({"seed": -1}, "seed"),
        ({"unknown_field": 1}, "unexpected"), ({"resource_version": "9"}, "resource_version"),
        ({"measurement": {"percentiles": []}}, "percentiles"), ({"measurement": {"percentiles": [0]}}, "percentiles"),
        ({"measurement": {"percentiles": [100]}}, "percentiles"), ({"measurement": {"percentiles": [95, 50]}}, "percentiles"),
        ({"measurement": {"percentiles": [50, 50]}}, "percentiles"), ({"measurement": {"percentiles": "p50"}}, "percentiles"),
        ({"measurement": {"confidence": 1.0}}, "confidence"), ({"measurement": {"resamples": 0}}, "resamples"),
        ({"measurement": {"min_trials": 1}}, "min_trials"), ({"measurement": {"cpu": "yes"}}, "boolean"),
        ({"measurement": {"interval_method": "bca"}}, "interval_method"), ({"measurement": {"bogus": 1}}, "unexpected"),
        ({"stress": [{"family": "BATCH_SIZE", "parameters": {"batch_size": 4}}]}, "not a model-level stress"),
        ({"stress": [{"family": "REPEATED_EXECUTION", "parameters": {"repeats": 2}}]}, "not a model-level stress"),
        ({"stress": [{"family": "FEATURE_NOISE", "parameters": {"sigma": 1.0}, "seed": 1}]}, "not a model-level stress"),
        ({"stress": "PARAMETER_NOISE"}, "stress"), ({"stress_analyses": ["x"]}, "stress_analyses"),
        ({"subset": {"kind": "SLICE", "baseline_run": "run_" + "1" * 32}}, "exactly a slice"),
        ({"subset": {"kind": "COHORT", "baseline_run": "run_" + "1" * 32}}, "subset kind"),
        ({"subset": {"kind": "WINDOW", "baseline_run": "run_" + "1" * 32}}, "ordering and a window"),
        ({"subset": {"kind": "SLICE"}}, "missing"),
    ],
)  # fmt: skip
def test_invalid_definitions_are_refused_with_a_reason(over: dict[str, Any], match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        spec(**over)


def test_missing_required_fields_are_refused() -> None:
    with pytest.raises(ValidationError, match="missing"):
        ResourceSpec.from_dict({"model_id": M})


def test_a_window_subset_round_trips_and_a_slice_subset_takes_only_a_slice() -> None:
    d = {"kind": "WINDOW", "baseline_run": "run_" + "1" * 32, "ordering": {"field": "index"}, "window": {"name": "late", "start": 60, "end": 119}}  # fmt: skip
    sub = Subset.from_dict(d)
    assert Subset.from_dict(sub.to_dict()) == sub
    assert spec(subset=d).to_dict()["subset"] == sub.to_dict()
    with pytest.raises(ValidationError, match="exactly a slice"):
        Subset.from_dict({**d, "kind": "SLICE"})


# -- percentiles, throughput, intervals, autocorrelation -------------------------------------------------------------


def test_percentiles_use_linear_interpolation_on_the_raw_values() -> None:
    d = ms.describe([float(i) for i in range(1, 101)], (50.0, 90.0, 99.0))
    assert d["n"] == 100 and d["min"] == 1.0 and d["max"] == 100.0 and d["mean"] == 50.5 and d["median"] == 50.5  # fmt: skip
    assert d["percentiles"]["p50"] == pytest.approx(50.5) and d["percentiles"]["p90"] == pytest.approx(90.1)  # fmt: skip
    assert d["percentiles"]["p99"] == pytest.approx(99.01)
    assert d["std"] == pytest.approx(29.011491975882016) and d["status"] == "MEASURED" and d["warnings"] == []  # fmt: skip


def test_describe_is_independent_of_measurement_order_and_flags_thin_tails() -> None:
    xs = [0.5, 0.1, 0.9, 0.3, 0.7]
    assert ms.describe(xs, (50.0, 95.0)) == ms.describe(sorted(xs), (50.0, 95.0))
    d = ms.describe(xs, (50.0, 95.0))
    assert d["percentiles"]["p50"] == 0.5 and d["percentiles"]["p95"] == pytest.approx(0.86)
    assert any("p95 of 5" in w for w in d["warnings"]) and not any("p50" in w for w in d["warnings"])  # fmt: skip


def test_describe_of_nothing_is_unavailable_not_zero_and_one_value_has_no_spread() -> None:
    e = ms.describe([], (50.0,))
    assert e["status"] == "UNAVAILABLE" and e["mean"] is None and e["percentiles"] == {}
    one = ms.describe([2.5], (50.0,))
    assert one["mean"] == 2.5 and one["std"] is None and one["percentiles"] == {"p50": 2.5}


def test_throughput_is_completed_work_over_measured_time_and_never_theoretical() -> None:
    assert ms.throughput(120, 4, 0.04) == {"samples_per_second": pytest.approx(3000.0), "batches_per_second": pytest.approx(100.0)}  # fmt: skip
    none = {"samples_per_second": None, "batches_per_second": None}
    assert ms.throughput(0, 0, 1.0) == none and ms.throughput(10, 1, 0.0) == none and ms.throughput(10, 1, -1.0) == none  # fmt: skip


def test_lag1_autocorrelation_detects_drift_and_is_undefined_for_constants() -> None:
    assert ms.lag1_autocorrelation([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) == pytest.approx(0.5)
    alt = ms.lag1_autocorrelation([1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
    assert alt is not None and alt < -0.5
    assert ms.lag1_autocorrelation([3.0, 3.0, 3.0, 3.0]) is None and ms.lag1_autocorrelation([1.0, 2.0]) is None  # fmt: skip


def test_intervals_need_enough_trials_and_are_reproducible_and_order_independent() -> None:
    xs = [1.0, 1.2, 0.9, 1.1, 1.05, 0.95]
    thin = ms.interval(xs[:3], "mean", 0.95, 200, 0, 5)
    assert thin["status"] == "INSUFFICIENT_EVIDENCE" and "lower" not in thin and "at least 5" in thin["reason"]  # fmt: skip
    a = ms.interval(xs, "mean", 0.95, 400, 3, 5)
    assert a["status"] == "MEASURED" and a["lower"] < a["estimate"] < a["upper"] and a["seed"] == 3
    assert a == ms.interval(
        list(reversed(xs)), "mean", 0.95, 400, 3, 5
    )  # the resampling sorts first
    assert a["estimate"] == pytest.approx(sum(xs) / len(xs))
    assert ms.interval(xs, "median", 0.9, 400, 3, 5)["estimator"] == "median"


# -- the probe -----------------------------------------------------------------------------------------------------


def test_the_simulated_probe_reports_exactly_what_the_script_advanced() -> None:
    sim = Sim(duration=0.25, cpu_per_call=0.1)
    p = sim.probe()
    sim.on_call(4)
    sim.on_call(4)
    assert p.clock() == 0.5 and p.cpu is not None and p.cpu() == pytest.approx(0.2) and p.rss is not None and p.rss() == 200_000_000  # fmt: skip
    d = p.describe()
    assert d["probe"] == "simulated" and d["clock_resolution_seconds"] == 1e-9


def test_missing_backends_are_described_as_unavailable_never_estimated() -> None:
    p = Sim().probe(cpu=False, memory=False)
    assert p.cpu is None and p.rss is None
    d = p.describe()
    assert d["cpu"] == "UNAVAILABLE" and d["memory"] == "UNAVAILABLE" and str(d["gpu_memory"]).startswith("UNAVAILABLE")  # fmt: skip


def test_use_probe_swaps_the_probe_and_restores_it_even_when_the_body_raises() -> None:
    before = ms.current_probe()
    sim = Sim()
    with ms.use_probe(sim.probe()) as p:
        assert ms.current_probe() is p
    assert ms.current_probe() is before
    with pytest.raises(RuntimeError), ms.use_probe(sim.probe()):
        raise RuntimeError("boom")
    assert ms.current_probe() is before


def test_the_system_probe_measures_the_real_process() -> None:
    p = ms.current_probe()
    assert p.name == "system" and p.resolution > 0
    a = p.clock()
    sum(i * i for i in range(20_000))
    assert p.clock() >= a  # monotonic
    assert p.cpu is not None and p.cpu() >= 0.0
    if sys.platform in ("darwin",) or sys.platform.startswith("linux"):
        assert (
            p.rss is not None and (p.rss() or 0) > 1_000_000
        )  # a Python process is more than 1 MB
        first = p.rss() or 0
        blob = bytearray(80_000_000)
        blob[::4096] = b"x" * len(blob[::4096])  # touch the pages
        assert (p.rss() or 0) >= first  # a high-water mark never decreases
        del blob
        assert (p.rss() or 0) >= first
    else:
        assert p.rss is None  # no supported backend: unavailable, not guessed
    assert "GPU" in str(p.describe()["gpu_memory"]) or "gpu" in str(p.describe()["gpu_memory"])


# -- entities ------------------------------------------------------------------------------------------------------


def analysis(**over: Any) -> ResourceAnalysis:
    body: dict[str, Any] = {"investigation_id": "inv_" + "1" * 32, "run_id": "run_" + "2" * 32, "spec_id": "rsp_" + "3" * 32, "spec": {"batch_size": 4}, "model_id": M, "dataset_id": D, "provenance_fingerprint": "sha256:" + "4" * 64, "analysis_status": "COMPLETE", "summary": {"trials": {"used": 5}}, "created_at": NOW}  # fmt: skip
    body.update(over)
    return ResourceAnalysis(**body)


def trial(**over: Any) -> ResourceTrial:
    body: dict[str, Any] = {"analysis_id": "rsa_" + "5" * 32, "phase": "MEASURED", "trial_index": 0, "status": "COMPLETED", "planned_samples": 10, "completed_samples": 10, "failed_samples": 0, "wall_seconds": 0.5, "cpu_seconds": 0.4, "created_at": NOW}  # fmt: skip
    body.update(over)
    return ResourceTrial(**body)


def test_analysis_identity_includes_the_run_so_a_remeasurement_is_a_new_record() -> None:
    a = analysis()
    assert a.id.startswith("rsa_") and analysis().id == a.id
    assert analysis(run_id="run_" + "9" * 32).id != a.id  # same definition, another execution
    assert ResourceAnalysis.from_dict(a.to_dict()) == a


@pytest.mark.parametrize(
    "over",
    [
        {"analysis_status": "DONE"},
        {"provenance_fingerprint": "sha256:xyz"},
        {"run_id": "exp_" + "1" * 32},
        {"model_id": D},
        {"created_at": "yesterday"},
    ],
)
def test_an_analysis_with_a_malformed_field_is_refused(over: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        analysis(**over)


def test_trial_round_trip_and_identity_is_the_position_not_the_timing() -> None:
    t = trial()
    assert t.id.startswith("rst_") and ResourceTrial.from_dict(t.to_dict()) == t
    assert trial(wall_seconds=9.0).id == t.id and trial(trial_index=1).id != t.id and trial(phase="WARMUP").id != t.id  # fmt: skip
    assert trial(cpu_seconds=None).cpu_seconds is None


@pytest.mark.parametrize(
    "over",
    [
        {"phase": "COOLDOWN"}, {"status": "OK"}, {"trial_index": -1}, {"wall_seconds": -0.1},
        {"wall_seconds": math.nan}, {"cpu_seconds": -1.0}, {"completed_samples": 11},
        {"completed_samples": 6, "failed_samples": 5}, {"status": "COMPLETED", "completed_samples": 9},
        {"analysis_id": "rst_" + "1" * 32},
    ],
)  # fmt: skip
def test_a_malformed_or_impossible_trial_is_refused(over: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        trial(**over)


def test_a_timed_out_trial_may_complete_fewer_samples_but_a_completed_one_may_not() -> None:
    t = trial(status="TIMED_OUT", completed_samples=4, failed_samples=0)
    assert t.completed_samples == 4 and t.status == "TIMED_OUT"
