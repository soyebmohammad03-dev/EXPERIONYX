"""The resource & systems procedure, run by the normal execution engine so every measurement is a Run
with provenance, artifacts and observations.

What is measured is REAL: the bound model's `predict` (or `predict_proba`) is called on the bound
dataset's batches, timed with the probe (`measure.py`). One trial = one full pass over the workload.
Warmup passes run first and are recorded but excluded from steady-state statistics. Initialization
(model load, stress build, batch materialization), warmup, steady-state and end-to-end are separate
numbers. Failures and timeouts are recorded as evidence with the work completed before them; a
timed-out or failed trial is never counted as a success and never enters a latency or throughput
statistic.

Timeout semantics: the deadline is COOPERATIVE. It is checked as each batch finishes; an inference
call already running cannot be interrupted from Python, so a trial can overrun its timeout by at most
the batch in flight (recorded as `overrun_seconds`), and the pool of workers is always drained before
the trial ends, so nothing keeps running after a trial is recorded.

Timing is environment-specific engineering measurement, not a universal benchmark and not
reproducible bit for bit. Replay reproduces the DEFINITION (spec, model, dataset, environment
identity) and, where the model is deterministic, the OUTPUT digests; it never promises equal times."""

import hashlib
import json
import os
import platform
import statistics
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from experionyx.adapters.base import DatasetAdapter, ModelAdapter, ParameterAccess
from experionyx.adapters.capabilities import DeviceKind, ModelCapability
from experionyx.adapters.records import RegisteredDataset, RegisteredModel
from experionyx.artifacts import ArtifactStore
from experionyx.domain import (
    ConfigurationRef,
    EnvironmentSnapshot,
    EpistemicKind,
    Experiment,
    ExperimentStatus,
    RunStatus,
    to_jsonable,
)
from experionyx.drift.data import order_keys, resolve_window
from experionyx.errors import ArtifactIntegrityError, ExperionyxError, ValidationError
from experionyx.evaluation.config import _thaw
from experionyx.execution import Executor, RunContext, resolve_procedure
from experionyx.faults.report import read_artifact
from experionyx.hashing import content_hash
from experionyx.interactions.lifecycle import _compare
from experionyx.provenance import ResourceLimits
from experionyx.registry import Registry
from experionyx.resources import measure as ms
from experionyx.resources.entities import ResourceAnalysis, ResourceTrial
from experionyx.resources.spec import ResourceSpec
from experionyx.slices.data import (
    Baseline,
    feature_columns,
    load_baseline,
    sample_table,
)
from experionyx.slices.evaluate import MembershipStatus, evaluate
from experionyx.stats import core as st
from experionyx.stress.capability import StressUnsupported
from experionyx.stress.transform import build_stressed_model

PROCEDURE = "experionyx.resources.engine:run_resource_analysis"
ARTIFACT_DIR = "resources"
DOCUMENTS = ("spec", "environment", "trials", "observations", "statistics", "summary")
DEFINITION_DOCUMENTS = ("spec",)  # replayed and compared exactly; timing documents never are
ANALYSIS_VERSION = "1"
MAX_MESSAGE = 300

METHODOLOGY = (
    "wall time: time.perf_counter around each inference call and around each whole pass; one trial "
    "is one pass over the workload; warmup passes are recorded and excluded from steady-state "
    "statistics; initialization, warmup, steady-state and end-to-end are reported separately"
)
LIMITATIONS = (
    "environment-specific engineering measurement: not a universal hardware benchmark",
    "timings vary between executions; only the definition and (where deterministic) the model outputs are reproducible",
    "repeated trials on one machine are not independent samples of any wider population; an interval describes measurement variability on this machine under these conditions, not hardware or environment variability",
    "peak memory is the process high-water mark (it cannot decrease); no GPU memory is measured",
    "CPU time is this process's user+system time, not machine utilization",
    "the timeout is cooperative: an inference call in flight is not interrupted",
)


class ResourceError(ExperionyxError):
    """The requested measurement cannot be made (reason attached)."""


class ResourceUnavailable(ResourceError):
    """UNAVAILABLE: the platform, the adapter or the workload cannot provide the requested
    capability, and nothing is measured, enforced or estimated in its place."""


# -- capabilities and refusal (before anything runs) ------------------------------------------------------------


def capability_report(
    spec: ResourceSpec, model: ModelAdapter, probe: ms.Probe
) -> list[dict[str, Any]]:
    """What this model, this adapter and this platform can and cannot do for the spec. `blocking`
    entries refuse the request; the others describe what will be reported UNAVAILABLE."""

    def entry(name: str, ok: bool, reason: str, *, blocking: bool = True) -> dict[str, Any]:
        return {
            "capability": name,
            "status": "SUPPORTED" if ok else "UNAVAILABLE",
            "reason": None if ok else reason,
            "blocking": blocking and not ok,
        }

    need = (
        ModelCapability.PREDICT_PROBA
        if spec.operation == "PREDICT_PROBA"
        else ModelCapability.PREDICT
    )
    out = [
        entry(
            f"operation:{spec.operation}",
            need in model.capabilities,
            f"the model does not support {need.value}",
        )
    ]
    if spec.workers > 1:
        safe = getattr(model, "THREAD_SAFE_INFERENCE", False) is True
        out.append(
            entry(
                "concurrency",
                safe,
                f"the {model.NAME} adapter does not declare thread-safe inference, so concurrent execution is not attempted",
            )
        )
    if spec.memory_limit_mb is not None:
        out.append(
            entry(
                "memory_limit",
                False,
                "a memory limit cannot be enforced in-process on this platform (an OS-level limit would also terminate the host process); it is not simulated",
            )
        )
    if spec.timeout_seconds is not None:
        out.append(
            entry(
                "timeout",
                spec.timeout_seconds > probe.resolution,
                f"the timeout is not larger than the timer resolution ({probe.resolution:g} s)",
            )
        )
    if spec.stress:
        ok = isinstance(model, ParameterAccess) or all(c.family == "THRESHOLD" for c in spec.stress)
        out.append(
            entry(
                "model_stress",
                ok,
                f"the {model.NAME} adapter does not expose safe parameter access",
            )
        )
    out.append(
        entry(
            "cpu_accounting",
            probe.cpu is not None,
            "no process CPU clock on this platform",
            blocking=False,
        )
    )
    out.append(
        entry(
            "memory_accounting",
            probe.rss is not None,
            "no process memory backend on this platform",
            blocking=False,
        )
    )
    return out


def refuse_unavailable(report: Sequence[Mapping[str, Any]]) -> None:
    bad = [r for r in report if r["blocking"]]
    if bad:
        raise ResourceUnavailable(
            "UNAVAILABLE: " + "; ".join(f"{r['capability']}: {r['reason']}" for r in bad)
        )


def _load(
    registry: Registry, executor_adapters: Any, inputs_root: Any, spec: ResourceSpec
) -> tuple[ModelAdapter, DatasetAdapter, RegisteredModel, RegisteredDataset]:
    from experionyx.slices.data import load_dataset_record
    from experionyx.stress.capability import load_model

    if executor_adapters is None:
        raise ResourceError("no adapter registry is available to load the model and dataset")
    rec_m, rec_d = (
        registry.get(RegisteredModel, spec.model_id),
        registry.get(RegisteredDataset, spec.dataset_id),
    )
    try:
        model = load_model(registry, executor_adapters, inputs_root, spec.model_id)
    except StressUnsupported as exc:
        raise ResourceError(str(exc)) from exc
    return model, load_dataset_record(executor_adapters, inputs_root, rec_d), rec_m, rec_d


def _check_data(spec: ResourceSpec, dataset: DatasetAdapter) -> int:
    if spec.split is not None and spec.split not in dataset.splits():
        raise ValidationError(
            f"unknown split {spec.split!r} (dataset has {list(dataset.splits())})"
        )
    n = dataset.num_samples(spec.split)
    if n == 0:
        raise ValidationError("the selected split has no samples")
    return n


def preflight(
    registry: Registry, store: ArtifactStore, adapters: Any, inputs_root: Any, spec: ResourceSpec
) -> dict[str, Any]:
    """Refuse before anything is created: the model and dataset must exist and load, the split must
    exist, every requested capability must be available, and every referenced analysis must exist."""
    from experionyx.calibration.entities import CalibrationAnalysis
    from experionyx.data_quality.entities import QualityAnalysis
    from experionyx.stress.entities import StressAnalysis

    model, dataset, _, _ = _load(registry, adapters, inputs_root, spec)
    n = _check_data(spec, dataset)
    report = capability_report(spec, model, ms.current_probe())
    refuse_unavailable(report)
    for cls, ids in (
        (CalibrationAnalysis, spec.calibration_analyses),
        (StressAnalysis, spec.stress_analyses),
        (QualityAnalysis, spec.quality_analyses),
    ):
        for i in ids:
            registry.get(cls, i)
    members: int | None = None
    if spec.subset is not None:
        members = len(_subset_members(registry, store, dataset, spec)[0])
    return {
        "model": model.NAME,
        "samples_in_split": n,
        "subset_members": members,
        "capabilities": report,
    }


# -- the workload -----------------------------------------------------------------------------------------------


def _subset_members(
    reg: Registry, store: ArtifactStore, dataset: DatasetAdapter, spec: ResourceSpec
) -> tuple[set[int], dict[str, Any]]:
    sub = spec.subset
    if sub is None:
        raise ResourceError("no subset was requested")
    base: Baseline = load_baseline(reg, store, sub.baseline_run)
    if base.dataset_fingerprint is not None and base.dataset_fingerprint != dataset.fingerprint():
        raise ResourceError(
            "the subset's baseline run was made over a different dataset (fingerprint mismatch)"
        )
    if base.split != spec.split:
        raise ResourceError(
            f"the subset's baseline run used split {base.split!r}, not {spec.split!r}"
        )
    if sub.kind == "SLICE":
        if sub.slice is None:
            raise ResourceError("a SLICE subset takes a slice")
        m = evaluate(sub.slice, sample_table(base, dataset, sub.slice.fields))
        if m.status is not MembershipStatus.COMPUTED:
            raise ResourceError(
                f"the slice has no measurable members ({m.status.value}: {m.reason})"
            )
        ids = {int(i) for i in m.sample_ids}
        return ids, {
            "kind": "SLICE",
            "slice_id": m.slice_id,
            "name": m.name,
            "members": len(ids),
            "membership_digest": m.membership_digest,
        }
    if sub.ordering is None or sub.window is None:
        raise ResourceError("a WINDOW subset takes an ordering and a window")
    field = sub.ordering.field
    raw = (
        feature_columns(dataset, base.split, [field], set(base.rows), raw=True)
        if field != "index"
        else {}
    )
    order, _ = order_keys(sorted(base.rows), field, raw.get(field, {}))
    w = resolve_window(sub.window, field, order)
    if not w.sample_ids:
        raise ResourceError("the window has no members")
    return set(w.sample_ids), {
        "kind": "WINDOW",
        "window_id": w.window_id,
        "members": w.n,
        "sample_digest": w.digest,
    }


def _take(inputs: Any, positions: list[int]) -> Any:
    """Rows `positions` of a framework-native batch (list, array or tensor)."""
    if isinstance(inputs, list | tuple):
        return [inputs[i] for i in positions]
    return inputs[positions]


@dataclass
class Workload:
    batches: list[tuple[Any, tuple[int, ...]]]  # (native inputs, sample ids) in order
    n_samples: int
    sizes: list[int]

    @property
    def sample_digest(self) -> str:
        return content_hash({"ids": [i for _, ids in self.batches for i in ids]})


def materialize(dataset: DatasetAdapter, spec: ResourceSpec, members: set[int] | None) -> Workload:
    """The workload's batches, prepared BEFORE any timing: the dataset's own batches of the requested
    size, restricted to the subset if one was requested and to the first `n_samples`. A subset thins
    each requested batch, so effective batch sizes can be smaller than requested (they are recorded)."""
    out: list[tuple[Any, tuple[int, ...]]] = []
    total = 0
    for b in dataset.batches(spec.batch_size, spec.split):
        pos = (
            list(range(len(b.indices)))
            if members is None
            else [j for j, i in enumerate(b.indices) if i in members]
        )
        if spec.n_samples is not None:
            pos = pos[: spec.n_samples - total]
        if not pos:
            if spec.n_samples is not None and total >= spec.n_samples:
                break
            continue
        inputs = b.inputs if len(pos) == len(b.indices) else _take(b.inputs, pos)
        out.append((inputs, tuple(b.indices[j] for j in pos)))
        total += len(pos)
        if spec.n_samples is not None and total >= spec.n_samples:
            break
    if not out:
        raise ResourceError("the workload has no samples")
    return Workload(out, total, [len(ids) for _, ids in out])


# -- one pass ---------------------------------------------------------------------------------------------------


@dataclass
class _Call:
    index: int
    size: int
    start: float
    end: float
    ok: bool
    error: str | None
    error_type: str | None
    outputs: tuple[Any, ...] | None
    adapter_seconds: float | None


def _call(
    model: ModelAdapter, op: str, index: int, inputs: Any, ids: tuple[int, ...], probe: ms.Probe
) -> _Call:
    t0 = probe.clock()
    try:
        r = getattr(model, op)(inputs, sample_ids=ids)
        return _Call(
            index, len(ids), t0, probe.clock(), True, None, None, r.outputs, r.inference_seconds
        )
    except Exception as exc:  # a failing inference is a measured outcome, not a crash
        return _Call(
            index,
            len(ids),
            t0,
            probe.clock(),
            False,
            str(exc)[:MAX_MESSAGE],
            type(exc).__name__,
            None,
            None,
        )


def run_pass(
    model: ModelAdapter, spec: ResourceSpec, work: Workload, probe: ms.Probe, phase: str, index: int
) -> dict[str, Any]:
    """One full pass over the workload; the raw record of what happened and how long it took."""
    op = "predict_proba" if spec.operation == "PREDICT_PROBA" else "predict"
    want_cpu, want_mem = (
        spec.measurement.cpu and probe.cpu is not None,
        spec.measurement.memory and probe.rss is not None,
    )
    rss0 = probe.rss() if want_mem and probe.rss else None
    cpu0 = probe.cpu() if want_cpu and probe.cpu else None
    t0 = probe.clock()
    deadline = None if spec.timeout_seconds is None else t0 + spec.timeout_seconds
    calls: list[_Call] = []
    cancelled = 0
    if spec.workers == 1:
        for i, (inputs, ids) in enumerate(work.batches):
            c = _call(model, op, i, inputs, ids, probe)
            calls.append(c)
            if not c.ok or (deadline is not None and c.end > deadline):
                break
    else:
        pool = ThreadPoolExecutor(
            max_workers=spec.workers, thread_name_prefix="experionyx-resource"
        )
        futs: list[Future[_Call]] = []
        try:
            futs = [
                pool.submit(_call, model, op, i, inputs, ids, probe)
                for i, (inputs, ids) in enumerate(work.batches)
            ]
            try:
                for _ in as_completed(
                    futs, timeout=None if deadline is None else max(0.0, deadline - probe.clock())
                ):
                    pass
            except TimeoutError:
                for f in futs:
                    f.cancel()
        finally:
            pool.shutdown(wait=True)  # cleanup: nothing outlives the trial
        calls = sorted((f.result() for f in futs if not f.cancelled()), key=lambda c: c.index)
        cancelled = sum(f.cancelled() for f in futs)
    t1 = probe.clock()
    cpu1 = probe.cpu() if cpu0 is not None and probe.cpu else None
    rss1 = probe.rss() if rss0 is not None and probe.rss else None
    on_time = [c for c in calls if c.ok and (deadline is None or c.end <= deadline)]
    late = [c for c in calls if c.ok and deadline is not None and c.end > deadline]
    failed = [c for c in calls if not c.ok]
    done = sum(c.size for c in on_time)
    if failed:
        status = "FAILED"
    elif late or cancelled or (deadline is not None and t1 > deadline):
        status = "TIMED_OUT"
    else:
        status = "COMPLETED"
    wall = t1 - t0
    cpu = None if cpu0 is None or cpu1 is None else cpu1 - cpu0
    first = failed[0] if failed else None
    rec: dict[str, Any] = {
        "phase": phase,
        "trial_index": index,
        "status": status,
        "planned_samples": work.n_samples,
        "completed_samples": done,
        "failed_samples": sum(c.size for c in failed),
        "late_samples": sum(c.size for c in late),
        "not_attempted_samples": work.n_samples - sum(c.size for c in calls),
        "batches_planned": len(work.batches),
        "batches_completed": len(on_time),
        "batches_failed": len(failed),
        "wall_seconds": wall,
        "cpu_seconds": cpu,
        "cpu_utilization": None if cpu is None or wall <= 0 else cpu / wall,
        "adapter_inference_seconds": sum(c.adapter_seconds or 0.0 for c in on_time),
        "batch_seconds": [c.end - c.start for c in on_time],
        "batch_sizes": [c.size for c in on_time],
        "rss_before_bytes": rss0,
        "rss_after_bytes": rss1,
        "rss_growth_bytes": None if rss0 is None or rss1 is None else rss1 - rss0,
        "overrun_seconds": 0.0 if deadline is None else max(0.0, t1 - deadline),
        "error": None
        if first is None
        else {"type": first.error_type, "message": first.error, "batch_index": first.index},
        "throughput": ms.throughput(done, len(on_time), wall)
        if status == "COMPLETED"
        else {"samples_per_second": None, "batches_per_second": None},
        "per_sample_seconds": None,
        "outputs_digest": None,
    }
    if status == "COMPLETED":  # digest and per-sample views are built AFTER the timed region
        rec["outputs_digest"] = hashlib.sha256(
            repr([o for c in calls for o in (c.outputs or ())]).encode()
        ).hexdigest()
        if all(c.size == 1 for c in calls):
            rec["per_sample_seconds"] = [
                [work.batches[c.index][1][0], c.end - c.start] for c in calls
            ]
    return rec


# -- statistics -------------------------------------------------------------------------------------------------


def _clean(x: Any) -> Any:
    return json.loads(json.dumps(to_jsonable(x), allow_nan=False))


def check_trials(trials: Sequence[Mapping[str, Any]], repeats: int) -> list[int]:
    """Refuse a duplicated trial identity (phase, index); return the measured trial indices that are
    absent. Missing trials are reported, never averaged over as if they had run."""
    keys = [(t["phase"], t["trial_index"]) for t in trials]
    dup = sorted({k for k in keys if keys.count(k) > 1})
    if dup:
        raise ResourceError(f"duplicate trial identity in the measurements: {dup}")
    return [i for i in range(repeats) if ("MEASURED", i) not in keys]


def statistics_doc(
    spec: ResourceSpec, trials: Sequence[Mapping[str, Any]], probe: ms.Probe
) -> dict[str, Any]:
    m = spec.measurement
    missing = check_trials(trials, spec.repeats)
    measured = sorted(
        (t for t in trials if t["phase"] == "MEASURED"), key=lambda t: t["trial_index"]
    )
    warm = sorted((t for t in trials if t["phase"] == "WARMUP"), key=lambda t: t["trial_index"])
    used = [t for t in measured if t["status"] == "COMPLETED"]
    secs = [t["wall_seconds"] for t in used]
    sps = [t["throughput"]["samples_per_second"] for t in used]
    pooled = [s for t in used for s in t["batch_seconds"]]
    per_sample = [s for t in used if t["per_sample_seconds"] for _, s in t["per_sample_seconds"]]
    amortized = [t["wall_seconds"] / t["completed_samples"] for t in used]
    cpu = [t["cpu_seconds"] for t in used if t["cpu_seconds"] is not None]
    util = [t["cpu_utilization"] for t in used if t["cpu_utilization"] is not None]
    growth = [t["rss_growth_bytes"] for t in used if t["rss_growth_bytes"] is not None]

    def block(
        values: Sequence[float], *, ci: bool = True, why: str = "no completed measured trial"
    ) -> dict[str, Any]:
        d = ms.describe(values, m.percentiles)
        if not values:
            d["reason"] = why
        if ci and values:
            d["interval_mean"] = ms.interval(
                values, "mean", m.confidence, m.resamples, m.seed, m.min_trials
            )
            d["interval_median"] = ms.interval(
                values, "median", m.confidence, m.resamples, m.seed, m.min_trials
            )
        return d

    cv = (
        None
        if len(secs) < 2 or sum(secs) == 0
        else (st.summarize(secs).std.value or 0.0) / (sum(secs) / len(secs))
    )
    per_sample_ok = bool(used) and all(t["per_sample_seconds"] for t in used)
    doc = {
        "statistics_version": "1",
        "population": "MEASURED trials with status COMPLETED; warmup, failed and timed-out trials are excluded from every statistic below and preserved in trials.json",
        "trials": {
            "planned": spec.repeats,
            "missing": missing,
            "measured": len(measured),
            "used": len(used),
            "failed": sum(t["status"] == "FAILED" for t in measured),
            "timed_out": sum(t["status"] == "TIMED_OUT" for t in measured),
            "warmup": len(warm),
        },
        "trial_seconds": block(secs),
        "throughput_samples_per_second": block([float(x) for x in sps if x is not None]),
        "batch_seconds": {
            **block(pooled, ci=False),
            "note": "pooled over completed measured trials; batch timings inside a trial are not independent, so no interval is given",
        },
        "per_sample_seconds": block(
            per_sample,
            ci=False,
            why="per-sample latency is measurable only when every batch holds one sample (batch size 1)",
        )
        if per_sample_ok
        else {
            "status": "UNAVAILABLE",
            "reason": "per-sample latency is measurable only when every batch holds one sample (batch size 1)",
        },
        "amortized_per_sample_seconds": {
            **block(amortized, ci=False),
            "note": "trial time / samples: an average over the batch, NOT a per-sample latency",
        },
        "cpu_seconds": block(
            cpu, ci=False, why="process CPU accounting unavailable or not requested"
        ),
        "cpu_utilization": {
            **block(util, ci=False, why="process CPU accounting unavailable or not requested"),
            "definition": "process CPU seconds / wall seconds; can exceed 1 with several threads; not machine utilization",
        },
        "memory_growth_bytes": block(
            [float(x) for x in growth],
            ci=False,
            why="process memory accounting unavailable or not requested",
        ),
        "warmup_seconds": block(
            [t["wall_seconds"] for t in warm], ci=False, why="no warmup trials"
        ),
        "variability": {
            "measurement": {
                "coefficient_of_variation_trial_seconds": cv,
                "lag1_autocorrelation_trial_seconds": ms.lag1_autocorrelation(secs),
                "note": "spread across repeated trials on this machine; a large lag-1 autocorrelation means trials drift (thermal, cache, other load) and the interval assumptions are weakened",
            },
            "statistical_uncertainty": {
                "note": "the bootstrap intervals above (Phase 10) quantify sampling uncertainty of the mean/median of these trials, assuming exchangeable trials"
            },
            "environment": {
                "status": "UNAVAILABLE",
                "reason": "one machine and one environment were measured; hardware/environment variability cannot be estimated from repeated trials here",
            },
        },
        "timer": {
            "resolution_seconds": probe.resolution,
            "warning": None
            if not secs or statistics.median(secs) > 100 * probe.resolution
            else "median trial time is within 100x of the timer resolution: quantization dominates",
        },
        "assumptions": [
            "trials are treated as exchangeable repeated measurements of one workload on one machine; they are not independent samples of a hardware population",
            "the interval method is the bootstrap percentile interval of the mean or median of trial-level values",
        ],
    }
    return dict(_clean(doc))


# -- documents on disk ------------------------------------------------------------------------------------------


def _write(ctx: RunContext, name: str, payload: object) -> str:
    directory = ctx.artifact_dir / ARTIFACT_DIR
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.json").write_text(
        json.dumps(to_jsonable(payload), sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return ctx.register_artifact(
        f"{ARTIFACT_DIR}/{name}.json", name=f"resources-{name}", media_type="application/json"
    ).id


def environment_doc(ctx: RunContext, probe: ms.Probe) -> dict[str, Any]:
    snap = ctx.registry.get(EnvironmentSnapshot, ctx.run.environment_id)
    try:
        load: float | None = os.getloadavg()[0]
    except (OSError, AttributeError):
        load = None
    return {
        "environment_id": snap.id,
        "python_version": snap.python_version,
        "python_implementation": platform.python_implementation(),
        "os": snap.os,
        "machine": snap.machine,
        "platform": platform.platform(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "packages": to_jsonable(snap.packages),
        "source": to_jsonable(ctx.source),
        "device": to_jsonable(ctx.device),
        "measurement_backends": probe.describe(),
        "load_average_1min_at_start": load,
        "note": "environment-specific: these measurements describe this machine, this interpreter and this moment only",
    }


@dataclass(frozen=True)
class ResourceRequest:
    investigation_id: str
    spec: Mapping[str, object]

    def to_parameters(self) -> dict[str, object]:
        return {
            "resource_analysis": {
                "investigation_id": self.investigation_id,
                "spec": dict(self.spec),
            }
        }

    @classmethod
    def from_parameters(cls, parameters: Mapping[str, object]) -> "ResourceRequest":
        if set(parameters) != {"resource_analysis"}:
            raise ValidationError(
                "a resource configuration must be exactly {'resource_analysis': ...}"
            )
        body = _thaw(parameters["resource_analysis"])
        if (
            not isinstance(body, dict)
            or set(body) != {"investigation_id", "spec"}
            or not isinstance(body["spec"], dict)
        ):
            raise ValidationError("'resource_analysis' has the wrong fields")
        return cls(str(body["investigation_id"]), body["spec"])


def _bound(ctx: RunContext, spec: ResourceSpec) -> tuple[ModelAdapter, DatasetAdapter]:
    if (
        ctx.model is None
        or ctx.dataset is None
        or ctx.inputs is None
        or ctx.inputs.model is None
        or ctx.inputs.dataset is None
    ):
        raise ResourceError(
            "a resource measurement needs a registered model and dataset bound to the run"
        )
    if (
        ctx.inputs.model.record_id != spec.model_id
        or ctx.inputs.dataset.record_id != spec.dataset_id
    ):
        raise ResourceError("the run's bound model/dataset differ from the specification's")
    return ctx.model, ctx.dataset


def run_resource_analysis(ctx: RunContext) -> None:
    req = ResourceRequest.from_parameters(ctx.parameters)
    spec = ResourceSpec.from_dict(req.spec)
    model, dataset = _bound(ctx, spec)
    probe = ms.current_probe()
    reg, now = ctx.registry, ctx.started_at
    _check_data(spec, dataset)
    started = probe.clock()
    rss_start = probe.rss() if spec.measurement.memory and probe.rss else None

    # -- initialization: everything before the first timed pass --------------------------------------------------
    base_model = model
    t = probe.clock()
    stress_record: dict[str, Any] | None = None
    if spec.stress:
        try:
            build = build_stressed_model(model, spec.stress)
        except StressUnsupported as exc:
            raise ResourceUnavailable(f"UNAVAILABLE: {exc}") from exc
        model, stress_record = build.model, build.record
    stress_seconds = probe.clock() - t
    refuse_unavailable(capability_report(spec, base_model, probe))
    members, subset_record = (
        (None, None) if spec.subset is None else _subset_members(reg, ctx.store, dataset, spec)
    )
    t = probe.clock()
    work = materialize(dataset, spec, members)
    prepare_seconds = probe.clock() - t
    rss_init = probe.rss() if rss_start is not None and probe.rss else None
    try:
        model_load_seconds: float | None = float(base_model.load_seconds)
    except (AttributeError, TypeError, ValueError):
        model_load_seconds = None

    # -- warmup then measured passes ------------------------------------------------------------------------------
    trials: list[dict[str, Any]] = []
    for i in range(spec.warmup_trials):
        trials.append(run_pass(model, spec, work, probe, "WARMUP", i))
    rss_warm = probe.rss() if rss_start is not None and probe.rss else None
    for i in range(spec.repeats):
        trials.append(run_pass(model, spec, work, probe, "MEASURED", i))
    rss_end = probe.rss() if rss_start is not None and probe.rss else None
    procedure_seconds = probe.clock() - started

    # -- documents ---------------------------------------------------------------------------------------------------
    stats = statistics_doc(spec, trials, probe)
    env = environment_doc(ctx, probe)
    fingerprint = content_hash(
        {
            "analysis_version": ANALYSIS_VERSION,
            "spec_id": spec.spec_id,
            "model_fingerprint": ctx.inputs.model.fingerprint
            if ctx.inputs and ctx.inputs.model
            else None,
            "dataset_fingerprint": ctx.inputs.dataset.fingerprint
            if ctx.inputs and ctx.inputs.dataset
            else None,
            "split": spec.split,
            "source": {"state": ctx.source.state.value, "commit": ctx.source.commit},
            "environment_id": env["environment_id"],
            "device": env["device"],
            "measurement_backends": env["measurement_backends"],
            "stress_ids": list(spec.stress_ids),
            "workload_digest": work.sample_digest,
        }
    )
    measured = [x for x in trials if x["phase"] == "MEASURED"]
    used = [x for x in measured if x["status"] == "COMPLETED"]
    digests = {x["outputs_digest"] for x in used}
    sizes = work.sizes
    enough = len(used) >= spec.measurement.min_trials
    high_water = [x for x in (rss_start, rss_init, rss_warm, rss_end) if x is not None]
    memory = (
        {
            "status": "MEASURED",
            "backend": probe.describe()["memory"],
            "high_water_at_procedure_start_bytes": rss_start,
            "high_water_after_initialization_bytes": rss_init,
            "high_water_after_warmup_bytes": rss_warm,
            "high_water_after_measured_bytes": rss_end,
            "peak_bytes": max(high_water),
            "growth_during_measured_bytes": None
            if rss_warm is None or rss_end is None
            else rss_end - rss_warm,
            "baseline_note": "the executor loaded the model and dataset BEFORE this procedure, so the starting high-water mark already includes them; the model-load memory delta is UNAVAILABLE here",
            "high_water_note": "peak RSS is a process high-water mark: it never decreases, and zero growth means the earlier peak was not exceeded, not that no memory was used",
            "gpu": "UNAVAILABLE: no supported GPU memory backend",
        }
        if high_water
        else {
            "status": "UNAVAILABLE",
            "reason": "process memory accounting was not requested or is unavailable on this platform",
            "gpu": "UNAVAILABLE: no supported GPU memory backend",
        }
    )
    cpu_ok = spec.measurement.cpu and probe.cpu is not None
    summary: dict[str, Any] = _clean(
        {
            "resource_version": ANALYSIS_VERSION,
            "spec_id": spec.spec_id,
            "provenance_fingerprint": fingerprint,
            "model": {
                "model_id": spec.model_id,
                "fingerprint": ctx.inputs.model.fingerprint
                if ctx.inputs and ctx.inputs.model
                else None,
                "adapter": model.NAME,
            },
            "dataset": {
                "dataset_id": spec.dataset_id,
                "fingerprint": ctx.inputs.dataset.fingerprint
                if ctx.inputs and ctx.inputs.dataset
                else None,
            },
            "split": spec.split,
            "workload": {
                "operation": spec.operation,
                "samples": work.n_samples,
                "requested_samples": spec.n_samples,
                "batches": len(work.batches),
                "sample_digest": work.sample_digest,
                "subset": subset_record,
            },
            "batch_size": {
                "requested": spec.batch_size,
                "effective_min": min(sizes),
                "effective_max": max(sizes),
                "effective_mean": sum(sizes) / len(sizes),
                "final_batch": sizes[-1],
            },
            "concurrency": {
                "workers": spec.workers,
                "status": "MEASURED" if spec.workers > 1 else "NOT_REQUESTED",
                "adapter_declared_thread_safe": getattr(base_model, "THREAD_SAFE_INFERENCE", False)
                is True,
                "trial_failures": sum(x["status"] == "FAILED" for x in trials),
                "note": "worker count is a configuration; latency and throughput under contention are measured on this machine and thread scheduler",
            },
            "trials": stats["trials"],
            "initialization": {
                "model_load_seconds": model_load_seconds,
                "model_load_note": "measured by the adapter when the executor loaded the model, before this procedure",
                "stress_build_seconds": stress_seconds if spec.stress else None,
                "data_materialization_seconds": prepare_seconds,
            },
            "warmup": {
                "trials": len([x for x in trials if x["phase"] == "WARMUP"]),
                "seconds": sum(x["wall_seconds"] for x in trials if x["phase"] == "WARMUP"),
                "excluded_from_steady_state": True,
            },
            "steady_state": {
                "status": "MEASURED" if used else "UNAVAILABLE",
                "interval_status": "MEASURED" if enough else "INSUFFICIENT_EVIDENCE",
                "trial_seconds": stats["trial_seconds"],
                "throughput_samples_per_second": stats["throughput_samples_per_second"],
                "batch_seconds": stats["batch_seconds"],
                "per_sample_seconds": stats["per_sample_seconds"],
            },
            "end_to_end": {
                "procedure_seconds": procedure_seconds,
                "measured_seconds": sum(x["wall_seconds"] for x in measured),
                "note": "initialization, warmup and steady-state are separate quantities; procedure_seconds spans all of them",
            },
            "cpu": {
                "status": "MEASURED" if cpu_ok and stats["cpu_seconds"]["n"] else "UNAVAILABLE",
                "cpu_seconds": stats["cpu_seconds"],
                "utilization": stats["cpu_utilization"],
                "reason": None if cpu_ok else "process CPU accounting not requested or unavailable",
            },
            "memory": memory,
            "failures": {
                "failed": sum(x["status"] == "FAILED" for x in measured),
                "timed_out": sum(x["status"] == "TIMED_OUT" for x in measured),
                "failed_samples": sum(x["failed_samples"] for x in measured),
                "completed_samples": sum(x["completed_samples"] for x in measured),
                "timeout_seconds": spec.timeout_seconds,
                "timeout_semantics": "cooperative: checked as each batch completes; an inference call in flight is not interrupted",
            },
            "outputs": {
                "consistent_across_trials": len(digests) <= 1,
                "distinct_digests": len(digests),
                "digest": next(iter(digests)) if len(digests) == 1 else None,
                "note": "digest of the per-sample model outputs of each completed trial, independent of how they were batched; a mismatch means the model was not deterministic under these conditions",
            },
            "stress": None
            if not spec.stress
            else {
                "stress_ids": list(spec.stress_ids),
                "build": stress_record,
                "note": "stress identity is recorded separately from the resource identity; a timing difference under stress is an observation, not a causal claim",
            },
            "references": {
                "calibration_analyses": list(spec.calibration_analyses),
                "stress_analyses": list(spec.stress_analyses),
                "quality_analyses": list(spec.quality_analyses),
                "note": "referenced as execution context; nothing referenced is recomputed",
            },
            "environment": {
                "environment_id": env["environment_id"],
                "device": env["device"],
                "measurement_backends": env["measurement_backends"],
                "environment_note": spec.environment_note,
            },
            "evidence_status": "MEASURED" if used and enough else "INSUFFICIENT_EVIDENCE",
            "methodology": METHODOLOGY,
            "limitations": list(LIMITATIONS),
        }
    )
    partial = len(used) != spec.repeats or not enough

    observations = [
        {"name": n, "trial_index": x["trial_index"], "value": x[k], "unit": u}
        for x in measured
        for n, k, u in (
            ("resources.trial_wall_seconds", "wall_seconds", "s"),
            ("resources.trial_completed_samples", "completed_samples", None),
        )
    ]
    spec_doc = {
        "spec_id": spec.spec_id,
        "spec": spec.to_dict(),
        "provenance_fingerprint": fingerprint,
        "model": summary["model"],
        "dataset": summary["dataset"],
        "split": spec.split,
        "source_revision": to_jsonable(ctx.source),
        "environment_id": env["environment_id"],
        "seed": spec.seed,
        "stress_ids": list(spec.stress_ids),
        "workload_digest": work.sample_digest,
        "analysis_version": ANALYSIS_VERSION,
    }
    art = {
        "spec": _write(ctx, "spec", spec_doc),
        "environment": _write(ctx, "environment", env),
        "trials": _write(
            ctx, "trials", {"trials": trials, "warmup_excluded_from_statistics": True}
        ),
        "observations": _write(ctx, "observations", {"observations": observations}),
        "statistics": _write(ctx, "statistics", stats),
    }
    summary["artifacts"] = art
    art["summary"] = _write(ctx, "summary", summary)
    analysis = ResourceAnalysis(
        req.investigation_id,
        ctx.run.id,
        spec.spec_id,
        spec.to_dict(),
        spec.model_id,
        spec.dataset_id,
        fingerprint,
        "PARTIAL" if partial else "COMPLETE",
        {**summary, "artifacts": art},
        now,
    )
    with reg.transaction():
        reg.add(analysis)
        for x in trials:
            reg.add(
                ResourceTrial(
                    analysis.id,
                    x["phase"],
                    x["trial_index"],
                    x["status"],
                    x["planned_samples"],
                    x["completed_samples"],
                    x["failed_samples"],
                    x["wall_seconds"],
                    x["cpu_seconds"],
                    now,
                )
            )
    for x in measured:
        ctx.observe(
            "resources.trial_wall_seconds",
            x["wall_seconds"],
            unit="s",
            kind=EpistemicKind.OBSERVATION,
        )
    ctx.observe("resources.measured_trials", len(measured), kind=EpistemicKind.OBSERVATION)
    ctx.observe("resources.completed_trials", len(used), kind=EpistemicKind.OBSERVATION)


# -- entry points -----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ResourceRunResult:
    experiment_id: str
    run_id: str
    status: RunStatus
    analysis_id: str | None


def run_resource_request(
    registry: Registry,
    store: ArtifactStore,
    executor: Executor,
    investigation_id: str,
    spec: ResourceSpec,
) -> ResourceRunResult:
    """Validate everything, then measure as a new Run. A refused request creates nothing."""
    preflight(registry, store, executor.adapters, executor.inputs_root, spec)
    rec_m, rec_d = (
        registry.get(RegisteredModel, spec.model_id),
        registry.get(RegisteredDataset, spec.dataset_id),
    )
    conf = ConfigurationRef(ResourceRequest(investigation_id, spec.to_dict()).to_parameters())
    if not registry.exists(ConfigurationRef, conf.id):
        registry.add(conf)
    exp = Experiment(
        investigation_id,
        "resource measurement",
        "Measured latency, throughput and resource use of one workload (environment-specific)",
        rec_m.ref(),
        rec_d.ref(),
        conf.id,
        datetime.now(UTC),
    )
    if not registry.exists(Experiment, exp.id):
        registry.add(exp)
        registry.update_status(exp.with_status(ExperimentStatus.READY))
    result = executor.execute(
        exp.id,
        resolve_procedure(PROCEDURE),
        seed=spec.seed,
        procedure_name=PROCEDURE,
        resources=ResourceLimits(max_workers=spec.workers, timeout_seconds=spec.timeout_seconds),
        device=DeviceKind(spec.device),
    )
    found = registry.find(ResourceAnalysis, run_id=result.run.id)
    return ResourceRunResult(exp.id, result.run.id, result.status, found[0].id if found else None)


def replay_check(
    reg: Registry, store: ArtifactStore, executor: Executor, analysis_id: str
) -> dict[str, Any]:
    """Replay as a NEW run and compare the DEFINITION (spec.json) and the model OUTPUT digests. Timing
    is never expected to be equal; its difference is reported for information only."""
    a = reg.get(ResourceAnalysis, analysis_id)
    replay = executor.replay(a.run_id)
    out: dict[str, object] = {
        "analysis_id": a.id,
        "original_run": a.run_id,
        "replay_run": replay.run.id,
        "replay_status": replay.status.value,
        "timing": "NOT_EXPECTED_TO_REPRODUCE",
    }
    if replay.status is not RunStatus.COMPLETED:
        return {
            **out,
            "definition_reproduced": False,
            "differences": [f"replay run ended {replay.status.value}"],
        }
    diffs: list[str] = []
    for name in DEFINITION_DOCUMENTS:
        try:
            x = read_artifact(reg, store, a.run_id, f"{ARTIFACT_DIR}/{name}.json")
        except ArtifactIntegrityError:
            raise
        except ExperionyxError:
            diffs.append(f"{name}: missing from the original run")
            continue
        _compare(
            name, x, read_artifact(reg, store, replay.run.id, f"{ARTIFACT_DIR}/{name}.json"), diffs
        )
    new = reg.find(ResourceAnalysis, run_id=replay.run.id)
    o_sum = _clean(a.summary)
    n_sum = _clean(new[0].summary) if new else {}
    o_out, n_out = o_sum.get("outputs", {}), n_sum.get("outputs", {})
    same_outputs = o_out.get("digest") == n_out.get("digest") and o_out.get(
        "consistent_across_trials"
    ) == n_out.get("consistent_across_trials")
    if not same_outputs:
        diffs.append(f"outputs: {o_out.get('digest')!r} vs {n_out.get('digest')!r}")

    def med(x: Mapping[str, Any]) -> Any:
        return (x.get("steady_state", {}).get("trial_seconds") or {}).get("median")

    return {
        **out,
        "definition_reproduced": not diffs,
        "outputs_reproduced": same_outputs,
        "differences": diffs[:50],
        "compared": [*DEFINITION_DOCUMENTS, "outputs"],
        "replay_analysis": new[0].id if new else None,
        "informational_median_trial_seconds": {"original": med(o_sum), "replay": med(n_sum)},
    }


__all__ = [
    "PROCEDURE",
    "ResourceError",
    "ResourceRunResult",
    "ResourceUnavailable",
    "capability_report",
    "materialize",
    "preflight",
    "replay_check",
    "run_pass",
    "run_resource_analysis",
    "run_resource_request",
    "statistics_doc",
]
