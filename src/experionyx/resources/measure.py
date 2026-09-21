"""Measurement primitives: the probe (timer, process CPU, process memory) and the pure statistics
over raw trial measurements.

The probe is the ONLY place a clock or the operating system is read, so tests can substitute a
deterministic one (`use_probe`) and everything else is pure. A backend the platform lacks is `None`
and its measurement is reported UNAVAILABLE, never estimated.

What the system probe measures, exactly:
  * wall time: `time.perf_counter` (monotonic, the highest-resolution clock Python offers; its
    resolution is read from `time.get_clock_info` and reported with every analysis);
  * CPU: `time.process_time` = user + system CPU seconds of THIS process across all threads; it is
    CPU time, not wall time, and not utilization of the machine;
  * memory: `resource.getrusage(RUSAGE_SELF).ru_maxrss`, the process's peak resident set size so far.
    It is a HIGH-WATER MARK: it can only grow and cannot be reset, so a trial's memory "growth" is how
    far it pushed the process high-water mark, and zero growth means "did not exceed the earlier
    peak", not "used no memory". Python and the allocator may keep freed memory, so it also is not
    the model's own footprint. No GPU memory is measured (no supported backend exists here)."""

import math
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from experionyx.evaluation.bootstrap import quantile
from experionyx.stats import core as st


@dataclass(frozen=True)
class Probe:
    name: str
    clock: Callable[[], float]
    resolution: float  # seconds; the smallest step the clock can report
    cpu: Callable[[], float] | None = None  # cumulative process CPU seconds
    cpu_backend: str | None = None
    rss: Callable[[], int | None] | None = None  # process peak resident bytes (high-water mark)
    rss_backend: str | None = None

    def describe(self) -> dict[str, object]:
        return {
            "probe": self.name,
            "clock": "time.perf_counter" if self.name == "system" else self.name,
            "clock_resolution_seconds": self.resolution,
            "cpu": self.cpu_backend if self.cpu is not None else "UNAVAILABLE",
            "memory": self.rss_backend if self.rss is not None else "UNAVAILABLE",
            "gpu_memory": "UNAVAILABLE: no supported GPU memory backend",
        }


def _system_probe() -> Probe:
    rss: Callable[[], int | None] | None = None
    backend: str | None = None
    try:
        import resource

        scale = 1 if sys.platform == "darwin" else 1024 if sys.platform.startswith("linux") else 0
        if scale:
            backend = f"resource.getrusage(RUSAGE_SELF).ru_maxrss ({'bytes' if scale == 1 else 'KiB'} on {sys.platform}): process peak RSS, a high-water mark"
            rss = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale  # noqa: E731
    except ImportError:  # e.g. Windows: no `resource` module
        pass
    return Probe(
        "system",
        time.perf_counter,
        time.get_clock_info("perf_counter").resolution,
        time.process_time,
        "time.process_time: user+system CPU seconds of this process, all threads",
        rss,
        backend,
    )


_PROBE = _system_probe()


def current_probe() -> Probe:
    return _PROBE


@contextmanager
def use_probe(probe: Probe) -> Iterator[Probe]:
    """Substitute the probe (deterministic clocks and fake or absent backends in tests)."""
    global _PROBE
    previous, _PROBE = _PROBE, probe
    try:
        yield probe
    finally:
        _PROBE = previous


# -- pure statistics over raw measurements ----------------------------------------------------------------------


def _q(x: st.Quantity) -> float | None:
    return x.value


def describe(values: Sequence[float], percentiles: Sequence[float]) -> dict[str, Any]:
    """Descriptive statistics of raw values. Tail percentiles of few observations are close to the
    maximum and are flagged."""
    s = st.summarize(values)
    out: dict[str, Any] = {
        "n": s.n,
        "mean": _q(s.mean),
        "median": _q(s.median),
        "std": _q(s.std),
        "min": _q(s.minimum),
        "max": _q(s.maximum),
        "percentiles": {},
        "warnings": [],
    }
    if not values:
        out["status"] = "UNAVAILABLE"
        return out
    sv = sorted(float(x) for x in values)
    out["status"] = "MEASURED"
    out["percentiles"] = {f"p{p:g}": quantile(sv, p / 100.0) for p in percentiles}
    for p in percentiles:
        if len(sv) * (1.0 - p / 100.0) < 1.0:
            out["warnings"].append(
                f"p{p:g} of {len(sv)} observation(s) is at or near the maximum: fewer than one observation lies above it"
            )
    return out


def lag1_autocorrelation(values: Sequence[float]) -> float | None:
    """Lag-1 autocorrelation in TRIAL ORDER: a diagnostic for drift (thermal, cache, load) that
    would break the independence a bootstrap interval assumes. None if undefined."""
    n = len(values)
    if n < 3:
        return None
    m = math.fsum(values) / n
    den = math.fsum((x - m) ** 2 for x in values)
    if den == 0.0:
        return None
    return math.fsum((values[i] - m) * (values[i + 1] - m) for i in range(n - 1)) / den


def interval(
    values: Sequence[float],
    estimator: str,
    confidence: float,
    resamples: int,
    seed: int,
    min_trials: int,
) -> dict[str, Any]:
    """Bootstrap-percentile interval of the mean or median of trial-level values (Phase 10). Below
    `min_trials` observations the interval is INSUFFICIENT_EVIDENCE and none is reported."""
    if len(values) < min_trials:
        return {
            "status": "INSUFFICIENT_EVIDENCE",
            "estimator": estimator,
            "n": len(values),
            "reason": f"{len(values)} completed measured trial(s); at least {min_trials} are needed for an interval",
        }
    ci = st.bootstrap_interval(
        values,
        estimator=estimator,
        method="percentile",
        confidence=confidence,
        resamples=resamples,
        seed=seed,
    )
    return {
        "status": "MEASURED" if ci.lower is not None else ci.status.value,
        "estimator": estimator,
        "n": len(values),
        "confidence": confidence,
        "lower": ci.lower,
        "upper": ci.upper,
        "estimate": ci.estimate,
        "resamples": ci.resamples,
        "seed": seed,
        "method": ci.method,
        "warnings": list(ci.warnings),
        "reason": ci.reason,
    }


def throughput(
    completed_samples: int, completed_batches: int, seconds: float
) -> dict[str, float | None]:
    """samples/sec and batches/sec from ACTUAL completed work and ACTUAL elapsed time."""
    if seconds <= 0.0 or completed_samples <= 0:
        return {"samples_per_second": None, "batches_per_second": None}
    return {
        "samples_per_second": completed_samples / seconds,
        "batches_per_second": completed_batches / seconds,
    }
