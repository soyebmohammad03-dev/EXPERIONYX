"""A tiny end-to-end demonstration of registering and executing an experiment.

    python examples/basic_experiment.py            # register two experiments in ./.experionyx
    experionyx status
    experionyx execute <experiment-id> --seed 1 --procedure examples.basic_experiment:sample_mean

The statistic is a toy (mean of seeded uniform samples); it demonstrates provenance capture, not ML.
"""

import json
import random
import sys
from datetime import UTC, datetime
from pathlib import Path

from experionyx.domain import (
    ConfigurationRef,
    DatasetRef,
    Experiment,
    ExperimentStatus,
    Investigation,
    ModelRef,
)
from experionyx.execution import RunContext
from experionyx.sqlite import SqliteRegistry


def sample_mean(ctx: RunContext) -> None:
    n = int(str(ctx.parameters["n_samples"]))
    rng = random.Random(ctx.seed)
    samples = [rng.random() for _ in range(n)]
    ctx.observe("sample_mean", sum(samples) / n, unit="ratio")
    ctx.observe("n_samples", n)
    (ctx.artifact_dir / "samples.json").write_text(json.dumps(samples), encoding="utf-8")
    ctx.register_artifact("samples.json", name="samples")


def always_fails(ctx: RunContext) -> None:
    ctx.observe("started", True)
    raise RuntimeError("deliberate failure for demonstration")


def register(workspace: Path) -> list[str]:
    workspace.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)
    inv = Investigation("demo", "Does the execution engine record what it runs?", now)
    cfg = ConfigurationRef({"n_samples": 1000})
    ids = []
    with SqliteRegistry(workspace / "registry.sqlite") as reg:
        for e in (inv, cfg):
            if not reg.exists(type(e), e.id):
                reg.add(e)
        for name in ("sampler", "always-fails"):
            exp = Experiment(
                inv.id,
                name,
                f"demo experiment {name}",
                ModelRef("toy-sampler", "0.1"),
                DatasetRef("synthetic", "1"),
                cfg.id,
                now,
            )
            if not reg.exists(Experiment, exp.id):
                reg.add(exp)
                reg.update_status(exp.with_status(ExperimentStatus.READY))
            ids.append(f"{name}: {exp.id}")
    return ids


if __name__ == "__main__":
    for line in register(Path(sys.argv[1] if len(sys.argv) > 1 else ".experionyx")):
        print(line)
