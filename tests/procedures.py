"""Real procedures executed by the engine in tests (importable as `procedures:<name>`)."""

import random

from experionyx.execution import RunContext


def ok(ctx: RunContext) -> None:
    ctx.observe("score", ctx.seed * 0.5, unit="ratio")
    ctx.observe("draw", random.random())  # seeded by the executor
    ctx.observe("curve", {"xs": [1, 2, 3]})
    (ctx.artifact_dir / "result.txt").write_text(f"seed={ctx.seed}\n", encoding="utf-8")
    ctx.register_artifact("result.txt", name="result")


def fails(ctx: RunContext) -> None:
    ctx.observe("before_failure", 1)
    (ctx.artifact_dir / "partial.txt").write_text("partial", encoding="utf-8")
    ctx.register_artifact("partial.txt")
    raise ValueError("boom")


def registers_missing_file(ctx: RunContext) -> None:
    ctx.register_artifact("does-not-exist.txt")


def deletes_after_registering(ctx: RunContext) -> None:
    path = ctx.artifact_dir / "gone.txt"
    path.write_text("x", encoding="utf-8")
    ctx.register_artifact("gone.txt")
    path.unlink()


def modifies_after_registering(ctx: RunContext) -> None:
    path = ctx.artifact_dir / "changed.txt"
    path.write_text("original", encoding="utf-8")
    ctx.register_artifact("changed.txt")
    path.write_text("tampered", encoding="utf-8")


def interrupted(ctx: RunContext) -> None:
    ctx.observe("before_interrupt", 1)
    raise KeyboardInterrupt


def nothing(ctx: RunContext) -> None:
    return None


def adapter_eval(ctx: RunContext) -> None:
    """Uses whatever registered model/dataset the executor bound to the run."""
    assert ctx.model is not None
    assert ctx.dataset is not None
    predictions: list[object] = []
    for batch in ctx.dataset.batches(4, "test"):
        result = ctx.model.predict(batch.inputs, sample_ids=batch.indices)
        predictions.extend(result.outputs)
        ctx.observe("batch_seconds", result.inference_seconds, unit="s")
    ctx.observe("n_predictions", len(predictions))
    (ctx.artifact_dir / "predictions.txt").write_text(repr(predictions), encoding="utf-8")
    ctx.register_artifact("predictions.txt", name="predictions")
