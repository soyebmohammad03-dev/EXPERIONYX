"""Framework-neutral batching: contiguous, deterministic, bounded-memory index ranges."""

from collections.abc import Iterator

from experionyx.errors import InferenceError


def batch_slices(total: int, batch_size: int) -> Iterator[slice]:
    """Consecutive slices of at most `batch_size` covering range(total); the last may be short."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise InferenceError(f"batch_size must be a positive integer, got {batch_size!r}")
    for start in range(0, total, batch_size):
        yield slice(start, min(start + batch_size, total))
