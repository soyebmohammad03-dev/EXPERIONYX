"""Registering a `BenchmarkProtocol`. The model-independent protocol identity is not just
`BenchmarkSpec.protocol_key()` -- it is the FULLY EXPANDED protocol (resolved fault versions,
the expanded unit list, interaction pairs), exactly what `benchmark.protocol.expand` already
computes as `Protocol.protocol_hash`. Registration reuses that expansion directly (the same
validation `benchmark run`/`benchmark validate` perform) so a protocol's identity always matches
what a real submission's `BenchmarkResult.protocol_hash` will be; nothing here executes anything
(see docs/leaderboard.md)."""

from datetime import UTC, datetime

from experionyx.benchmark.protocol import expand
from experionyx.benchmark.spec import ENGINE_VERSION, BenchmarkSpec
from experionyx.errors import ValidationError
from experionyx.faults.library import default_fault_registry
from experionyx.leaderboard.entities import BenchmarkProtocol
from experionyx.registry import Registry


def protocol_hash_of(registry: Registry, spec: BenchmarkSpec) -> str:
    """The exact hash `benchmark.protocol.expand` embeds in every `Protocol`/`BenchmarkResult` it
    builds. Raises `BenchmarkRefusal` with every issue if `spec` itself is invalid -- refused
    before anything is registered, the same as `benchmark validate`."""
    return expand(registry, spec, default_fault_registry()).protocol_hash


def register_protocol(registry: Registry, spec: BenchmarkSpec) -> BenchmarkProtocol:
    """Idempotent: registering the same protocol identity twice returns the existing row."""
    phash = protocol_hash_of(registry, spec)
    existing = registry.find(BenchmarkProtocol, protocol_hash=phash)
    if existing:
        return existing[0]
    protocol = BenchmarkProtocol(
        spec.name, spec.version, phash, spec.dataset, ENGINE_VERSION, datetime.now(UTC)
    )
    registry.add(protocol)
    return protocol


def resolve_protocol(registry: Registry, ref: str) -> BenchmarkProtocol:
    if ref.startswith(BenchmarkProtocol.PREFIX + "_"):
        return registry.get(BenchmarkProtocol, ref)
    matches = [
        p
        for p in registry.find(BenchmarkProtocol)
        if p.protocol_hash.startswith(ref) or p.id.startswith(ref)
    ]
    if len(matches) != 1:
        raise ValidationError(f"{'no' if not matches else 'ambiguous'} protocol matches {ref!r}")
    return matches[0]


__all__ = ["protocol_hash_of", "register_protocol", "resolve_protocol"]
