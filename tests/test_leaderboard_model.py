"""Leaderboard entity identity, validation and round-trip -- no registry, no execution. Protocol
HASH behavior (model-independence, sensitivity to dataset/faults/seeds) needs real `expand()`
validation against a registered model/dataset and is covered in test_leaderboard_e2e.py."""

from collections.abc import Mapping
from datetime import UTC, datetime

import pytest

from experionyx.errors import ValidationError
from experionyx.leaderboard.entities import (
    BenchmarkProtocol,
    BenchmarkSubmission,
    LeaderboardEntry,
    LeaderboardSnapshot,
)
from experionyx.leaderboard.taxonomy import ReproducibilityState

MDL_A = "mdl_" + "1" * 32
MDL_B = "mdl_" + "2" * 32
DST = "dst_" + "1" * 32
FAKE_HASH = "sha256:" + "cd" * 32
T0 = datetime(2026, 9, 24, tzinfo=UTC)


def test_benchmark_protocol_identity_is_the_protocol_hash() -> None:
    a = BenchmarkProtocol("p", "1.0.0", FAKE_HASH, DST, "1.0.0", T0)
    b = BenchmarkProtocol("different-name", "9.9.9", FAKE_HASH, DST, "1.0.0", T0)
    assert a.id == b.id  # only the hash matters for identity


def test_benchmark_protocol_identity_changes_with_hash() -> None:
    a = BenchmarkProtocol("p", "1.0.0", FAKE_HASH, DST, "1.0.0", T0)
    b = BenchmarkProtocol("p", "1.0.0", "sha256:" + "ef" * 32, DST, "1.0.0", T0)
    assert a.id != b.id


def test_benchmark_protocol_round_trips() -> None:
    p = BenchmarkProtocol("p", "1.0.0", FAKE_HASH, DST, "1.0.0", T0)
    again = BenchmarkProtocol.from_dict(p.to_dict())
    assert again.id == p.id


def test_submission_identity_is_protocol_and_result() -> None:
    a = BenchmarkSubmission("bpr_" + "1" * 32, "inv_" + "1" * 32, MDL_A, "bmk_" + "1" * 32, "brs_" + "1" * 32, T0)  # fmt: skip
    b = BenchmarkSubmission("bpr_" + "1" * 32, "inv_" + "2" * 32, MDL_B, "bmk_" + "2" * 32, "brs_" + "1" * 32, T0)  # fmt: skip
    assert a.id == b.id  # same (protocol, result) -> same submission regardless of everything else  # fmt: skip


def test_snapshot_identity_is_protocol_and_source_fingerprint() -> None:
    digest = "sha256:" + "ab" * 32
    a = LeaderboardSnapshot("bpr_" + "1" * 32, "inv_" + "1" * 32, "1.0.0", (), {}, {}, (), {}, digest, T0)  # fmt: skip
    b = LeaderboardSnapshot("bpr_" + "1" * 32, "inv_" + "2" * 32, "2.0.0", ("m",), {"x": 1}, {}, (), {}, digest, T0)  # fmt: skip
    assert a.id == b.id


def test_snapshot_rejects_non_string_metric_ids() -> None:
    digest = "sha256:" + "ab" * 32
    with pytest.raises(ValidationError):
        LeaderboardSnapshot("bpr_" + "1" * 32, "inv_" + "1" * 32, "1.0.0", (1,), {}, {}, (), {}, digest, T0)  # type: ignore[arg-type]  # fmt: skip


def test_entry_identity_is_snapshot_and_submission() -> None:
    a = LeaderboardEntry("lbs_" + "1" * 32, "bsb_" + "1" * 32, MDL_A, {}, {}, ReproducibilityState.UNAVAILABLE, T0)  # fmt: skip
    b = LeaderboardEntry("lbs_" + "1" * 32, "bsb_" + "1" * 32, MDL_B, {"x": 1}, {"y": 2}, ReproducibilityState.REPRODUCED, T0)  # fmt: skip
    assert a.id == b.id


def test_entry_round_trips() -> None:
    e = LeaderboardEntry("lbs_" + "1" * 32, "bsb_" + "1" * 32, MDL_A, {"accuracy": {"value": 0.9}}, {}, ReproducibilityState.REPRODUCED, T0)  # fmt: skip
    again = LeaderboardEntry.from_dict(e.to_dict())
    assert again.id == e.id
    accuracy = again.metrics["accuracy"]
    assert isinstance(accuracy, Mapping) and accuracy["value"] == 0.9
