import logging
import re

import pytest

import experionyx
from experionyx.cli import main
from experionyx.domain import ClaimStatus, EpistemicKind


def test_version_is_semver_like() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+.*", experionyx.__version__)


def test_library_logger_has_null_handler() -> None:
    handlers = logging.getLogger("experionyx").handlers
    assert any(isinstance(h, logging.NullHandler) for h in handlers)


def test_claim_statuses_match_methodology() -> None:
    assert {s.value for s in ClaimStatus} == {
        "SUPPORTED",
        "NOT_SUPPORTED",
        "INCONCLUSIVE",
        "CONFOUNDED",
        "INSUFFICIENT_EVIDENCE",
    }


def test_epistemic_kinds_are_distinct() -> None:
    assert len({k.value for k in EpistemicKind}) == 4


def test_cli_info_reports_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["info"]) == 0
    assert f"experionyx {experionyx.__version__}" in capsys.readouterr().out


def test_cli_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert experionyx.__version__ in capsys.readouterr().out


def test_optional_frameworks_are_installed_when_required() -> None:
    """CI sets EXPERIONYX_REQUIRE_FRAMEWORKS so ML tests can never skip silently there."""
    import importlib.util
    import os

    if not os.environ.get("EXPERIONYX_REQUIRE_FRAMEWORKS"):
        pytest.skip("EXPERIONYX_REQUIRE_FRAMEWORKS is not set (local run without the extras)")
    missing = [m for m in ("numpy", "sklearn", "torch") if importlib.util.find_spec(m) is None]
    assert not missing, f"optional frameworks required but not installed: {missing}"
