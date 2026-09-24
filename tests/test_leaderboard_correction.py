"""Multiple-comparison correction: `leaderboard.compare.correct_family` reuses Phase 10's
`stats.store` CORRECTION kind end to end, over an explicitly named family of already-registered
COMPARE analyses -- never fabricating a p-value itself (see docs/leaderboard.md,
docs/statistics.md)."""

from collections.abc import Mapping

import pytest

from conftest import Lab
from experionyx.errors import NotFoundError, ValidationError
from experionyx.leaderboard.compare import CORRECTION_METHODS, correct_family
from experionyx.stats import store as stats_store


def _compare(lab: Lab, reference: list[float], treatment: list[float]) -> str:
    analysis, _new = stats_store.create(
        lab.registry, lab.store, "COMPARE",
        {"kind": "inline", "reference": reference, "treatment": treatment},
        {"pairing": "UNPAIRED", "permutations": 500, "resamples": 500},
    )  # fmt: skip
    return analysis.id


def test_correct_family_with_none_leaves_pvalues_unadjusted(lab: Lab) -> None:
    a = _compare(lab, [1.0, 2.0, 3.0, 4.0, 5.0], [1.1, 2.1, 3.1, 4.1, 5.1])
    b = _compare(lab, [1.0, 2.0, 3.0], [5.0, 6.0, 7.0])
    result = correct_family(lab.registry, (a, b), method="NONE")
    assert result.config["method"] == "NONE"
    assert result.result["controls"] == "nothing (no correction applied)"


def test_correct_family_bonferroni_widens_pvalues(lab: Lab) -> None:
    a = _compare(lab, [1.0, 2.0, 3.0, 4.0, 5.0], [1.1, 2.1, 3.1, 4.1, 5.1])
    b = _compare(lab, [1.0, 2.0, 3.0], [5.0, 6.0, 7.0])
    none = correct_family(lab.registry, (a, b), method="NONE")
    corrected = correct_family(lab.registry, (a, b), method="BONFERRONI")
    none_adjusted = none.result["adjusted"]
    corrected_adjusted = corrected.result["adjusted"]
    assert isinstance(none_adjusted, Mapping) and isinstance(corrected_adjusted, Mapping)
    for key, adj in corrected_adjusted.items():
        assert adj >= none_adjusted[key]


def test_correct_family_is_idempotent(lab: Lab) -> None:
    a = _compare(lab, [1.0, 2.0, 3.0], [1.5, 2.5, 3.5])
    first = correct_family(lab.registry, (a,), method="BENJAMINI_HOCHBERG")
    second = correct_family(lab.registry, (a,), method="BENJAMINI_HOCHBERG")
    assert first.id == second.id


def test_correct_family_rejects_an_unknown_method(lab: Lab) -> None:
    a = _compare(lab, [1.0, 2.0], [3.0, 4.0])
    with pytest.raises(ValidationError):
        correct_family(lab.registry, (a,), method="HOLM")


def test_correct_family_requires_at_least_one_analysis(lab: Lab) -> None:
    with pytest.raises(ValidationError):
        correct_family(lab.registry, ())


def test_correct_family_rejects_an_unknown_analysis_id(lab: Lab) -> None:
    with pytest.raises(NotFoundError):
        correct_family(lab.registry, ("sta_" + "9" * 32,))


def test_correction_methods_match_stats_core() -> None:
    assert set(CORRECTION_METHODS) == {"NONE", "BONFERRONI", "BENJAMINI_HOCHBERG"}
