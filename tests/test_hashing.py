import pytest

from experionyx.hashing import canonical_json, content_hash


def test_known_vector() -> None:
    assert content_hash({}) == (
        "sha256:44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
    )


def test_key_order_does_not_matter() -> None:
    assert content_hash({"a": 1, "b": [1, 2]}) == content_hash({"b": [1, 2], "a": 1})
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'


def test_different_content_different_hash() -> None:
    assert content_hash({"a": 1}) != content_hash({"a": 2})
    assert content_hash({"a": [1, 2]}) != content_hash({"a": [2, 1]})


def test_int_and_float_are_distinct_content() -> None:
    assert content_hash({"a": 1}) != content_hash({"a": 1.0})


def test_non_finite_rejected() -> None:
    with pytest.raises(ValueError, match="Out of range"):
        content_hash({"a": float("nan")})
