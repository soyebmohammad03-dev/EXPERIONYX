"""Mechanical enforcement: no API router may mutate the registry. A router calling `.add(` or
`.update_status(` on anything would make this a second write path into the domain, not a viewer."""

import re
from pathlib import Path

import experionyx

ROUTERS_DIR = Path(experionyx.__file__).parent / "api" / "routers"
_FORBIDDEN = re.compile(r"\.(add|update_status)\(")


def test_no_router_source_calls_add_or_update_status() -> None:
    offenders = []
    for path in sorted(ROUTERS_DIR.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        if _FORBIDDEN.search(text):
            offenders.append(path.name)
    assert not offenders, f"read-only invariant violated in: {offenders}"


def test_routers_directory_is_not_empty() -> None:
    assert list(ROUTERS_DIR.glob("*.py"))
