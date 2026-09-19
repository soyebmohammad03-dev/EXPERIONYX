# Development

**Prerequisites:** Python 3.11+, git. (`uv` optional.)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

| Task | Command |
|---|---|
| Test | `pytest` |
| Lint | `ruff check .` |
| Format | `ruff format .` (check: `ruff format --check .`) |
| Type check | `mypy` |

**Workflow:** branch from `main`, keep changes scoped to the current phase, run all four
commands before committing.

**Phase workflow:** each major phase produces (1) implementation, (2) tests, (3) documentation,
(4) validation, (5) one Git commit.
