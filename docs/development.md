# Development

**Prerequisites:** Python 3.11+, git. (`uv` optional.)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,sklearn,torch,faults]"
```

| Task | Command |
|---|---|
| Test | `pytest` (ML adapter tests skip themselves if sklearn/torch are missing) |
| Lint | `ruff check .` |
| Format | `ruff format .` (check: `ruff format --check .`) |
| Type check | `mypy` |

**Workflow:** branch from `main`, keep changes scoped to the current phase, run all four
commands before committing.

## Phase workflow

Every completed EXPERIONYX phase receives its own meaningful Git commit. For each phase:

1. Inspect the existing implementation.
2. Implement one substantial capability.
3. Add or update tests.
4. Run validation (`pytest`, `ruff check .`, `ruff format --check .`, `mypy`).
5. Update documentation.
6. Inspect the Git diff.
7. Verify no secrets or large accidental files.
8. Verify repository state (`git status` clean).
9. Create a meaningful commit.
10. Continue to the next phase.

Do not create commits for tiny changes. Use conventional, descriptive messages, e.g.
`feat: implement experiment provenance foundation`, `feat: add model adapter architecture`.
Never use `update`, `changes`, `stuff`, `final` or `new`.

## Public research repository principles

EXPERIONYX is developed with a transparent public history so researchers and engineers can
inspect it.

- Implementation is auditable; commits correspond to meaningful milestones.
- Experimental results must be reproducible; claims must be evidence-backed.
- Documentation must match implementation; unfinished features are clearly marked.
- Do not commit generated artifacts unnecessarily, or datasets/models that are large or
  inappropriate to redistribute.
- Secrets never enter the repository.

## CI
`.github/workflows/ci.yml` runs Ruff (lint and format), strict mypy and pytest on Python 3.11 and
3.12 with scikit-learn and the CPU-only torch wheel. `fail-fast` is off and every check runs even
if an earlier one failed, so one run reports every problem. Tool versions are bounded in the
`dev` extra. `EXPERIONYX_REQUIRE_FRAMEWORKS=1` makes a missing sklearn/torch a test *failure*
instead of a skip. Lessons behind these choices: numpy stubs differ between the versions each
Python resolves (so `# type: ignore` comments must tolerate both), and floating-point results can
differ in the last bits between platforms (so tests compare floats with tolerances).
