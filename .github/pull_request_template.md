**Summary**

What changed and why.

**Implementation**

Key files/modules touched and the approach taken.

**Tests**

- [ ] `pytest` passes locally
- [ ] `ruff check .` and `ruff format --check .` pass
- [ ] `mypy` passes (strict)
- [ ] New behavior has a corresponding test

**Scientific impact**

Does this change affect what a claim, statistic, reliability dimension, or benchmark comparison
means? If so, explain — and confirm it does not introduce a composite score, a causal claim, or
treat missing evidence as negative evidence.

**Provenance impact**

Does this change what gets recorded about environment, seed, configuration, or artifacts? If so,
does the schema/migration need to change?

**Reproducibility**

Does this affect `replay`, `reproduce`, or determinism of any engine's output?

**Documentation**

- [ ] `docs/` updated for behavior changes
- [ ] README updated if user-facing behavior changed

**Breaking changes**

None / describe them, including any schema version bump.
