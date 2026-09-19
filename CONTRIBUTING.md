# Contributing

- **Setup:** see [docs/development.md](docs/development.md).
- **Branches:** short-lived branches off `main`; one concern per PR.
- **Checks:** `pytest`, `ruff check .`, `ruff format --check .`, `mypy` must pass.
- **Types:** strict mypy; avoid `Any`, blanket ignores and silent exception handling.
- **Docs:** update `docs/` and README with behavior changes; never describe unbuilt features as
  existing.
- **Reproducibility:** experiments must record seed, configuration and environment.
- **Evidence:** claims in code, docs or PRs must link to real, recorded observations.
- **No fabrication:** never invent results, benchmarks, datasets or significance.
- **PRs:** state what changed, why, how it was validated, and known limitations.
- Do not commit datasets, model weights, secrets or generated outputs.
