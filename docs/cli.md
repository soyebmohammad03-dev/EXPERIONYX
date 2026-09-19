# CLI

Every command operates on a real workspace registry (`--workspace`, default `./.experionyx`).
Read commands never create a workspace (`init` and `demo` do). Relative `source` paths of
registered models/datasets resolve against the workspace; `builtin:<name>` is a dataset bundled
with scikit-learn. Errors print `error: …` to stderr and exit 2.

| Command | Purpose |
|---|---|
| `experionyx info` | package, Python, platform, registry schema version |
| `experionyx status` | entity counts and runs by state |
| `experionyx experiment <experiment-id>` | experiment, configuration and its runs (JSON) |
| `experionyx run <run-id>` | run, outcome, observations, artifacts (JSON) |
| `experionyx provenance <run-id>` | provenance record and fingerprint (JSON) |
| `experionyx verify <run-id>` | re-hash artifacts; exit 1 on any mismatch |
| `experionyx execute <experiment-id> --procedure module:function --seed N` | execute as a new run; exit 1 if it fails |
| `experionyx replay <run-id> [--procedure module:function]` | request a replay as a NEW run |
| `experionyx recover` | close RUNNING runs whose process has ended |
| `experionyx init` | create an empty workspace registry |
| `experionyx adapters list` | model/dataset adapters, versions, capabilities, availability |
| `experionyx adapters inspect <name>` | details of one adapter (model and dataset side) |
| `experionyx model inspect <mdl_id \| path> [--adapter A --version V --device D --option k=v]` | show a registered model, or load a file and show its metadata |
| `experionyx model register --name N --source S --adapter A [--version V --option k=v]` | load, fingerprint and register a model |
| `experionyx dataset inspect <dst_id \| source> [--adapter A --deep]` | show a registered dataset, or load a source (`--deep` computes expensive statistics) |
| `experionyx dataset register --name N --source S --adapter A` | load, fingerprint and register a dataset |
| `experionyx demo {sklearn-classification,sklearn-regression,torch-classification}` | train a tiny model, register it and its dataset, and execute a real adapter-backed experiment |

`-v` logs lifecycle events. `--procedure` is imported with the current directory on `sys.path`
and executes arbitrary code: only run procedures you trust. Experiments are registered through
the Python API (see `examples/basic_experiment.py`); there is no registration command yet.

```bash
python examples/basic_experiment.py            # registers two demo experiments
experionyx status
experionyx execute <experiment-id> --seed 1 --procedure examples.basic_experiment:sample_mean
experionyx provenance <run-id>
experionyx replay <run-id>
```
