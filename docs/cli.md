# CLI

Every command operates on a real workspace registry (`--workspace`, default `./.experionyx`).
Read commands never create a workspace. Errors print `error: …` to stderr and exit 2.

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
