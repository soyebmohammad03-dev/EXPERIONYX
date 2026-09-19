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
| `experionyx demo {sklearn-classification,sklearn-regression,torch-classification}` | train a tiny model, register it and its dataset, and run a real baseline evaluation |
| `experionyx metrics list [--task T]` | the metric registry: tasks, requirements, scales |
| `experionyx evaluate --model mdl_… --dataset dst_… [--split S --batch-size N --metric M… --score-source X --bins N --bootstrap-resamples N --config FILE --seed N]` | run a baseline evaluation as a real run (a `--config` file is strict: unknown fields fail) |
| `experionyx model autopsy mdl_… --dataset dst_… [same options]` | evaluate, then print the model profile and findings |
| `experionyx evaluation inspect <run> [--full]` | summary (or the full result) of a run's stored evaluation; verifies the artifact digest |
| `experionyx faults` | list fault types (reserved, unimplemented ones are marked) |
| `experionyx fault inspect <name>` | a fault's parameters, requirements and semantics |
| `experionyx fault run --model mdl_… --dataset dst_… --type T --param k=v … --seed N [--scope-fraction F \| --scope-class C] [--spec-file F]` | one seeded fault: baseline control + faulted run + analysis |
| `experionyx fault sweep … --sweep param=v1,v2,… --seeds 1,2,3 [--baseline-run RUN --max-failed-trials N]` | a real parameter sweep with repeated seeds |
| `experionyx fault compare <baseline-run> <treatment-run>` | direction-aware degradation of one treatment against its control |
| `experionyx fault experiment inspect <fxp_…> [--full]` | design, trials and analysis of a fault experiment |
| `experionyx fault demo {tabular,tensor,labels}` | real example fault experiments |
| `experionyx failure discover [--fault-experiment fxp_… …] [--run run_… …] [--investigation inv_…] [--config FILE] [--seed N]` | run failure discovery over stored fault experiments and evaluation runs as a real Run; lists the modes it produced. The home investigation is inferred when all sources share one, otherwise `--investigation` is required. `--config` is a strict `DiscoveryConfig` JSON |
| `experionyx failures [--investigation I --status S --category C]` | list registered failure modes |
| `experionyx failure inspect <fmd_… \| fcl_… \| fsg_…>` | a failure mode, cluster or signal |
| `experionyx failure cluster [fcl_…] [--investigation I]` | list clusters, or one cluster with its member signals |
| `experionyx failure candidates [filters]` | modes not yet confirmed, with their criteria checks (required vs observed) |
| `experionyx failure evidence <fmd_…>` | all retained evidence of a mode |
| `experionyx failure reproduce <fmd_…> [--config FILE]` | replay supporting runs as new runs and compare within tolerances; exit 1 if the check fails; never changes status |
| `experionyx failure confirm <fmd_…> --by WHO --reason WHY` | confirm a `SUPPORTED`, reproduced mode (an explicit decision) |
| `experionyx failure set-status <fmd_…> --to REJECTED\|DEPRECATED\|… --by WHO --reason WHY` | validated lifecycle change (never to `CONFIRMED`) |
| `experionyx failure graph <fmd_…>` | relationships around a mode (backend graph, no visualization) |
| `experionyx interaction validate\|analyze SPEC.json [--format text]` | validate a four-cell design (refused with every issue listed before any run exists), or analyze it as a real Run |
| `experionyx interaction list [--fault --metric --status --class --failure-mode --experiment --dataset --model]` | search registered interactions |
| `experionyx interaction inspect <ian_…> [--full]` | effects, evidence, and (with `--full`) every effect record and raw trial |
| `experionyx interaction failures\|replay\|export\|related <ian_…>` | failure-mode comparison; replay-and-compare (exit 1 on any difference); JSON bundle; structurally similar analyses |
| `experionyx interaction reproduce <ian_…> --replicate <ian_…>` | check an independent replicate; moves `SUPPORTED` to `REPRODUCIBLE` only if it agrees |
| `experionyx interaction confirm\|set-status <ian_…> --by WHO --reason WHY` | explicit human confirmation, or reject/deprecate |
| `experionyx reliability profile SPEC.json [--format text]` | build a reliability profile from stored evidence (refused, with every issue listed, if sources are incompatible) |
| `experionyx reliability list\|inspect\|evidence <rpf_…>` | search profiles; dimension statuses and observations; source references, provenance and artifacts |
| `experionyx reliability compare <rpf_…> <rpf_…>` | raw per-dimension differences between two compatible profiles (no winner) |
| `experionyx reliability replay <rpf_…>` | replay the profile as a new run and compare; exit 1 unless verified deterministic |
| `experionyx benchmark validate\|run SPEC.json [--format text]` | expand and validate a benchmark protocol (refused with every issue listed, before any run exists), or execute it and collect coverage and results (exit 0 complete, 3 incomplete coverage, 2 refused) |
| `experionyx benchmark list\|inspect\|coverage <bmk_…\|brs_…>` | search definitions; a definition or result with its section statuses; the coverage account |
| `experionyx benchmark compare <brs_…> <brs_…>` | raw differences between two results of an identical protocol (refused otherwise; no winner) |
| `experionyx benchmark replay <brs_…>` | replay the collect run as a new run and compare; exit 1 unless verified deterministic |
| `experionyx stats compare\|bootstrap\|proportion\|correct …` | deterministic, recorded statistics over inline values, interaction trials, fault trials or run artifacts; see [statistics.md](statistics.md) |
| `experionyx stats list\|inspect\|verify <sta_…>` | registered analyses; `verify` recomputes from the recorded sources and exits 1 unless reproduced |
| `experionyx evaluation compare <run-a> <run-b>` | structured comparison; no winner, no significance test |

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
