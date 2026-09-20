# Model stress laboratory (Phase 14)

How does this model behave when a controlled aspect of its inputs, parameters, decision rule or
evaluation conditions is deliberately stressed? The stress laboratory answers with **observed changes
against an explicitly validated baseline**, each trial a real run with full provenance. It does not
produce a robustness or stress score, rank models, promise that an untested stress is safe, or explain
why a change happened.

## Stress vs fault

| | Fault Laboratory (Phase 5) | Stress Laboratory (Phase 14) |
|---|---|---|
| Question | how does measured behaviour change under controlled **data/system faults** (noise, missing values, corruption, label noise)? | how does a **model** behave when inputs, parameters, its decision rule or the evaluation condition are stressed? |
| Applies to | the dataset (a faulted view; the registered data is never modified) | the input view, a derived copy of the model, its decision rule, or the evaluation condition |
| Implementation | fault library, faulted evaluation procedure | **input stress reuses the fault library and procedure**; model-level stress is new |

When a stress is semantically an input fault (`FEATURE_SCALING`, `FEATURE_OFFSET`, `FEATURE_NOISE`,
`IMAGE_NOISE`, `BRIGHTNESS`, `CONTRAST`, `OCCLUSION`, `BLUR`) the stress spec maps to exactly the Fault
Laboratory's spec, the trials run through `run_fault_experiment` (the same runs, `FaultExperiment` and
`FaultTrial` records), and every trial's evidence names its origin: `FAULT_LABORATORY`, the fault ID and
the fault procedure. Nothing is implemented twice, and failure discovery, interaction analysis and the
fault-slice integration see these runs unchanged.

## Families and the capabilities they need

| Family | Origin | What it does | Needs |
|---|---|---|---|
| `FEATURE_SCALING`, `FEATURE_OFFSET`, `FEATURE_NOISE` | fault | multiply / shift / add Gaussian noise to inputs | dataset the fault library accepts (tabular) |
| `IMAGE_NOISE`, `BRIGHTNESS`, `CONTRAST`, `OCCLUSION`, `BLUR` | fault | salt-and-pepper noise, brightness, contrast, occlusion, box blur | image tensors |
| `PARAMETER_NOISE`, `PARAMETER_SCALE` | model | seeded Gaussian noise (relative to each array's RMS) / scaling of safely accessible parameters | the adapter's optional `ParameterAccess` |
| `THRESHOLD` | model | classify by comparing the positive-class probability with a threshold | a binary classifier with predicted probabilities |
| `BATCH_SIZE` | evaluation | evaluate with another batch size | any model that predicts |
| `INPUT_SHAPE` | evaluation | feed another declared shape | an adapter that declares `SUPPORTED_INPUT_SHAPES` (none do today) |
| `REPEATED_EXECUTION` | evaluation | evaluate the unmodified model N times | any model that predicts |

An unsupported request is reported **UNAVAILABLE with the reason and refused before anything runs**; it
is never approximated. `ParameterAccess` (an optional adapter protocol) promises: `parameter_arrays()`
returns *copies* of a documented, safe subset of parameters, and `with_parameters()` returns a **new**
adapter built from a private copy of the model. The scikit-learn adapter exposes `coef_` and
`intercept_` for estimators that have them as float arrays (nothing for trees or forests); the PyTorch
adapter exposes floating-point state-dict entries. The registered model is never touched: restoration is
by construction and each parameter trial records the original's digest before and after (`restored`).
A threshold is only defined for binary classifiers with probabilities: an iris three-class model, or a
logits-only network, is refused rather than given a made-up threshold.

**Deterministic vs stochastic.** Seeded stress (`PARAMETER_NOISE`, noise/occlusion faults, scaling or
offset with `fraction < 1`) derives randomness only from the recorded seed. Deterministic stress
(`PARAMETER_SCALE`, `THRESHOLD`, `BATCH_SIZE`, `BRIGHTNESS`, …) takes **no seed**, and several seeds for
it are refused, because they would repeat identical trials. `REPEATED_EXECUTION` observes repeat-to-repeat
behaviour and reports `DETERMINISTIC` only if every predictions digest and metric value is identical;
otherwise `NONDETERMINISTIC`. Randomness is never manufactured.

## Plans and trials

A `StressPlan` is one or more ordered stresses plus an optional sweep and seeds:

- **single** point, **sweep** (one numeric parameter over explicit values), **repeated** (seeds),
- **compound** (ORDER IS IDENTITY: A then B ≠ B then A). Input stress may be compounded freely (a fault
  compound); model-level stress only as parameter perturbations followed by a decision threshold;
  mixed origins are refused. A compound cannot be swept.
- **factorial 2x2**: two Fault Laboratory factors expand to cells `A`, `B`, `AB` per seed and feed the
  **existing Phase 7 interaction analysis** (the same engine, the same records). An invalid or
  incomplete design is refused by that engine and recorded in the analysis, never hidden.

`expand()` turns a plan into explicit units (point × repeat × cell), each with a deterministic key
(`sxu_…`), and each unit is its own Run and its own `StressTrial` record. Invalid plans are refused
before any model runs. The analysis records requested, executed, completed, failed, skipped and
unsupported counts and the seeds.

## Baseline

A baseline must be a COMPLETED run of the **same model, dataset, split, evaluation and metric
configuration** as the stressed evaluations (only an evaluation condition the plan deliberately
stresses, the batch size, may differ). An incompatible baseline is refused with the reason; a baseline
is never silently reused. Without one, the analysis creates it from the design's evaluation.

## What is compared, and how

Every difference is **stressed minus baseline**. For each scalar metric both values, both statuses,
sample counts and intervals are kept; `deterioration` is direction-aware (positive when the stressed
value is worse: `baseline − stressed` if higher is better, `stressed − baseline` otherwise; none when a
metric has no direction). Relative change is undefined, with the reason, when the baseline is 0.

The per-sample measure of the task (accuracy or absolute error) is compared with Phase 10: **PAIRED**
when both runs evaluated the same samples with the same ground truth (a per-sample difference, exact
sign-flip/permutation test, bootstrap interval, effect sizes), otherwise **UNPAIRED** with the reason;
non-finite values are counted and excluded, never averaged in. Repeats of a point are aggregated with
their raw values and a bootstrap interval (one trial has no spread and says so). One family of
hypotheses per design (one test per trial) can be corrected with `NONE`, `BONFERRONI` or
`BENJAMINI_HOCHBERG`; raw and adjusted p-values and the family definition are stored.

Statistical evidence is not practical importance: a tiny, significant change can be irrelevant, and a
large change on a few samples can be insignificant. No result carries a score, a rank or a causal
statement; a change is an observation under the stated stress, model, data and evaluation only.

## Slices, failures, input and data-quality evidence

- **Slices** (Phase 11) only when listed explicitly. Membership comes from the baseline's sample table
  and must be static (a slice reading `predicted`/`correct`/`confidence` is refused because its
  membership would differ between runs). Each trial gets a paired comparison on the members, with the
  slice identity and member count; a slice with no members or fewer than `min_members` is
  `INSUFFICIENT_EVIDENCE`.
- **Failure discovery** (Phase 6) when `discover_failures` is set: model-level trial runs, or the fault
  experiments of input stress, are the sources. Stress identity, trial identity and run links are kept;
  no mode is confirmed.
- **Input evidence**: for input stress on tabular data, the distribution before and after (mean, KS,
  Wasserstein-1, missing/non-finite counts) is recorded. It is labelled `STRESS_INDUCED`: the same
  samples before and after a deliberate, recorded transformation, so no test is applied and it is not
  naturally occurring drift. Model-level stress reports `INPUTS_UNCHANGED`.

## Provenance, replay and artifacts

Every trial records the model and its fingerprint, the dataset, split, the source revision and
environment (its own Provenance record), the exact stress parameters and seed, the baseline run, the
transformation's identity (fault ID or parameter digests and predictions digest), the evaluation
configuration and the digests of its artifacts. The analysis fingerprint changes with any of the design,
model or data fingerprint, evaluation, baseline, trial evidence, slices and versions.

`stress replay` re-executes the analysis **and every completed trial** as new runs and compares all seven
documents, each trial's identity (unit, stress IDs, parameters, seed, fault or build record) and its
metric values within the repository's tolerance (predictions digests where recorded). A document that no
longer matches its digest is refused, not reported as a difference. `stress compare` recomputes the
analysis from the stored trials and diffs it against the stored documents without storing anything.

Artifacts (`stress/`): `spec.json`, `plan.json`, `trials.json` (each trial with its evidence), `baseline.json`
(with the compatibility checks), `results.json`, `analyses.json`, `summary.json`.

## Benchmarks and reliability profiles

A benchmark spec may name one `stress` plan; the benchmark runs it on its own model, dataset, evaluation
and baseline and **references** the analysis (planned, executed, unsupported, failed, insufficient
evidence). A profile may reference stress analyses (`stress_analyses`) as the `MODEL_STRESS` dimension:
`UNAVAILABLE` when none is referenced ("absence of stress evidence is not evidence of robustness"),
`INSUFFICIENT_EVIDENCE` when no trial had enough evidence, `DERIVED` otherwise, copied as recorded, never
merged with another dimension. Benchmarks and profiles without stress keep their earlier identity.

## Records and commands

`StressAnalysis` (`sxa_`) and one `StressTrial` (`sxt_`) per unit; schema v13 (`stress_analyses`,
`stress_trials`), append-only.

```
experionyx stress families                        # families, origin, parameters
experionyx stress validate SPEC [--preflight]     # identity, expanded trials; --preflight checks model, dataset, baseline
experionyx stress run SPEC [--investigation ID]   # single, compound, factorial, repeated execution; exit 3 if evidence is incomplete
experionyx stress sweep SPEC                      # a plan with a sweep or several seeds
experionyx stress list [--model-id ID --status S]
experionyx stress inspect ID [--full]             # sxa_ with provenance and trials, or an sxt_
experionyx stress compare ID                      # recompute from stored trials; nothing stored
experionyx stress replay ID [--no-trials]         # exit 1 on any difference
```

```json
{"model_id": "mdl_…", "dataset_id": "dst_…", "baseline_run": "run_…",
 "evaluation": {"split": "test"},
 "plan": {"components": [{"family": "PARAMETER_NOISE", "parameters": {"relative_sigma": 0.2}}],
          "sweep": {"parameter": "relative_sigma", "values": [0.1, 0.2, 0.5]}, "seeds": [1, 2, 3]},
 "correction": "BENJAMINI_HOCHBERG"}
```

## Limitations

Only stresses an adapter can support safely are available; parameter stress covers linear parameters in
scikit-learn and floating-point state-dict entries in PyTorch (a perturbation of raw weights is not a
realistic deployment condition). A threshold exists only for binary probabilistic classifiers. Shape
stress is refused until an adapter declares shapes. Evaluation is on one split with independent-sample
statistics; a few trials give wide uncertainty; the tests are permutation tests and cost grows with
trials and samples. Results are specific to the stated stress, model, data and evaluation and say nothing
about other stresses or other data. Wall-clock latency is reported but never used as evidence.
