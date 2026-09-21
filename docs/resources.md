# Resource & systems reliability laboratory (Phase 16)

The laboratory answers one question: **how does the behaviour of a model/system change when the execution
conditions change?** Execution conditions here are the batch size, the size of the workload, the number of
workers, the model-level stress applied, the requested timeout and the environment the run happens in.

Everything it reports is **measured by actually executing the bound model on the bound dataset**. It never
estimates from hardware specifications, never fabricates telemetry, and never turns one machine's numbers
into a statement about hardware in general. Every number is an **environment-specific engineering
measurement**: it describes this machine, this interpreter, these libraries and this moment.

There is **no composite resource or systems score** anywhere in the package, and **performance
measurement is not reliability**: a fast model can be unreliable and a slow model can be reliable. The
laboratory supplies evidence for the `RESOURCE_SYSTEM` dimension of a reliability profile; it does not judge it.

## What one measurement is

A `ResourceSpec` (`rsp_`) is the typed, immutable, content-addressed definition. It names everything that can
change what is measured: the registered model and dataset (and split), the timed operation (`PREDICT` or
`PREDICT_PROBA`), the requested batch size, the number of samples, the number of warmup and measured trials, the
worker count, the per-trial timeout, the measurement configuration (percentiles, confidence, resamples, seeds,
CPU and memory accounting), an explicit workload subset (a slice or a window of a baseline run), model-level
stress components, references to calibration/stress/data-quality analyses, the requested device, the seed and a
free-text note about the environment (power mode, other load, ...). Any meaningful change alters `spec_id`;
optional blocks that are empty are omitted from the identity, so adding a feature never renames an older spec.
The identity is the **experimental definition**. It says nothing about the machine: the environment is captured
per run and enters the provenance fingerprint.

One **trial** is one full pass over the workload, in batches of the requested size. A measurement is
`warmup_trials` warmup passes followed by `repeats` measured passes, run as **one Run** through the normal
execution engine (`experionyx.resources.engine:run_resource_analysis`). Every batch size of a sweep is its
own spec, its own Run and its own `ResourceAnalysis`.

## Four different times, never one number

| Quantity | What it is |
|---|---|
| **initialization** | model load (as measured by the adapter when the executor loaded the model, before the procedure), model-stress build, and the preparation of the workload's batches. Reported separately, never in a trial. |
| **warmup** | the warmup passes. Recorded raw, reported separately (`warmup_seconds`), **excluded from every steady-state statistic**. |
| **steady state** | the measured passes that completed. Latency, throughput and their statistics use only these. |
| **end to end** | the wall time of the whole procedure (initialization, warmup and measured passes), reported next to the parts, not instead of them. |

The workload's batches are materialized once, **before any timing**, so data loading is not in a trial. A trial
times only the model calls (and the loop around them).

## Timing: what is measured and its limits

* Wall time is `time.perf_counter` (monotonic; the highest-resolution clock Python provides). Its **resolution**
  is read from `time.get_clock_info` and stored with every analysis. A timer step on the development machine is
  tens of nanoseconds; the statistics document warns when the median trial is within 100x of the resolution, where
  quantization dominates.
* A trial is timed around the whole pass; each inference call is timed around the adapter call. The adapter's own
  reported `inference_seconds` is kept beside it as a cross-check.
* Timing is **not reproducible bit for bit**. It varies with frequency scaling, thermal state, other processes,
  caches and the operating system's scheduler. Only the **definition** and, where the model is deterministic, the
  **model outputs** are reproducible.
* **Per-sample latency** is measurable only where every batch holds one sample (batch size 1); otherwise it is
  `UNAVAILABLE`. `amortized_per_sample_seconds` (trial time / samples) is reported separately and labelled as an
  average over the batch, not a latency.

## Latency and throughput

Latency reports the raw per-batch and per-trial timings (always preserved in `trials.json`), and for the completed
measured trials: `n`, mean, median, standard deviation, min, max and the requested percentiles (default 50, 90,
95, 99; linear interpolation, the "type 7" definition used elsewhere in the repository). A tail percentile of few
observations is close to the maximum and is **flagged**. Bootstrap percentile intervals (Phase 10) of the mean and
median of the trial times are given only with at least `min_trials` completed trials (default 5); below that the
interval is `INSUFFICIENT_EVIDENCE` and none is reported.

Throughput is computed from **actual completed samples and actual elapsed time**: `samples/sec` and `batches/sec`
per trial; nothing is derived from a hardware specification. A trial that failed or timed out has no throughput.

## Batch size and workload size

A sweep runs an explicit list of batch sizes; each one is a separate persisted unit recording the **requested**
batch size, the **effective** batch sizes (min, max, mean, and the final partial batch), the sample count,
latency, throughput, memory and failures. Invalid batch sizes (zero, negative, non-integer, boolean, above the
limit, duplicated in a sweep) are refused **before anything executes**. A batch size larger than the split is
allowed and its effective size is recorded. `n_samples` limits the workload to the first `n` samples. A subset
(slice or window) thins each requested batch, so effective batch sizes can be smaller than requested; they are
recorded.

## Concurrency

`workers > 1` runs the batches of a trial on a thread pool. It is supported **only for adapters that declare
`THREAD_SAFE_INFERENCE = True`** (read-only inference on one loaded instance; the scikit-learn and PyTorch adapters
declare it). Any other adapter is reported **`UNAVAILABLE`** and nothing is measured; concurrency is never
introduced to produce numbers. Worker count is a configuration: latency and throughput under contention are
measured on this machine and its thread scheduler; a slowdown from contention (for example many tiny batches) is a
valid result. Output digests are assembled in batch order, so a difference in outputs between worker counts is real
evidence of non-determinism.

## Memory and CPU

* **Memory** is the process's **peak resident set size** (`resource.getrusage`, `ru_maxrss`), a **high-water
  mark**: it can only grow and cannot be reset. The analysis reports the mark at procedure start (which already
  includes the model and dataset, because the executor loaded them first), after initialization, after warmup and
  after the measured passes, plus each trial's growth. **Zero growth means the earlier peak was not exceeded, not
  that no memory was used.** The model-load memory delta is therefore `UNAVAILABLE`. Python and the allocator
  may retain freed memory, so the mark is not the model's own footprint. Units are handled per platform (bytes on
  macOS, KiB on Linux); other platforms report `UNAVAILABLE`. **No GPU memory** is measured: no supported backend
  exists, and the report says so.
* **CPU** is `time.process_time`: user + system CPU seconds of this process across all threads, per trial, and
  `cpu_seconds / wall_seconds` as utilization (it can exceed 1 with several threads; it is not machine
  utilization). Where the platform has no process CPU clock, or CPU accounting was not requested, both are
  `UNAVAILABLE`; they are **never inferred from wall time**.

## Timeouts, failures and resource limits

`timeout_seconds` is a per-trial limit. Its semantics are **cooperative**: it is checked as each inference call
finishes. A call already running cannot be interrupted from Python, so a trial can overrun its timeout by at most
the call in flight; that is recorded as `overrun_seconds`. A trial that exceeds the timeout is `TIMED_OUT`, with
the samples completed before the deadline, the samples finished late, and the samples never attempted. Under
concurrency the pending work is cancelled and the running work is drained before the trial ends, so nothing keeps
running after a trial is recorded. A trial whose inference raised is `FAILED`, with the exception type and
message, the failing batch, and the completed / failed / not-attempted counts. **Neither a timeout nor a failure is
ever counted as a success**, and neither enters a latency or throughput statistic (they remain in `trials.json`).
A timeout not larger than the timer resolution is refused as `UNAVAILABLE`.

A **memory limit** (or any OS-level limit) cannot be enforced in-process without terminating the host process, so
requesting one is refused as **`UNAVAILABLE`**; nothing is simulated. The executor's own `ResourceLimits` remains
advisory; the timeout and the worker count recorded there are enforced by this procedure and nothing else is.

## Statistics

Statistics reuse Phase 10 (`stats.core`): summaries, bootstrap percentile intervals, effect sizes, permutation and
paired sign-flip tests, and the multiple-comparison corrections (`NONE`, `BONFERRONI`, `BENJAMINI_HOCHBERG`).

* Three kinds of variability are kept apart. **Measurement variability** is the spread of repeated trials (the
  coefficient of variation and the lag-1 autocorrelation in trial order, a drift diagnostic). **Statistical
  uncertainty** is the bootstrap interval of the mean/median. **Hardware/environment variability** cannot be
  estimated from one machine and is reported `UNAVAILABLE`.
* Repeated trials on one machine are **exchangeable repeated measurements**, not independent samples of a hardware
  population. The bootstrap assumes exchangeable trials; a large lag-1 autocorrelation weakens it and is reported.
* `resources compare` and `resources sweep` compare analyses against the first: **UNPAIRED** on trial-level
  values (throughput, trial seconds; separate executions are not matched), **PAIRED by sample ID** on per-sample
  latency (batch size 1, identical sample IDs). The family of treatments against the reference is corrected for
  multiple comparisons. `trial_seconds` is compared only between analyses that processed the same number of
  samples. Analyses from different environments (environment, device or measurement backend) are refused unless
  explicitly allowed, and are then labelled as confounded. Nothing is stored by a comparison, and a difference is an
  observation, never a causal claim.

## Provenance, replay and reproducibility

Every analysis is a Run with its own provenance and records: the model and dataset (IDs and verified fingerprints),
split, source revision, the environment snapshot (OS, Python, package versions, CPU count, device, measurement
backends, timer resolution, load average at start), the resource specification (batch size, workers, warmup,
measurement configuration, timeout), the seed, the stress identities, the workload digest and the digests of every
artifact. `provenance_fingerprint` covers the definition and the environment and **not the timings**: a change to
the configuration, the environment snapshot, the measurement backend or the stress changes it; a repeated
measurement of the same definition in the same environment does not, and is a **new** analysis (its identity
includes its Run), never an overwrite.

`resources replay` re-executes the Run as a **new** Run and compares the **definition** (`spec.json`) exactly and the
**model output digests**. Timing is deliberately not compared (the median times of both runs are shown for
information). The output digest is over the per-sample outputs regardless of batching, so a batch size that changed
the outputs, or a model that is not deterministic, is visible.

## Artifacts and records

```
resources/
  spec.json           the definition, fingerprints, source revision, environment id, workload digest
  environment.json    OS, Python, packages, CPU count, device, measurement backends, timer resolution
  trials.json         EVERY trial (warmup and measured, failed and timed out): raw per-batch timings, counts, CPU, memory
  observations.json   the raw per-trial wall times and completed samples
  statistics.json     the descriptive statistics, intervals, variability, timer warnings and assumptions
  summary.json        what the analysis states: initialization, warmup, steady state, end to end, memory, CPU, failures
```

Schema **v15** adds `resource_analyses` (`rsa_`) and `resource_trials` (`rst_`, one per warmup or measured pass);
both are append-only and immutable. The raw measurements always live in the digest-verified artifacts; the records
are the queryable index.

## Integrations

* **Reliability profile**: `RESOURCE_SYSTEM` dimension (`resource_analyses` in the profile spec) exposes the latency,
  throughput, timeout/failure, memory, CPU and environment observations as recorded. `UNAVAILABLE` if none is
  referenced (the absence of measurements is neither good nor bad); `INSUFFICIENT_EVIDENCE` if none had enough
  completed measured trials. A resource analysis of another model or dataset is refused. It is separate from the
  baseline's own `LATENCY` dimension and forms no score.
* **Benchmarks**: `resources` (a list of resource specs over the benchmark's own model and dataset) makes each spec
  a unit; the `resource_analysis` section counts `planned`, `executed`, `unsupported`, `failed`, `timed_out` and
  `insufficient_evidence` units and references the analyses without duplicating them. A benchmark without
  `resources` keeps its identity.
* **Model stress**: `stress` applies model-level Phase 14 components (`PARAMETER_NOISE`, `PARAMETER_SCALE`,
  `THRESHOLD`) before measuring. The stress identity and the resource identity are recorded separately; a timing
  difference under stress is an observation, not a causal claim. Batch size and repeated execution are resource
  parameters here; input stress runs through the Fault Laboratory.
* **Calibration, stress and data-quality analyses**: referenced as execution context (`calibration_analyses`,
  `stress_analyses`, `quality_analyses`), verified to exist, part of the identity, never recomputed. A resource
  change is execution context unless it changes model outputs, which the output digest shows.
* **Slices and windows**: population-level by default. `subset` measures only the samples of an explicitly named
  slice or window of a baseline run (Phase 11 / Phase 12 machinery), which is meaningful only where the workload
  itself differs; nothing is run for every slice, and ordinary hardware variation is never called drift.

## CLI

`experionyx resources validate | list | inspect | run | sweep | compare | replay` (see [cli.md](cli.md)). Invalid
requests are clean `error:` messages (exit 2) and execute nothing; `run`/`sweep` exit 3 when a trial failed or
timed out or the evidence is insufficient; `replay` exits 1 when the definition or outputs differ.

## Known limitations

* One process, one machine, one interpreter; no cross-machine or cross-environment estimate.
* Peak memory is a process high-water mark; no per-phase or model-only footprint; no GPU memory.
* The timeout is cooperative and cannot interrupt a call in flight; memory and CPU limits cannot be enforced.
* Concurrency is threads only, and only for adapters that declare thread-safe inference.
* Only `PREDICT` and `PREDICT_PROBA` are timed; training, data loading and end-to-end serving latency are not.
* Timing values are environment-specific engineering measurements and not a hardware benchmark.

## Example (an engineering measurement, not a benchmark)

The following was measured on the development machine (macOS on Apple silicon, Python 3.12, CPU) with the
bundled iris workload (150 samples) and a real scikit-learn logistic regression, 30 measured trials after 3
warmups. It shows the *shape* of the output, not a performance claim; on another machine, or later on this one, the
values differ.

| batch size | batches | median trial | mean throughput | trial CV |
|---|---|---|---|---|
| 1 | 150 | 6.80 ms | 22,000 samples/s | 1.8 % |
| 8 | 19 | 0.89 ms | 165,000 samples/s | 3.7 % |
| 38 | 4 | 0.20 ms | 728,000 samples/s | 3.0 % |

Outputs were identical at every batch size (one output digest), all trials completed, and the sweep comparison
(Benjamini-Hochberg over five comparisons against batch size 1) was significant for every batch size.
