# Experiment scheduler and orchestration (Phase 17)

The scheduler answers one question: **given a collection of experiments and the dependencies
between them, what should run, in what order, and did it actually happen?** It is a durable
orchestration layer over the existing laboratory engines (fault injection, failure discovery,
interaction analysis, robustness benchmarks, statistical analysis, slices, temporal/distribution
shift, data quality, model stress, calibration/uncertainty, resource/systems reliability). It
never re-implements any of them.

```
Scheduler -> Experiment Dispatcher -> Existing Engine
```

Every dispatched unit keeps its original engine identity: the record IDs a dispatch produces (a
`Run`, a `BenchmarkResult`, a `StressAnalysis`, ...) are exactly what calling that engine directly
would have produced. The scheduler adds planning, ordering, retry, resume, concurrency and
auditability on top; it adds no science of its own.

## Compact spec, explicit plan

A `ScheduleSpec` (`ssp_`) is a small, typed, content-addressed definition: a name, a version, an
`ExecutionPolicy`, and a list of `UnitDef`s. Each `UnitDef` names a `kind` (which engine to call),
a `parameters` JSON body (exactly what that engine's own `Spec.from_dict` expects), an explicit
`depends_on` (other unit keys in the same spec), an optional `RetryPolicy`, an optional
`ResourceRequirement`, an optional `timeout_seconds`, and a `priority`. Any meaningful change
(kind, parameters, dependencies, retry, resources, timeout, the execution policy) changes
`spec_id`.

`scheduler.graph.expand` deterministically expands a spec into an ordered `Plan`: a topological
sort (Kahn's algorithm) with a deterministic tie-break among simultaneously-available units
(higher `priority` first, then `key`), so two equivalent specs always expand to the same order.
Every reference is checked before anything is persisted: duplicate keys and references to unknown
keys are refused by `ScheduleSpec` itself; a cycle is refused by `expand` with every unit left in
the cycle named. Nothing has been dispatched at this point — `scheduler validate`/`scheduler
expand` never touch an engine.

`scheduler.engine.materialize` idempotently turns a validated plan into persisted registry
records: one `Schedule` (the immutable definition) and one `ScheduleUnit` per planned unit.

## Deterministic identity, including dependencies

A `ScheduleUnit`'s ID is a content hash of `(schedule_id, unit_key, kind, parameters,
depends_on, retry_policy, resources, timeout_seconds, priority)` — where `depends_on` holds the
**resolved IDs** of the units it depends on, not their raw keys. Changing any upstream unit's
definition changes its ID, which changes every downstream unit's ID too, exactly the way any other
content-addressed record in this project propagates a change. This is what "deterministic identity
derived from all meaningful inputs" means in practice: model, dataset, configuration, experiment
type, parameters, seed, execution policy and dependencies are all inputs to some unit's identity.

## Unit kinds

At minimum: `BASELINE_EVALUATION`, `FAULT_EXPERIMENT` (fault experiments and fault sweeps),
`INTERACTION`, `BENCHMARK`, `STRESS`, `CALIBRATION`, `RESOURCE`, `DRIFT`, `DATA_QUALITY`,
`FAILURE_DISCOVERY`, `RELIABILITY_PROFILE` and `STATISTICAL_ANALYSIS`. `scheduler.dispatch` has
exactly one function per kind; each deserializes the unit's parameters with the target engine's own
`Spec.from_dict` and calls its own entry point (`run_fault_experiment`, `run_stress_experiment`,
`run_calibration_request`, `run_resource_request`, `run_drift_request`, `run_quality_request`,
`run_discovery`, `run_interaction`, `run_profile`, `run_benchmark`, or `stats.store.create`). A
`BASELINE_EVALUATION` unit names an already-registered `Experiment`; the scheduler does not build
experiments for you — registering models, datasets and experiments is unchanged, existing
machinery ([adapters.md](adapters.md), [execution.md](execution.md)).

### Dependency references (`$dep:`)

A unit's `parameters` may embed the sentinel string `"$dep:<key>"` anywhere a dependency's result
ID belongs (a baseline run ID, an analysis ID, ...). Before dispatch, the scheduler walks the
parameters and replaces every `"$dep:<key>"` with that dependency's `primary_ref` — the ID its
successful `ExecutionAttempt` recorded (a Run ID for engines that execute through the normal
execution engine, an analysis ID for the direct-computation ones like statistical analysis).
Dry-run statically catches a `$dep:` reference to a key the unit never declared in `depends_on`
(`scheduler.dispatch.referenced_dependencies`), before anything runs.

## Execution states and transitions

```
PLANNED -> READY -> RUNNING -> SUCCEEDED
                       |    \-> FAILED -> RETRY_PENDING -> READY (retry)
                       |                \-> CANCELLED
                       \----------------> TIMED_OUT -> RETRY_PENDING (retry)
PLANNED / READY / BLOCKED -> CANCELLED
BLOCKED <-> READY (re-evaluated every scheduling pass)
```

`scheduler.taxonomy.UNIT_TRANSITIONS` is the single source of truth, in the same shape as
`domain._EXPERIMENT_TRANSITIONS`; `ScheduleUnit.with_status` raises on any transition not in that
table. `SUCCEEDED`, `SKIPPED` and `CANCELLED` are terminal. Every transition is committed together
with an immutable, append-only `UnitStateTransition` record — nothing changes status silently.
`ScheduleRun` (one orchestration session over a `Schedule`) has its own, coarser state machine
(`PLANNED -> RUNNING -> COMPLETED | FAILED | CANCELLED`).

## Failure handling and classification

A dispatch never raises: `scheduler.dispatch.dispatch` catches every exception and turns it into a
`DispatchOutcome`, classified into a `FailureCategory` from real information, never a guess:

* If the unit went through the normal execution engine, the classification reads the run's own
  `RunOutcome.error.stage` (`provenance.FailureStage`): `PREPARATION` → `CONFIGURATION`,
  `EXECUTION` → `DATA_MODEL` (the model/data itself raised during real execution),
  `ARTIFACT_REGISTRATION` / `INTERRUPTED` → `TRANSIENT`.
* If a dispatch was refused before any run existed (a `*Refusal`, a `ValidationError`, a
  `NotFoundError`), it is `CONFIGURATION` — retrying cannot help a bad definition.
* `UnsupportedCapabilityError` / `AdapterUnavailableError` / `DeviceUnavailableError` are
  `UNSUPPORTED` — never retried.
* A unit- or attempt-level timeout is `TIMEOUT`.
* Anything else (an unexpected exception) is `TRANSIENT`.

A failed attempt is recorded, never hidden: an immutable `ExecutionAttempt` (attempt number, both
timestamps, the classified outcome, the error type/message, the produced `primary_ref`/`run_id` if
any). The unit's own record is never overwritten.

## Retries

A `RetryPolicy` is `NONE` (one attempt) or `FIXED` (`max_attempts`, and which `FailureCategory`
values are retryable — `TRANSIENT` and `TIMEOUT` by default). A retry creates a **new**
`ExecutionAttempt` (attempt numbers 0, 1, 2, ...) and never overwrites the failed one before it. A
unit classified `CONFIGURATION` or `UNSUPPORTED` is never retried regardless of policy, because the
definition itself is the problem. `scheduler retry <schedule> --unit <key>` explicitly requests one
more attempt at a `FAILED`/`TIMED_OUT` unit regardless of whether its own policy is exhausted,
recorded as its own auditable transition ("manual retry requested").

## Resume and idempotency

There is no separate "resume mode": `run_schedule` always starts by reading the persisted status of
every `ScheduleUnit` and only dispatches units that are not already `SUCCEEDED`. A `ScheduleUnit`
found `RUNNING` at the start of an invocation belongs to a process that ended before finishing it
(the scheduler runs one orchestrator per schedule at a time); it is recovered as a `FAILED`
attempt tagged `TRANSIENT`, then unconditionally moved to `RETRY_PENDING` — recovery grants one
fresh attempt independent of the unit's own retry policy, because an interrupted process is not a
retry decision. `scheduler resume <schedule>` reconstructs the exact original `ScheduleSpec` from
the persisted `Schedule.spec` (no spec file needed) and calls `run_schedule` again; each invocation
is its own `ScheduleRun` (numbered by `sequence`, like `Run.attempt`), so the history of every
attempt is kept, while the `Schedule`'s own identity never changes.

Idempotency follows directly from content addressing: dispatching the exact same unit definition
calls the exact same engine entry point, and every one of those entry points is already idempotent
per definition (`run_benchmark`, `run_resource_request`, etc. reuse an existing result rather than
recomputing). The scheduler's own idempotency is about not re-dispatching a `SUCCEEDED` unit; the
underlying engines' idempotency is about not duplicating the science. A meaningfully different
definition gets a different `ScheduleUnit` ID and is dispatched as new work.

## Concurrency

Each worker thread opens its **own** registry connection to the same SQLite file. This needs no new
machinery: `SqliteRegistry` already serializes concurrent writers on one file via `BEGIN IMMEDIATE`
plus a connection timeout, exactly the mechanism it documents for "two writers to the same file."
Only the coordinator thread ever calls `Registry.update_status` on a `ScheduleUnit`/`ScheduleRun`
or writes a `UnitStateTransition`; workers only ever `add` immutable records (through the engines
they dispatch to, and their own `ExecutionAttempt`), so there is exactly one writer of scheduler
bookkeeping at a time. `ExecutionPolicy.max_workers` bounds concurrency (1–64); independent units
are never serialized beyond that bound. Ordering among simultaneously-ready units is deterministic
(`priority` descending, then `key`) — never random.

### Resource awareness

A `ResourceRequirement` is a **declared, advisory** need: `cpu_intensity` (informational),
`memory_mb`, `exclusive` (no other unit runs while this one is `RUNNING`), `workload_size`
(informational). `exclusive` and `memory_budget_mb` (on the `ExecutionPolicy`) are honored as
admission control before a unit starts — but only ever gate *starting new work*; a unit that alone
exceeds the budget is still admitted once nothing else is running, so a requirement the scheduler
cannot literally satisfy never deadlocks the schedule. Nothing here is an OS-level guarantee: no
memory is reserved, no CPU is pinned. A requirement the scheduler cannot validate is left
unenforced, never fabricated as satisfied — the same honesty as [resources.md](resources.md)'s
`ResourceLimits` ("advisory... the executor does NOT enforce them").

## Cancellation

Within one `run_schedule` invocation, a `threading.Event` passed as `cancel_event` stops the
coordinator from starting new units; already-`RUNNING` units are left to finish (their evidence is
kept), and every still-open unit is marked `CANCELLED`. `scheduler cancel <schedule>` is a
registry-only operation for after the fact: it marks every unit that is not already terminal
(`PLANNED`, `READY`, `BLOCKED`, `RETRY_PENDING`, and `RUNNING` as a best-effort marker) `CANCELLED`
and closes the latest `ScheduleRun`. Completed evidence (`SUCCEEDED` units and the records they
produced) is never touched by cancellation. A `RUNNING` unit cannot be forcibly killed by `cancel`
— Python cannot revoke another process's or thread's work in flight; this records intent, not a
kill signal.

## Timeouts

Honest about what a thread-based scheduler can and cannot do: Python cannot revoke a running
thread. `timeout_seconds` (per-unit, defaulting from the schedule's `default_timeout_seconds`) is
checked cooperatively — the coordinator stops **waiting** on a unit once its deadline passes and
records it `TIMED_OUT`, but the dispatch keeps running in the background (inside its own thread and
its own registry connection) until it finishes naturally. This mirrors the same cooperative-timeout
honesty as [resources.md](resources.md)'s per-trial timeout. For a guaranteed hard kill, run the
engine out of process.

## Dry-run

`scheduler validate` checks only the DAG (cycles, references) — no registry is touched. `scheduler
run --dry-run` (and the library function `scheduler.engine.dry_run`) does everything short of
dispatch: materializes the `Schedule` and every `ScheduleUnit`, reports the deterministic execution
order, every structural issue (a `$dep:` reference to an undeclared dependency), and every
statically-known `ResourceRequirement` — and executes **zero** experiments. No `ExecutionAttempt`
is ever created by a dry run; every unit stays `PLANNED`.

## Provenance and auditability

* `Schedule` records the exact compact spec (`spec`), its expanded graph hash (`graph_hash`), and
  the engine version.
* `ScheduleUnit` records its resolved kind, parameters, dependency IDs, retry policy, resources,
  timeout and priority — the complete, inspectable expansion of one unit.
* `ScheduleRun` records the execution policy actually used (which may override the spec's default,
  e.g. `--max-workers`), whether it was a dry run, and its sequence among repeated invocations of
  the same `Schedule`.
* `ExecutionAttempt` records every attempt, append-only, including its classified failure and the
  engine record it produced.
* `UnitStateTransition` records every status change, append-only, with a reason.

Together these reconstruct what was planned, what executed, in what order, why a unit was skipped
or blocked, what failed, what was retried, and what evidence resulted — without re-deriving it from
logs. Every dispatched unit's own real `Run` (where the target engine goes through the execution
engine) carries its own full [provenance.md](provenance.md) exactly as it would running that engine
directly; the scheduler adds a second, orchestration-level layer of provenance on top, never a
replacement for it.

## Replay

`scheduler replay <schedule>` reconstructs the schedule's exact spec and runs it again as a new
`ScheduleRun`, then diffs each unit's status and `primary_ref` against the previous run. `RESOURCE`
units are marked `environment_dependent` in the report and excluded from the pass/fail verdict:
they measure the live machine, so a difference there is expected variance, not nondeterminism —
the same distinction [resources.md](resources.md) and [benchmarks.md](benchmarks.md) make for
timing-dependent observations. Every other kind is a deterministic computation given the same
seeds and evidence, so any difference there is a real one.

## Schema and artifacts

Schema **v16** adds `schedules` (`sch_`), `schedule_runs` (`scr_`, mutable status),
`schedule_units` (`sun_`, mutable status), `execution_attempts` (`att_`, append-only) and
`unit_state_transitions` (`utr_`, append-only). `ScheduleRun` and `ScheduleUnit` participate in
`Registry.update_status` the same way `Experiment`/`Run`/`FaultExperiment`/`FailureMode`/
`InteractionAnalysis` already do; every other scheduler table is immutable and delete-triggers
refuse to remove a row, like every other table in the registry.

## CLI

`experionyx scheduler validate | expand | run | status | inspect | graph | resume | cancel | retry
| replay` (see [cli.md](cli.md)). `validate` never touches the registry. `run`/`resume` exit 1 if
the final status is not `COMPLETED`; `--dry-run` executes nothing. `replay` exits 1 on any
non-environment-dependent difference. Invalid requests (a malformed spec, an unknown unit key, a
cyclic graph) are clean `error:`/refusal messages, never a crash.

## Known limitations

* Concurrency is threads sharing one process, not process- or machine-level distribution; a unit's
  own engine determines whether its work is itself thread-safe (e.g. [resources.md](resources.md)'s
  `workers` requires a thread-safe adapter).
* A `TIMED_OUT` unit's underlying dispatch is not forcibly killed; it keeps running in the
  background until it finishes on its own.
* `cancel` on a `RUNNING` unit records intent; it cannot stop work already in flight.
* Resource requirements are declared and advisory; nothing is enforced at the OS level.
* `exclusive` and `memory_budget_mb` gate the **start** of new units; they do not preempt units
  already running.
