# Provenance

`Provenance` (one per run, written when the run starts) records *inputs and context*.
`RunOutcome` (one per run, written when it ends) records *results*. Both are immutable registry
entities (`src/experionyx/provenance.py`); capture code lives in `src/experionyx/capture.py`.

## What is recorded

| Field | Meaning | In fingerprint |
|---|---|---|
| `experiment_id`, `configuration_id`, `seed` | what was run | yes |
| `environment_id` | content-addressed `EnvironmentSnapshot` | yes |
| `dependency_digest` | sha256 of the canonical `{package: version}` mapping | yes |
| `source` | `SourceRevision(state, commit, branch)` | state + commit |
| `executor_version` | EXPERIONYX version | yes |
| `execution` | procedure import path, advisory resource limits, `metadata` | yes |
| `run_id`, `started_at` | identity/time of this execution | no |
| `replay_of` | original run, if this is a replay | no |
| `inputs` | for adapter-backed runs: bound model/dataset (record ID, adapter, adapter version, fingerprint measured at load) and requested/resolved device | yes, when present |
| `runtime` | informational: Python implementation, processor, CPU count, pid, seeded libraries, dependency issues | no |

Artifacts and observations are referenced from the registry by `run_id`; the outcome lists
artifact IDs and the observation count.

## Fingerprint
`Provenance.fingerprint` = sha256 of the canonical JSON of the "yes" rows above. It is *execution
identity*: two runs with the same fingerprint were started from the same recorded inputs. It
excludes wall-clock time, the run's own ID, the branch name, pid and other runtime details, so
logically identical executions compare equal. It changes when any reproducibility-relevant input
changes, including a MODIFIED_WORKTREE state. **Equal fingerprints do not mean equal results.**

## Environment snapshot
`EnvironmentSnapshot` (reproducibility-critical, content-addressed): Python version, OS string
(`platform.platform()`), machine architecture, and every installed distribution as
`{normalized-name: version}`. Source revision is deliberately *not* part of it (`None`), so the
environment identity is stable across commits. Distributions that cannot be read reliably (no
name, invalid version) are skipped and listed in `runtime.dependency_issues`; a name installed
at several versions is reported as ambiguous. Nothing is invented. The package set is the whole
active environment, so unrelated tooling changes the environment ID.

## Source revision
`SourceState`: `REPRODUCIBLE_SOURCE` (clean git worktree at `commit`), `MODIFIED_WORKTREE`
(tracked changes **or untracked files** present), `UNKNOWN` (not a git repo, no commits, git
missing or failing). Any doubt yields UNKNOWN; a dirty tree is never recorded as clean. The
directory inspected is the executor's `source_root` (CLI: current directory). Keep workspaces
(`.experionyx/`) git-ignored or outside the repository, or every run will look modified.

## Privacy model
Provenance is meant for public repositories. It **never** captures environment variables,
usernames, hostnames, home or working directory paths, IP/MAC addresses, credentials or git
remotes. Included on purpose: OS/architecture strings, package names/versions, commit SHA, branch
name, processor name, CPU count, pid. Tracebacks stored as diagnostics have the home directory
replaced by `~` but may still contain other path fragments or data from exception messages;
review diagnostics before publishing a workspace. Procedure code and configuration values you
supply are recorded as given.

## Limits
Provenance describes what was *recorded*; it does not prove the recorded source is what ran (a
dirty tree is flagged, its diff is not stored), nor capture system libraries, hardware
accelerators, or data contents.

## Resource analyses (Phase 16)

Every resource analysis is a Run with its own Provenance record (source revision, environment, dependency
digest, and the recorded worker count and timeout). Its `resources/spec.json` records the model and dataset
fingerprints, the split, the source revision, the environment snapshot ID, the seed, the stress identities, the
workload digest and the whole specification (batch size, warmup, measurement configuration, timeout, workers);
`environment.json` adds the OS, Python and package versions, CPU count, device, the measurement backends and the
timer resolution. The provenance fingerprint covers the definition and the environment but **not the timings**,
so a repeated measurement in the same environment has the same fingerprint and is a new analysis. See
[resources.md](resources.md).

## Calibration analyses (Phase 15)

Every calibration analysis is a Run with its own Provenance record (source revision, environment,
dependency digest). Its `calibration/spec.json` records the model and dataset fingerprints, the split, the
evaluation-config hash, the declared and verified prediction representation (and the softmax assumption when
the scores were derived from logits), the target representation and class order, the binning, the
calibration method with its calibration/evaluation split or run (data identities, digests, parameters), the
bootstrap configuration and seeds, the slice / window context, the stress trials analyzed (analysis, trial and
run IDs, seeds, predictions digests) and the referenced data-quality analyses. The provenance fingerprint
changes with any of them and with the stored predictions. See [calibration.md](calibration.md).

## Stress trials (Phase 14)

Every stress trial is a Run with its own Provenance record (source revision, environment, dependency
digest) and records the model and dataset fingerprints, the split, the exact stress specification and
seed, the baseline run, the transformation's identity (a Fault Laboratory fault ID, or the parameter
digests before, after and of the stressed model, and the predictions digest), the evaluation configuration
and its artifact digests. See [stress.md](stress.md).
