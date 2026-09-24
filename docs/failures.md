# Failure Registry and Failure Discovery

Failure discovery turns the *stored* results of real runs (evaluations and fault experiments)
into **candidate failure modes**: groups of similar failure signals that meet configurable
evidence requirements. It is a deterministic, rule-based pipeline. No LLM and no learned model
decides anything here, and nothing is measured that was not already recorded by a run.

```
existing runs -> failure signals -> normalized signatures -> interpretable similarity
-> deterministic clusters -> stability diagnostics -> candidate failure modes
-> configurable evidence evaluation -> (optional) reproduction check -> explicit confirmation
```

## Vocabulary (kept distinct on purpose)

| Concept | What it is | Record |
|---|---|---|
| **Failure signal** | One normalized failure observation from one run (e.g. "class 1 recall fell by 0.25 under noise sigma=4"). Keeps full lineage. | `FailureSignal` (`fsg_`) |
| **Failure cluster** | Signals grouped by a named, versioned, deterministic algorithm. A statistical grouping only. | `FailureCluster` (`fcl_`) |
| **Candidate failure mode** | A cluster registered with a lifecycle status of `DISCOVERED`, `CANDIDATE` or `SUPPORTED`. | `FailureMode` (`fmd_`) |
| **Registered / confirmed failure mode** | A mode a person has explicitly moved to `CONFIRMED` after a passing reproduction. | `FailureMode`, status `CONFIRMED` |
| **Evidence** | Retained, append-only support (signals, criteria evaluations, reproductions, and every status change with its actor and reason). | `FailureEvidence` (`fev_`) |
| **Relationship** | A typed edge of the backend-only knowledge graph. | `FailureRelationship` (`frl_`) |

The structured fields of a mode (classes, slices, fault types, effect statistics, prevalence,
impact, severity vector, criteria checks) are **authoritative**. Its `title` and `description`
are deterministic templates that explain them; they are not evidence.

## Taxonomy

`INPUT_SENSITIVITY`, `DISTRIBUTION_SHIFT`, `LABEL_ERROR`, `CLASS_SPECIFIC_ERROR`,
`FEATURE_DEPENDENCY`, `CALIBRATION_FAILURE`, `HIGH_CONFIDENCE_ERROR`, `LOW_CONFIDENCE_ERROR`,
`SYSTEMATIC_MISCLASSIFICATION`, `REGRESSION_ERROR`, `PERFORMANCE_DEGRADATION`,
`RESOURCE_SENSITIVITY`, `UNKNOWN`.

These are **EXPERIONYX diagnostic categories**, a working vocabulary for organizing evidence.
They are not a universal or validated classification of ML failures. `DISTRIBUTION_SHIFT`,
`LOW_CONFIDENCE_ERROR`, `RESOURCE_SENSITIVITY` and `PERFORMANCE_DEGRADATION` are in the vocabulary
but no built-in extractor emits them yet (there are no shift or resource faults). A cluster whose
signals disagree on category is labelled `UNKNOWN` rather than guessed.

## Signal extraction

Extractors are pure functions over stored, digest-verified results. A signal exists only when a
stored value crosses a threshold in `ExtractionConfig`.

| Signal kind | Source | Fires when |
|---|---|---|
| `PER_CLASS_RECALL_LOW` | evaluation | a class's recall is below `class_recall_floor` |
| `CONFUSION_PAIR` | evaluation | true class *a* is predicted as *b* at a row rate >= `confusion_pair_rate_min` |
| `HIGH_CONFIDENCE_ERRORS` | evaluation | the high-confidence error rate >= `high_confidence_error_rate_min` |
| `CALIBRATION_ECE` | evaluation | ECE >= `ece_min` |
| `REGRESSION_TAIL` / `REGRESSION_BIAS` | evaluation | max abs error / MAE, or \|mean signed error\| / MAE, above a ratio |
| `SLICE_DEGRADATION` | evaluation | a slice is worse than overall by >= `slice_deterioration_min` (direction-aware) |
| `METRIC_DEGRADATION` | fault vs baseline | the primary metric deteriorated by >= `fault_effect_min` |
| `CLASS_RECALL_DEGRADATION` | fault vs baseline | a class's recall dropped by >= `class_recall_drop_min` |
| `ECE_INCREASE` | fault vs baseline | ECE rose by >= `ece_increase_min` |

Explicitly selected evaluation runs contribute the failures visible in their own evaluation.
Fault experiments contribute how each treatment run is *worse than its baseline*; weaknesses the
baseline already has are **not** attributed to the fault. Failed and skipped trials contribute no
signals and are listed, with their reasons, under `skipped` in `failure-analysis.json`.

**Sample IDs** are preserved: a signal lists the affected sample indices of the evaluated split
(for fault signals, the samples that were correct in the baseline and wrong under the fault).
They are bounded by `max_sample_ids_per_signal`; when cut, `sample_count` keeps the true number
and the signal reports `truncated`. Signals for aggregate quantities (ECE, slices) carry no
sample IDs, and their `detail.sample_ids_recorded` is `false`.

**Classes are exact.** A class is keyed by the JSON form of its label, so class `2` and class `20`
(and `2` and `"2"`) are never the same class.

## Signatures, similarity and reasons

Each signal has a deterministic *signature* (kind, class, predicted class, slice, fault type,
direction, severity band). Similarity is a weighted mean over the dimensions that apply to a pair
(`SimilarityWeights`): error type, class, slice, fault type, fault-parameter proximity,
affected-fraction proximity, severity (relative magnitude), model and dataset fingerprint. A
dimension that does not apply to a pair (e.g. two signals with no fault) is *excluded*, not scored
zero. Every comparison returns its per-dimension reasons.

Class (with predicted class) and slice are **hard constraints**: signals about different
classes or different slices are never similar, however alike everything else is.

Cross-fault and cross-experiment reuse are configuration, not code: by default a Gaussian-noise
signal and a uniform-noise signal do not merge (fault type is a dimension); setting the fault
weights to zero lets them cluster. Fault signals use the *source* dataset fingerprint, so
signals from different faults on one dataset share it.

## Bounded comparison

Only signals in the same (class, predicted class, slice) block can be similar, so only those are
compared. `max_pairwise_comparisons` bounds the total; a block that would exceed the budget is
**not** compared pairwise, its signals are linked by identical signature only, and the block is
listed under `limits.blocks_compared_by_exact_signature_only`. Other bounds fail explicitly
(`max_source_runs`, `max_signals` raise `FailureLimitError` and analyze nothing) or record what
was cut (`max_clusters`: dropped clusters are listed with their sizes).

## Clustering and stability

`SIMILARITY_COMPONENTS` (default) takes connected components of the graph of pairs at or above
the threshold; `SIGNATURE_GROUPING` groups by identical signature. Both are deterministic and
independent of input order. **Stability** re-clusters at threshold +/- `sensitivity_delta` and
reports each cluster's worst best-Jaccard overlap with the re-clusterings; **compactness** is the
mean pairwise similarity of (at most 100) members. Connected components can chain loosely similar
signals together; low compactness or stability is the warning sign and both feed the criteria.

## Measurements

Kept separate, each with its denominator:

- **Prevalence**: runs with a signal / runs analyzed by this discovery.
- **Impact**: mean, min and max signal magnitude (units are signal-specific, see each signal's
  `detail.metric`; magnitudes of different kinds are not comparable).
- **Severity**: a *vector* (magnitude mean and max, mean affected-sample fraction with its
  denominator, run breadth, class breadth). No single severity score is computed.
- **Co-occurrence** is recorded as `CO_OCCURS_WITH` (two modes seen in the same runs, with the
  shared-run count and Jaccard). It is **not** an interaction: `INTERACTS_WITH` is reserved for a
  future, explicitly designed analysis and is never produced here.

## Evidence criteria

`EvidenceConfig` holds two `EvidenceCriteria` (candidate and supported): minimum signals, runs,
experiments and seeds; minimum mean effect; direction consistency; optionally that a seeded
bootstrap interval of the mean magnitude lies above a null region; minimum compactness and
threshold stability. Each check records `required` versus `observed`. A cluster is `CANDIDATE`
if it meets the candidate criteria and `SUPPORTED` if it also meets the supported criteria;
otherwise `DISCOVERED`. **The defaults are heuristic starting points, not calibrated
statistical guarantees**; tighten or relax them explicitly. The bootstrap interval treats
signals as exchangeable, which they are not (signals from one run/seed are related), so it is a
descriptive spread, not a confidence statement about a population.

## Lifecycle

`DISCOVERED -> CANDIDATE -> SUPPORTED -> CONFIRMED`, with `REJECTED` and `DEPRECATED` reachable
along validated transitions (`FAILURE_TRANSITIONS`); `REJECTED` and `DEPRECATED` are terminal and
`CONFIRMED` can only be deprecated. Every change stores a `TRANSITION` evidence record with actor
and reason. Evidence is never deleted, including after rejection. Discovery itself only assigns
`DISCOVERED`, `CANDIDATE` or `SUPPORTED`, and its actor is `experionyx-discovery`. Re-running a
discovery never changes the status of a mode a person has already moved.

**Nothing is confirmed automatically.** `CONFIRMED` requires the mode to be `SUPPORTED`, at least
one passing reproduction summary, and an explicit actor and reason (`failure confirm`).

## Reproduction check

`failure reproduce` replays up to `max_runs` of a mode's strongest supporting runs as **new**
runs (never touching the originals), re-extracts signals from each replay with the original
extraction configuration, and compares the reproduced magnitude to the original: a run passes if
`|reproduced - original| <= max(absolute, relative * original)`; a finding that does not
reappear fails. It records the original result, reproduction result, tolerance, pass/fail and
provenance (replay run, environments, seed, source state) as evidence and `REPRODUCED_BY` edges,
and reports how many supporting runs were not replayed because of the bound. The overall check
passes at `min_pass_fraction`. **It never changes the mode's status.** It uses the discovery's
configuration hash and refuses a different one.

What this establishes: the finding reappears when the same recorded procedure, seed and inputs
are run again in the *current* environment (repeatability). It does not establish that it holds
on other data, models or environments (reproducibility or replicability in the wider sense).

## Persistence, artifacts and determinism

Schema v5 adds `failure_signals`, `failure_clusters`, `failure_modes`, `failure_evidence` and
`failure_relationships` (foreign keys, immutability triggers; only a mode's `status` is mutable,
via compare-and-swap). A discovery is an ordinary Run of `experionyx.failures.engine:run_failure_discovery`
and writes `failure-signals.json`, `failure-clusters.json`, `failure-candidates.json` and
`failure-analysis.json` (configuration and its hash, algorithm versions, sources, skipped items,
limits, counts, and the statement of what can and cannot be concluded), plus observations and a
claim with run and artifact evidence for each `CANDIDATE`/`SUPPORTED` mode. Identity is
content-addressed, so re-running is idempotent. The only randomness is a seeded bootstrap (seed in
the configuration). Sources may span investigations; the result is homed in one, and each
source's own investigation is recorded.

## What discovery can and cannot conclude

**Can:** which stored signals are similar under the configured rules; how runs group; whether
each group meets the configured evidence requirements; that a finding reappears on replay.

**Cannot:** *why* a failure occurs (there is no causal claim anywhere in the pipeline); that a
group is a real failure mode of the model in general; that it will recur on other data; that two
co-occurring modes interact; anything about failure types no extractor emits. Fault experiments
are induced perturbations, so a mode from a fault says "under this perturbation the model
degraded in this way", not "the model fails this way in deployment".

## Known limitations

- Connected-components clustering can chain; check compactness and stability.
- Similarity weights and criteria defaults are unvalidated heuristics.
- Class-level signals need a classification confusion matrix; regression yields only tail and
  bias signals plus fault degradation of the primary metric.
- Slice and calibration signals carry no per-sample IDs.
- A later discovery over a larger source set creates *new* modes (different clusters); it does not
  merge into earlier ones. Deprecate the superseded ones explicitly.
- `LATENCY` and resource effects are not signals.
- Reproduction replays with whatever adapters and environment are present now.

## Evidence and failure knowledge graph (Phase 19)

`experionyx graph query <snapshot> runs-for-failure-mode <fmd_…>` and `... model-to-failure /
dataset-to-failure` traverse a constructed `GraphSnapshot` to trace which runs contributed to a
failure mode, and whether a bounded path exists between a model or dataset and it. These are
provenance/reference paths through what the registry already recorded, never a re-derivation of
similarity, clustering or evidence criteria above; a path existing is not itself evidence toward
`CANDIDATE`/`SUPPORTED`/`CONFIRMED` status. See [graph.md](graph.md).
