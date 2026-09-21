# Calibration & uncertainty laboratory (Phase 15)

The laboratory asks whether a model's probability outputs correspond to observed correctness, and how
uncertain the *measurements* of that correspondence are. It reads the **stored predictions** of finished
runs (the baseline evaluation, optionally a calibration run, and the completed trials of referenced stress
analyses). It never loads, calls or changes the model.

It is a scientific analysis subsystem, not a "confidence score" feature. Four different things are kept apart:

1. **Predictive confidence / calibration**: does a stated probability match how often the model is right?
2. **Statistical uncertainty of the measured metrics**: how much would ECE, accuracy, Brier score, ... move under
   resampling? (Wilson and bootstrap intervals.)
3. **Predictive uncertainty** exposed by the model output: only descriptive dispersion of the stored probability
   vector (entropy, margin); nothing else is available for the current adapters and everything else is reported
   `UNAVAILABLE`.
4. **Epistemic / aleatoric uncertainty**: never claimed. No adapter declares an architecture that supports the
   decomposition, so it is `UNAVAILABLE`.

**Confidence is not automatically uncertainty.** A softmax maximum is a number the model produced; whether it is
trustworthy is exactly what the calibration analysis measures. There is **no universal calibration or uncertainty
score**: no composite, no ranking, no verdict, anywhere in the package.

## Supported model outputs

| Output | Status |
|---|---|
| classification with `predict_proba` scores (e.g. scikit-learn) | supported; declared `PREDICT_PROBA` |
| classification whose outputs are logits, softmaxed by the baseline evaluation (`score_source: SOFTMAX_LOGITS`) | supported; declared `SOFTMAX_LOGITS`, and the artifact records the **assumption** that the outputs are logits |
| classification with labels only | refused before execution: `UNAVAILABLE`, no probability scores |
| regression | refused before execution: `UNAVAILABLE`, no documented uncertainty representation (none is fabricated) |
| raw logits | not stored by the baseline evaluation (only their softmax), so a raw-logit object is `UNAVAILABLE` |

The spec **declares** the representation (`prediction_source`); the stored run is checked against it. A mismatch is
refused (`scores are never treated as another representation`). The probability *for a specific class* (classwise),
the probability *of the predicted class* (top-label) and the whole *vector* are distinct objects (below).

## Definitions and conventions

`TOP_LABEL`: the confidence of a sample is the probability the model assigns to its **predicted** class (this equals
the maximum when the prediction is the arg-max); the event is "the prediction is correct". `CLASSWISE`: for each class
`k`, `p_k` against the event "the true class is `k`". `PROBABILITY_VECTOR`: proper scoring rules of the whole vector.
Top-label calibration is **not** full-vector calibration.

* **Bins.** Uniform: edges `k/B`. Quantile: the observed confidences at equal-count ranks (ties share a bin, so fewer
  bins than requested may result); the outer edges are the observed minimum and maximum. Bin `i = [lower, upper)`, the
  last closed at its upper edge. Membership is decided by bisection on the *reported* edges, so a confidence exactly on a
  boundary belongs to the upper bin and float rounding never disagrees with the stored bounds. Quantile binning is
  refused (`INSUFFICIENT_EVIDENCE`) below 5 observations per requested bin.
* **Every bin is listed**, including empty ones. Each has `count`, `mean_confidence`, `accuracy` (observed rate),
  `gap = mean_confidence - accuracy` (positive = overconfident), `abs_gap`, an evidence label (`EMPTY`,
  `SINGLE_OBSERVATION`, `SPARSE` below `min_bin_samples`, `OK`) and a Wilson interval of the observed rate.
* **ECE** = `sum_b (n_b/n) |accuracy_b - mean_confidence_b|` over non-empty bins. **MCE** = the maximum `|gap|` over
  non-empty bins (a single-observation bin can dominate it; look at the bin evidence). Both **depend on the binning and
  the sample**, which is why the binning is part of the spec identity.
* **Top-label Brier** = mean `(confidence - correct)^2`. **Top-label log loss** = `-mean[c ln(conf) + (1-c) ln(1-conf)]`
  with probabilities clipped to `[1e-15, 1-1e-15]` (so a confident error is large but finite).
* **Multiclass Brier** = mean over samples of `sum_k (p_k - 1[y=k])^2` (0 to 2); **NLL** = `-mean ln p_y` (clipped).
* **Classwise ECE** per class from the reliability of `p_k` against `1[y=k]`. A class with fewer than
  `min_class_positives` positive samples reports `INSUFFICIENT_EVIDENCE` and no ECE. The mean over classes is reported
  only next to the per-class values.
* **Probability validation.** A row is *invalid* (and reported with a reason, never repaired or silently dropped) if its
  scores are missing, of the wrong width, non-finite, outside `[0, 1]` (beyond `1e-9` of rounding), not summing to 1
  (`1e-6`), or if its target or prediction is not a declared class. Probabilities exactly 0 or 1 are valid.
* **Predictive dispersion** (descriptive, per sample and aggregated overall / when correct / when incorrect): raw
  entropy `H = -sum p ln p` in nats, normalized entropy `H / ln K` (defined for `K >= 2`), and the top-2 margin.
  These describe this model's output vector; they are not calibrated and not epistemic.

## Post-hoc calibration

Top-label **Platt scaling** (`P(correct) = sigmoid(a * logit(conf) + b)`, Newton's method with step halving from the
identity map, an L2 ridge of `1e-6` on the slope only so separable data converge; the ridge is recorded) and top-label
**isotonic regression** (pool adjacent violators over tied-then-sorted confidences; piecewise-linear, clipped outside the
fitted range). Both are deterministic. A calibrator is **not** fitted on fewer than `min_samples` observations or on a
single outcome (all correct / all incorrect): the method block is then `INSUFFICIENT_EVIDENCE`.

**Calibration and evaluation data are always separated:**

* `SPLIT` (default): the baseline's samples are partitioned by a seeded hash of the sample ID (independent of row
  order): the calibration part fits, the held-out part reports. The IDs, digests, fraction and seed are stored.
* `RUN`: fit on a **different run's** predictions (same model fingerprint, classes and representation) and evaluate on
  the baseline. A run over the **same dataset fingerprint and split** (or the baseline itself) is refused as
  `LEAKAGE` before anything executes. Leaky calibration is never allowed, so there is no "allow leakage" option.

Stored: the calibration and evaluation data identity (source, n, ID digest, dataset fingerprint, split, predictions
digest), the parameters, the leakage statement, and **before and after** metrics on the *same* held-out samples with a
paired comparison. The original model and baseline artifacts are untouched. After calibration only top-label metrics
exist: a top-label calibrator defines a calibrated confidence, not a calibrated probability vector, so the vector
metrics are `UNAVAILABLE` for it. Limitations: one split (no cross-fitting), top-label only, Platt assumes a
sigmoid-shaped miscalibration, isotonic overfits small calibration sets.

## Statistical uncertainty

* Accuracy and every bin's observed rate: Wilson intervals (Phase 10 `proportion_interval`).
* Every scalar metric (accuracy, mean confidence, ECE, MCE, top-label Brier and log loss, multiclass Brier and NLL):
  a **seeded percentile bootstrap** over observations (quantile edges are recomputed per resample). Method, seed,
  resamples, confidence and the assumptions (independent identically distributed observations, representative sample,
  ECE small-sample bias) are recorded. Zero-variance resamples are flagged as degenerate.
* **Insufficient data yields no interval**: below `min_samples` a context is `INSUFFICIENT_EVIDENCE`, its bins are
  still shown, and no metric or bootstrap interval is produced (a Wilson interval of accuracy remains valid).

## Contexts, comparisons and shifts

The analysis has one baseline context and only the contexts you request: **slices** (Phase 11 definitions, with membership
IDs and digests; a slice with too few observations is `INSUFFICIENT_EVIDENCE`, a slice over a missing field
`UNAVAILABLE`), **temporal / distribution windows** (Phase 12 windows resolved by ordering key, never row position) and
the completed trials of **stress analyses** (Phase 14; the trial's own dataset/model fingerprints, seed, stress IDs and
predictions digest are recorded, because input and parameter stress *deliberately* derive the dataset or model).

Comparisons (`b - a`): baseline vs each stress trial (PAIRED when the sample IDs match), window vs window
(UNPAIRED, disjoint samples), each slice vs the rest (UNPAIRED), before vs after calibration (PAIRED). **Partially
overlapping ID sets are neither paired nor independent, so no inference is made** (shifted IDs are never paired).
Each metric keeps its raw values, difference, interval, effect size, raw p-value, adjusted p-value (Bonferroni or
Benjamini-Hochberg over exactly the family of that kind and metric) and a separate `practically_material` flag
(`|difference| >= practical_delta`); significance and magnitude are never conflated. Window and stress comparisons add
descriptive confidence-distribution changes (KS, Wasserstein-1), predicted-class distribution change (total variation)
and accuracy change. **These are observations that the calibration differs between two sample sets, not
explanations: no analysis says a stress or a time period caused miscalibration.**

## Data-quality evidence

Every context records `n_rows`, `n_usable`, unusable counts by reason, examples, duplicate IDs (all rows of a duplicated ID
are set aside as ambiguous), the count excluded by the evaluation itself, and whether the data were `affected`. Invalid or duplicate rows
(like any insufficient, unavailable or failed context) make the analysis `PARTIAL`. Optional `quality_analyses` (`qan_`) are referenced as data-quality context.

## Failure-discovery signals

`candidates.json` lists high-confidence incorrect and low-confidence correct samples (thresholds in the spec) and the
confidence/correctness relationship (mean confidence when correct / incorrect, AUROC of confidence for separating them).
It is **candidate evidence only**: nothing is classified as a failure mode and no failure record is created.
`CalibrationRegistry.failure_candidates(analysis_id)` exposes it to failure analysis.

## Reliability profiles and benchmarks

A reliability profile gains a separate `CALIBRATION_ANALYSIS` dimension from `calibration_analyses` (the baseline's own
`CALIBRATION` and `UNCERTAINTY` dimensions are unchanged and never merged). No analysis referenced: `UNAVAILABLE`;
no analysis with enough observations: `INSUFFICIENT_EVIDENCE`; never `PASS`. A benchmark may request a calibration analysis
(`calibration`, a spec without `baseline_run`); it **references** the analysis and counts contexts as planned, executed,
unavailable, insufficient-evidence or failed (an unsupported request is `failed` coverage, not success).

## The spec and its identity

`CalibrationSpec` (`cbs_`) is content-addressed: baseline run, `prediction_source`, `target`, `objects` (`TOP_LABEL`,
`CLASSWISE`), `binning` (`UNIFORM` / `QUANTILE`, `n_bins`), `method` (`NONE` / `PLATT` / `ISOTONIC`) and `fit`, `statistics`
(confidence, resamples, permutations, seed, `min_samples`, `min_bin_samples`, `min_class_positives`, correction, alpha,
`practical_delta`, confidence thresholds), `slices`, `windows`, `stress_analyses`, `quality_analyses` and the optional
expected `model_id`, `dataset_id`, `split` (verified against the baseline). Any meaningful change alters the ID;
unknown fields are rejected; empty optional blocks are omitted so older specs keep their identity.

## Artifacts, provenance and replay

`calibration/spec.json`, `predictions.json` (every row with entropy and margin, the invalid rows, duplicates and context
membership), `bins.json`, `metrics.json` (with the formulas), `uncertainty.json`, `comparisons.json`, `candidates.json`,
`summary.json`: all digest-verified, so raw and intermediate evidence remain auditable. The registry indexes
`CalibrationAnalysis` (`cba_`) and one `CalibrationResult` (`cbr_`) per context (schema v14, append-only).
The provenance fingerprint covers the spec, the baseline's dataset / model / split / evaluation config, the stored
predictions digest, the calibration run, the stress trials' predictions digests, slice/window membership digests and the
calibration protocol, and changes with any of them. `experionyx calibration replay` re-executes the run and compares all
documents within `REPLAY_TOLERANCE`; `calibration compare` recomputes from the stored predictions and stores nothing.

```
experionyx calibration validate spec.json --preflight   # refuses unsupported baselines / leakage, runs nothing
experionyx calibration evaluate spec.json               # exit 3 if evidence is insufficient, unavailable or invalid
experionyx calibration list | inspect <cba_> [--document bins|metrics|...|--full] | compare <cba_> | replay <cba_>
```

## Validation

The tests are **engineering validation experiments, not scientific findings.** On the bundled iris data with a
scikit-learn logistic regression, and the blobs tensor data with a PyTorch linear classifier (softmax of logits), stored
probabilities, ECE/MCE, Brier score, log loss, reliability curves, classwise curves, and the fitted Platt and isotonic
calibrators are compared with independent NumPy, scikit-learn (`calibration_curve`, `log_loss`, `brier_score_loss`,
`LogisticRegression`, `IsotonicRegression`) and PyTorch (`softmax`) computations. Quantile bins are not compared with
scikit-learn because its edges interpolate percentiles while these use observed values.

## Limitations

* Only classification with a probability representation; regression uncertainty is `UNAVAILABLE`.
* Calibration is measured on the stored evaluation sample; it says nothing about other data, and ECE, MCE and every
  interval depend on the binning, the sample size and the i.i.d. assumption.
* No stochastic, ensemble, epistemic or aleatoric uncertainty until an adapter declares it.
* Post-hoc calibration is top-label, single-split; no temperature scaling, no cross-fitting.
* A stress or window comparison is descriptive; confounding, multiplicity beyond the declared families and distribution
  shift are the reader's to consider.
