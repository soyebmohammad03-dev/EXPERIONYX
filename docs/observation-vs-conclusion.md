# Observation, Finding, Interpretation, Conclusion

EXPERIONYX keeps four kinds of statement apart. Each layer may only assert what the layer
beneath it supports.

| Layer | What it is | Example | Who/what may produce it | Stored as |
|---|---|---|---|---|
| **Observation** | A value measured or derived from recorded data, with its context | "accuracy = 0.947 on 38 test samples"; "ECE = 0.057, 10 bins" | The evaluation engine | `Observation`, `MetricResult`, artifacts |
| **Finding** | An observed property that crossed an explicit, versioned threshold | "ECE 0.13 exceeds the configured 0.10 (rule ece-threshold v1)" | Deterministic rules | `AutopsyFinding` (`OBSERVED`), plus a narrow `Claim` with `Evidence` |
| **Interpretation** | A statement of what an observation or finding *means* or *why* it happened | "the model is overconfident because of class imbalance" | A researcher, or a future analysis backed by controlled experiments | `AutopsyFinding.interpretation = INTERPRETED` (never set automatically) |
| **Conclusion** | An interpretation supported by enough evidence to state with an explicit status | `SUPPORTED`, `NOT_SUPPORTED`, `INCONCLUSIVE`, `CONFOUNDED`, `INSUFFICIENT_EVIDENCE` | Later phases (reproducibility, statistics, experiments) | `Claim.status` after evidence evaluation |

## Rules
- A finding never asserts causation. Its text describes what was measured and the threshold.
- Thresholds are configuration, not scientific truth; changing them changes findings, not data.
- An unavailable analysis is reported as unavailable with a reason, never as a number.
- A `Claim` created by the evaluation states only the measurement ("[run …] ECE is 0.13 …")
  and is `SUPPORTED` by the very observation it cites. That is not a conclusion about the model.
- Nothing in EXPERIONYX uses an LLM to word findings or to draw conclusions.

## What Phase 4 provides
**Baseline measurement and forensic observations.** It does not provide causal diagnosis, failure
modes, significance testing, or evidence that a result reproduces. Those require controlled
experiments (fault injection), clustering, and statistical analysis, which are later phases.
