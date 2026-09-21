# Methodology

Scientific principles EXPERIONYX is built to enforce. Phase 0 documents them and encodes only the
vocabulary (`experionyx.domain`); enforcement arrives with later phases.

1. **Evidence over assertions.** A claim must be traceable to actual observations.
2. **Reproducibility over convenience.** Capture enough to reproduce a conclusion
   ([reproducibility.md](reproducibility.md)).
3. **Negative results are valid.** A claim resolves to one of `SUPPORTED`, `NOT_SUPPORTED`,
   `INCONCLUSIVE`, `CONFOUNDED`, `INSUFFICIENT_EVIDENCE` (`ClaimStatus`). No conclusion engine
   exists yet.
4. **No fabricated evidence.** Reports never invent values; every number derives from a recorded
   observation.
5. **Statistical humility.** A p-value alone is not sufficient evidence; effect sizes, intervals,
   sample sizes and multiplicity must accompany it.
6. **Explicit uncertainty.** Distinguish observations, derived metrics, interpretations and
   hypotheses (`EpistemicKind`).
7. **Reproducible provenance.** Record model, dataset, configuration, environment, seed, source
   revision and artifacts.
8. **Resource awareness.** Experiments must be designed to run on realistic local hardware.

9. **Discovery is grouping, not explanation.** Failure discovery groups measured signals under
   configured, versioned rules and reports which groups meet configured evidence requirements. A
   discovered pattern is a candidate until it is reproduced and a person confirms it, and it never
   carries a causal claim ([failures.md](failures.md)).

10. **An interaction is a contrast, not a cause.** Interaction analysis reports an additive contrast
    with its uncertainty under an explicit, validated design, keeps observed, derived and interpreted
    values apart, counts the hypotheses it tests, and never converts an observed contrast into a
    causal claim ([interactions.md](interactions.md)).

11. **A profile is evidence, not a grade.** A reliability profile lists what was observed, derived and
    interpreted, with sources, statuses and unresolved issues, and never a score, ranking or verdict
    ([reliability.md](reliability.md)).

12. **A benchmark reports its coverage.** A benchmark is a versioned protocol whose result lists what was
    and was not executed. Missing coverage is never robustness, results are comparable only under an
    identical protocol, and no score, ranking or verdict is computed ([benchmarks.md](benchmarks.md)).

13. **A p-value is not a finding.** Statistical results state their method, assumptions, uncertainty
    and any correction, keep raw observations reachable, and are reproducible from persisted evidence;
    a p-value or an interval is never rendered as practical importance or causation
    ([statistics.md](statistics.md)).

14. **A slice is a population, not a verdict.** Slice results are per-population observations with counts and
    uncertainty. Missing metadata is never a match, thin slices are flagged, no group is ranked or scored,
    and nothing says why a slice behaves differently ([slices.md](slices.md)).

15. **A distribution change is an observation, not an explanation.** Drift results compare explicit
    windows under a declared ordering, keep magnitude and statistical evidence apart, correct for many
    simultaneous comparisons on request, and never claim that a change caused a failure or is concept
    drift ([drift.md](drift.md)).

16. **Data quality is multidimensional evidence, not a score.** Checks report a status per scope under
    configured expectations; warnings need interpretation, statistical anomalies are not data errors,
    leakage indicators are not proof, missingness can be informative and class imbalance is
    context-dependent ([data-quality.md](data-quality.md)).

18. **Confidence is not uncertainty, and calibration is not a score.** A model's probability is
    compared with observed correctness on stored predictions, separately for the top label, each class
    and the whole vector; ECE depends on the binning and the sample; post-hoc calibration is fitted only
    on data disjoint from what it is evaluated on; predictive uncertainty is reported only where the model
    supports it and is otherwise `UNAVAILABLE`; no universal calibration or uncertainty score exists
    ([calibration.md](calibration.md)).

19. **A timing is an environment-specific measurement, not a property of the model.** Latency, throughput,
    CPU time and peak memory are measured by really running the model, from repeated warmed-up trials whose raw
    values are kept; initialization, warmup, steady state and end to end are separate numbers; a timeout or a
    failure is never a success; a measurement the platform or the adapter cannot provide is `UNAVAILABLE`;
    repeated trials on one machine are not independent samples of a hardware population; performance
    measurement is not reliability, and no composite resource score exists ([resources.md](resources.md)).

17. **A stress is an experiment, not a score.** A stress trial deliberately changes one recorded thing,
    is compared with a validated baseline, keeps raw values, direction-aware deterioration and paired
    statistics, and never yields a robustness score or a causal claim; a lack of stress evidence is
    never "safe" ([stress.md](stress.md)).

The deterministic scientific core never depends on a paid API or an LLM. Any LLM use must be
optional and outside that core.

The distinction between observations, findings, interpretations and conclusions is defined in
[observation-vs-conclusion.md](observation-vs-conclusion.md).
