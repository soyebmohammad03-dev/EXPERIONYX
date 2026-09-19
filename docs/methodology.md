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

The deterministic scientific core never depends on a paid API or an LLM. Any LLM use must be
optional and outside that core.

The distinction between observations, findings, interpretations and conclusions is defined in
[observation-vs-conclusion.md](observation-vs-conclusion.md).
