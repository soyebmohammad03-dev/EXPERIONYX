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

The deterministic scientific core never depends on a paid API or an LLM. Any LLM use must be
optional and outside that core.
