# Roadmap

**Provisional; subject to research findings. Not commitments.**

0. Foundation (done)
1. Domain model + experiment registry (done)
2. Execution engine + provenance (done)
3. Model adapter architecture (done)
4. Baseline evaluation / Model Autopsy (done)
5. Fault Injection Laboratory (done)
6. Failure Registry and Failure Discovery (done)
7. Fault Interaction Analysis (done)
8. Reliability Profiles (done)
9. Robustness Benchmark Engine (done)
10. Statistical Analysis & Evidence Engine (done; BCa for interaction contrasts and control-variability modelling remain open)
11. Slice & Subgroup Analysis Engine (done)
12. Temporal & Distribution Shift Engine (done; also the scheduled full-project verification)
13. Data Quality Laboratory (done)
14. Model Stress Laboratory (done)
15. Calibration & Uncertainty Laboratory (done; calibration is measured on stored predictions, not a new score; epistemic/aleatoric decomposition, ensembles and stochastic prediction remain unavailable until an adapter declares them)
16. Resource & Systems Reliability Laboratory (done; also the scheduled full-project verification; timing is measured on the real execution path and stays environment-specific; GPU memory, enforced resource limits and cross-machine variability remain unavailable)
17. Experiment Scheduler & Orchestration (done: a durable, resumable, dependency-aware orchestration layer that plans, dispatches and audits collections of experiments across the existing engines; it never re-implements any of them; concurrency is threads-only and a timeout cannot force-kill a dispatch in flight, see [scheduler.md](scheduler.md))
18. Full-project verification (done; combined with 19, see below)
19. Evidence Graph (done: a queryable, deterministic graph connecting every existing entity through
    typed, versioned relationships derived generically from the registry's own references; edges
    are provenance/evidence metadata, never an inferred causal claim; unresolved references are
    kept as explicit nodes, never dropped; see [graph.md](graph.md))
20. Advanced Reproducibility & Experiment Replay (done: a classification layer over every engine's
    own `replay_check` distinguishing exact/deterministic/tolerance/statistical/provenance-only
    agreement, environment and artifact-integrity checks, and scheduler/graph integration; never
    promises bit-for-bit reproduction the platform cannot guarantee; see
    [reproducibility.md](reproducibility.md)) + Benchmark & Leaderboard System
21. Research validation
22. Advanced UI/API
23. Public release / reproducibility package
