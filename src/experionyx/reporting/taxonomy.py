"""Controlled vocabulary for the research reporting layer (see docs/reporting.md).

A report never invents a category: its type and section kinds come from this closed, versioned
vocabulary, and `ClaimStatus`/`EvidenceRelation` are reused from `experionyx.domain` rather than
re-defined, so a finding's evidentiary status means the same thing everywhere in the project.
"""

from enum import StrEnum


class ReportType(StrEnum):
    """What kind of research artifact a report is. Extensible: a new member never changes the
    identity or meaning of an existing report."""

    INVESTIGATION_DOSSIER = "INVESTIGATION_DOSSIER"
    MODEL_RELIABILITY_REPORT = "MODEL_RELIABILITY_REPORT"
    FAILURE_DOSSIER = "FAILURE_DOSSIER"
    BENCHMARK_REPORT = "BENCHMARK_REPORT"
    MODEL_COMPARISON_REPORT = "MODEL_COMPARISON_REPORT"
    REPRODUCIBILITY_REPORT = "REPRODUCIBILITY_REPORT"


class ReportStatus(StrEnum):
    """Whether every section the report's template calls for actually found evidence.
    `INCOMPLETE` is not a failure: gaps are recorded explicitly (see `Report.evidence_gaps`),
    never hidden."""

    GENERATED = "GENERATED"
    INCOMPLETE = "INCOMPLETE"


class SectionKind(StrEnum):
    """The closed set of structural report sections. A template selects a subset and an order;
    the generator includes a section only if evidence exists for it, unless it is one of the
    always-present synthesized sections (summary/limitations/gaps/conclusions)."""

    EXECUTIVE_SUMMARY = "EXECUTIVE_SUMMARY"
    RESEARCH_QUESTION = "RESEARCH_QUESTION"
    EXPERIMENTAL_DESIGN = "EXPERIMENTAL_DESIGN"
    ENVIRONMENT_PROVENANCE = "ENVIRONMENT_PROVENANCE"
    DATASET_DESCRIPTION = "DATASET_DESCRIPTION"
    MODEL_DESCRIPTION = "MODEL_DESCRIPTION"
    METHODS = "METHODS"
    BASELINE_RESULTS = "BASELINE_RESULTS"
    ROBUSTNESS_RESULTS = "ROBUSTNESS_RESULTS"
    FAILURE_FINDINGS = "FAILURE_FINDINGS"
    STATISTICAL_ANALYSIS = "STATISTICAL_ANALYSIS"
    RELIABILITY_EVIDENCE = "RELIABILITY_EVIDENCE"
    DISTRIBUTION_DATA_QUALITY = "DISTRIBUTION_DATA_QUALITY"
    CALIBRATION_UNCERTAINTY = "CALIBRATION_UNCERTAINTY"
    RESOURCE_SYSTEM = "RESOURCE_SYSTEM"
    BENCHMARK_COMPARISON = "BENCHMARK_COMPARISON"
    REPRODUCIBILITY = "REPRODUCIBILITY"
    LIMITATIONS = "LIMITATIONS"
    EVIDENCE_GAPS = "EVIDENCE_GAPS"
    CONCLUSIONS = "CONCLUSIONS"
    REPRODUCIBILITY_APPENDIX = "REPRODUCIBILITY_APPENDIX"


# Sections synthesized from the evidence bundle as a whole (counts, gaps, static caveats) rather
# than from one evidence-source bucket. Always present, regardless of what evidence exists.
SYNTHESIZED_SECTIONS = frozenset(
    {
        SectionKind.EXECUTIVE_SUMMARY,
        SectionKind.RESEARCH_QUESTION,
        SectionKind.METHODS,
        SectionKind.LIMITATIONS,
        SectionKind.EVIDENCE_GAPS,
        SectionKind.CONCLUSIONS,
    }
)

# Section -> the evidence-bundle bucket keys it draws from (see reporting.collector.EVIDENCE_SOURCES).
SECTION_SOURCES: dict[SectionKind, tuple[str, ...]] = {
    SectionKind.EXPERIMENTAL_DESIGN: ("experiment",),
    SectionKind.ENVIRONMENT_PROVENANCE: ("run",),
    SectionKind.DATASET_DESCRIPTION: ("experiment",),
    SectionKind.MODEL_DESCRIPTION: ("experiment",),
    SectionKind.BASELINE_RESULTS: ("run",),
    SectionKind.ROBUSTNESS_RESULTS: ("fault_experiment", "stress_analysis", "interaction_analysis"),
    SectionKind.FAILURE_FINDINGS: ("failure_mode",),
    SectionKind.STATISTICAL_ANALYSIS: ("statistical_analysis",),
    SectionKind.RELIABILITY_EVIDENCE: ("reliability_profile",),
    SectionKind.DISTRIBUTION_DATA_QUALITY: ("drift_analysis", "quality_analysis", "slice_analysis"),
    SectionKind.CALIBRATION_UNCERTAINTY: ("calibration_analysis",),
    SectionKind.RESOURCE_SYSTEM: ("resource_analysis", "schedule_run"),
    SectionKind.BENCHMARK_COMPARISON: (
        "benchmark_result",
        "leaderboard_snapshot",
        "benchmark_submission",
    ),
    SectionKind.REPRODUCIBILITY: ("reproduction_attempt",),
    SectionKind.REPRODUCIBILITY_APPENDIX: ("graph_snapshot", "reproduction_attempt"),
}

# Every evidence-bucket key that the generator knows how to collect. Used to report evidence
# gaps for buckets a template section asked for but that turned up empty.
ALL_SOURCE_KINDS = frozenset(k for keys in SECTION_SOURCES.values() for k in keys)

# The canonical, ordered section list per report type. A future type is one dict entry away.
BUILTIN_TEMPLATE_SECTIONS: dict[ReportType, tuple[SectionKind, ...]] = {
    ReportType.INVESTIGATION_DOSSIER: (
        SectionKind.EXECUTIVE_SUMMARY,
        SectionKind.RESEARCH_QUESTION,
        SectionKind.EXPERIMENTAL_DESIGN,
        SectionKind.ENVIRONMENT_PROVENANCE,
        SectionKind.DATASET_DESCRIPTION,
        SectionKind.MODEL_DESCRIPTION,
        SectionKind.METHODS,
        SectionKind.BASELINE_RESULTS,
        SectionKind.ROBUSTNESS_RESULTS,
        SectionKind.FAILURE_FINDINGS,
        SectionKind.STATISTICAL_ANALYSIS,
        SectionKind.RELIABILITY_EVIDENCE,
        SectionKind.DISTRIBUTION_DATA_QUALITY,
        SectionKind.CALIBRATION_UNCERTAINTY,
        SectionKind.RESOURCE_SYSTEM,
        SectionKind.BENCHMARK_COMPARISON,
        SectionKind.REPRODUCIBILITY,
        SectionKind.LIMITATIONS,
        SectionKind.EVIDENCE_GAPS,
        SectionKind.CONCLUSIONS,
        SectionKind.REPRODUCIBILITY_APPENDIX,
    ),
    ReportType.MODEL_RELIABILITY_REPORT: (
        SectionKind.EXECUTIVE_SUMMARY,
        SectionKind.RESEARCH_QUESTION,
        SectionKind.EXPERIMENTAL_DESIGN,
        SectionKind.ENVIRONMENT_PROVENANCE,
        SectionKind.MODEL_DESCRIPTION,
        SectionKind.RELIABILITY_EVIDENCE,
        SectionKind.CALIBRATION_UNCERTAINTY,
        SectionKind.RESOURCE_SYSTEM,
        SectionKind.STATISTICAL_ANALYSIS,
        SectionKind.LIMITATIONS,
        SectionKind.EVIDENCE_GAPS,
        SectionKind.CONCLUSIONS,
        SectionKind.REPRODUCIBILITY_APPENDIX,
    ),
    ReportType.FAILURE_DOSSIER: (
        SectionKind.EXECUTIVE_SUMMARY,
        SectionKind.RESEARCH_QUESTION,
        SectionKind.EXPERIMENTAL_DESIGN,
        SectionKind.FAILURE_FINDINGS,
        SectionKind.ROBUSTNESS_RESULTS,
        SectionKind.STATISTICAL_ANALYSIS,
        SectionKind.LIMITATIONS,
        SectionKind.EVIDENCE_GAPS,
        SectionKind.CONCLUSIONS,
        SectionKind.REPRODUCIBILITY_APPENDIX,
    ),
    ReportType.BENCHMARK_REPORT: (
        SectionKind.EXECUTIVE_SUMMARY,
        SectionKind.RESEARCH_QUESTION,
        SectionKind.DATASET_DESCRIPTION,
        SectionKind.MODEL_DESCRIPTION,
        SectionKind.METHODS,
        SectionKind.BENCHMARK_COMPARISON,
        SectionKind.STATISTICAL_ANALYSIS,
        SectionKind.LIMITATIONS,
        SectionKind.EVIDENCE_GAPS,
        SectionKind.CONCLUSIONS,
        SectionKind.REPRODUCIBILITY_APPENDIX,
    ),
    ReportType.MODEL_COMPARISON_REPORT: (
        SectionKind.EXECUTIVE_SUMMARY,
        SectionKind.RESEARCH_QUESTION,
        SectionKind.MODEL_DESCRIPTION,
        SectionKind.BENCHMARK_COMPARISON,
        SectionKind.STATISTICAL_ANALYSIS,
        SectionKind.LIMITATIONS,
        SectionKind.EVIDENCE_GAPS,
        SectionKind.CONCLUSIONS,
    ),
    ReportType.REPRODUCIBILITY_REPORT: (
        SectionKind.EXECUTIVE_SUMMARY,
        SectionKind.RESEARCH_QUESTION,
        SectionKind.ENVIRONMENT_PROVENANCE,
        SectionKind.REPRODUCIBILITY,
        SectionKind.REPRODUCIBILITY_APPENDIX,
        SectionKind.LIMITATIONS,
        SectionKind.EVIDENCE_GAPS,
        SectionKind.CONCLUSIONS,
    ),
}

TEMPLATE_ENGINE_VERSION = "1.0.0"  # the section vocabulary/ordering; bump when it changes


__all__ = [
    "ALL_SOURCE_KINDS",
    "BUILTIN_TEMPLATE_SECTIONS",
    "SECTION_SOURCES",
    "SYNTHESIZED_SECTIONS",
    "TEMPLATE_ENGINE_VERSION",
    "ReportStatus",
    "ReportType",
    "SectionKind",
]
