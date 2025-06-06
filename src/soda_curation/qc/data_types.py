"""Data types for QC pipeline."""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class PanelStatsTest:
    """Statistics test analysis for a panel."""

    panel_label: str
    is_a_plot: str  # "yes" or "no"
    statistical_test_needed: str  # "yes" or "no"
    statistical_test_mentioned: str  # "yes", "no", or "not needed"
    justify_why_test_is_missing: str = ""
    from_the_caption: str = ""


@dataclass
class StatsTestResult:
    """Results of statistics test analysis for a figure."""

    outputs: List[PanelStatsTest]


@dataclass
class QCResult:
    """Quality control result for a figure."""

    figure_label: str
    qc_checks: Dict[str, Any] = field(default_factory=dict)
    qc_status: str = "passed"


@dataclass
class QCPipelineResult:
    """Overall results from the QC pipeline."""

    qc_version: str = "0.1.0"
    qc_status: str = "passed"
    figures_processed: int = 0
    figure_results: List[QCResult] = field(default_factory=list)
