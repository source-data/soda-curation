from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class QCResult:
    """Result of a QC analysis for a single figure."""

    figure_label: str
    qc_checks: Dict[str, Any] = field(default_factory=dict)
    qc_status: str = "pending"


@dataclass
class QCPipelineResult:
    """Result of running the entire QC pipeline."""

    qc_version: str
    qc_status: str
    figures_processed: int
    figure_results: List[QCResult] = field(default_factory=list)
