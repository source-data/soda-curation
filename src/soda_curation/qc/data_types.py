from typing import Any, Dict, List

from pydantic import BaseModel


class PanelStatsSignificanceLevel(BaseModel):
    """Significance level analysis for a panel."""

    panel_label: str
    is_a_plot: str  # "yes" or "no"
    significance_level_symbols_on_image: List[str]
    significance_level_symbols_defined_in_caption: List[str]
    from_the_caption: str


class StatsSignificanceLevelResult(BaseModel):
    """Results of significance level analysis for a figure."""

    outputs: List[PanelStatsSignificanceLevel]


class PanelStatsTest(BaseModel):
    """Statistics test analysis for a panel."""

    panel_label: str
    is_a_plot: str  # "yes" or "no"
    statistical_test_needed: str  # "yes" or "no"
    statistical_test_mentioned: str  # "yes", "no", or "not needed"
    justify_why_test_is_missing: str = ""
    from_the_caption: str = ""


class StatsTestResult(BaseModel):
    """Results of statistics test analysis for a figure."""

    outputs: List[PanelStatsTest]


class QCResult(BaseModel):
    """Quality control result for a figure."""

    figure_label: str
    qc_checks: Dict[str, Any] = {}
    qc_status: str = "passed"


class QCPipelineResult(BaseModel):
    """Overall results from the QC pipeline."""

    qc_version: str = "0.1.0"
    qc_status: str = "passed"
    figures_processed: int = 0
    figure_results: List[QCResult] = []


# --- Refactored models for plot_axis_units ---
class AxisUnit(BaseModel):
    axis: str  # e.g., "x" or "y"
    answer: str  # "yes" or "no"


class AxisJustification(BaseModel):
    axis: str
    justification: str


class AxisDefinition(BaseModel):
    axis: str
    definition: str


class PanelPlotAxisUnits(BaseModel):
    """Plot axis units analysis for a panel."""

    panel_label: str
    is_a_plot: str  # "yes" or "no"
    units_provided: List[AxisUnit]
    justify_why_units_are_missing: List[AxisJustification]
    unit_definition_as_provided: List[AxisDefinition]


class PlotAxisUnitsResult(BaseModel):
    """Results of plot axis units analysis for a figure."""

    outputs: List[PanelPlotAxisUnits]


# ReplicatesDefined QC Test
class ReplicatesDefinedPanelResult(BaseModel):
    panel_label: str
    involves_replicates: str  # "yes" or "no"
    number_of_replicates: List[str]
    type_of_replicates: List[str]


class ReplicatesDefinedResult(BaseModel):
    outputs: List[ReplicatesDefinedPanelResult]


# --- Plot Gap Labeling QC Test ---
class AxisGapLabeling(BaseModel):
    axis: str
    answer: str  # "yes", "no", or "not needed"


class AxisGapDescription(BaseModel):
    axis: str
    definition: str


class AxisGapJustification(BaseModel):
    axis: str
    justification: str


class PanelPlotGapLabeling(BaseModel):
    panel_label: str
    is_a_plot: str  # "yes" or "no"
    gaps_defined: List[AxisGapLabeling]
    gap_description: List[AxisGapDescription]
    justify_why_gaps_are_missing: List[AxisGapJustification]


class PlotGapLabelingResult(BaseModel):
    outputs: List[PanelPlotGapLabeling]


# --- Error Bars Defined QC Test ---
class ErrorBarsDefinedPanelResult(BaseModel):
    panel_label: str
    error_bar_on_figure: str  # "yes" or "no"
    error_bar_defined_in_caption: str  # "yes", "no", or "not needed"
    error_bar_definition: str


class ErrorBarsDefinedResult(BaseModel):
    outputs: List[ErrorBarsDefinedPanelResult]


# --- Individual Data Points QC Test ---
class IndividualDataPointsPanelResult(BaseModel):
    panel_label: str
    plot: str  # "yes" or "no"
    average_values: str  # "yes" or "no"
    individual_values: str  # "yes", "no", or "not needed"


class IndividualDataPointsResult(BaseModel):
    outputs: List[IndividualDataPointsPanelResult]


# --- Micrograph Scale Bar QC Test ---
class MicrographScaleBarPanelResult(BaseModel):
    panel_label: str
    micrograph: str  # "yes" or "no"
    scale_bar_on_image: str  # "yes" or "no"
    scale_bar_defined_in_caption: str  # "yes" or "no"
    from_the_caption: str


class MicrographScaleBarResult(BaseModel):
    outputs: List[MicrographScaleBarPanelResult]


# --- Micrograph Symbols Defined QC Test ---
class MicrographSymbolsDefinedPanelResult(BaseModel):
    panel_label: str
    micrograph: str  # "yes" or "no"
    symbols: List[str]
    symbols_defined_in_caption: List[str]
    from_the_caption: List[str]


class MicrographSymbolsDefinedResult(BaseModel):
    outputs: List[MicrographSymbolsDefinedPanelResult]
