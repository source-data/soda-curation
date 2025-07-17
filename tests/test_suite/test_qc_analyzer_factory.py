# tests/test_qc_analyzer_factory.py
from typing import Any, Dict, Type
from unittest.mock import patch

import pytest

from soda_curation.qc.qc_tests.error_bars_defined import ErrorBarsDefinedAnalyzer
from soda_curation.qc.qc_tests.individual_data_points import (
    IndividualDataPointsAnalyzer,
)
from soda_curation.qc.qc_tests.micrograph_scale_bar import MicrographScaleBarAnalyzer
from soda_curation.qc.qc_tests.micrograph_symbols_defined import (
    MicrographSymbolsDefinedAnalyzer,
)
from soda_curation.qc.qc_tests.plot_axis_units import PlotAxisUnitsAnalyzer
from soda_curation.qc.qc_tests.plot_gap_labeling import PlotGapLabelingAnalyzer
from soda_curation.qc.qc_tests.replicates_defined import ReplicatesDefinedAnalyzer
from soda_curation.qc.qc_tests.stat_significance_level import (
    StatsSignificanceLevelAnalyzer,
)
from soda_curation.qc.qc_tests.stat_test import StatsTestAnalyzer


# First, let's create a factory class if it doesn't exist
class QCAnalyzerFactory:
    """Factory for creating QC analyzers based on test names."""

    _analyzers = {
        "individual_data_points": IndividualDataPointsAnalyzer,
        "error_bars_defined": ErrorBarsDefinedAnalyzer,
        "micrograph_scale_bar": MicrographScaleBarAnalyzer,
        "micrograph_symbols_defined": MicrographSymbolsDefinedAnalyzer,
        "plot_axis_units": PlotAxisUnitsAnalyzer,
        "plot_gap_labeling": PlotGapLabelingAnalyzer,
        "replicates_defined": ReplicatesDefinedAnalyzer,
        "stat_significance_level": StatsSignificanceLevelAnalyzer,
        "stat_test": StatsTestAnalyzer,
    }

    @classmethod
    def create_analyzer(cls, test_name: str, config: Dict[str, Any]):
        """Create an analyzer for the given test name."""
        if test_name not in cls._analyzers:
            raise ValueError(f"Unknown test name: {test_name}")
        return cls._analyzers[test_name](config)


def test_analyzer_factory():
    config = {"model": "test_model"}

    # Create a sample schema that will be returned by get_schema
    sample_schema = {
        "type": "object",
        "properties": {
            "outputs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"panel_label": {"type": "string"}},
                },
            }
        },
    }

    # Create a simple mock model
    class MockModel:
        def model_validate_json(self, json_str):
            return "validated_model"

    # Test all analyzers
    test_cases = [
        ("individual_data_points", IndividualDataPointsAnalyzer),
        ("error_bars_defined", ErrorBarsDefinedAnalyzer),
        ("micrograph_scale_bar", MicrographScaleBarAnalyzer),
        ("micrograph_symbols_defined", MicrographSymbolsDefinedAnalyzer),
        ("plot_axis_units", PlotAxisUnitsAnalyzer),
        ("plot_gap_labeling", PlotGapLabelingAnalyzer),
        ("replicates_defined", ReplicatesDefinedAnalyzer),
        ("stat_significance_level", StatsSignificanceLevelAnalyzer),
        ("stat_test", StatsTestAnalyzer),
    ]

    # Mock all the registry methods used by analyzers
    with patch(
        "soda_curation.qc.prompt_registry.registry.get_schema",
        return_value=sample_schema,
    ):
        with patch(
            "soda_curation.qc.prompt_registry.registry.get_pydantic_model",
            return_value=MockModel,
        ):
            with patch("soda_curation.qc.prompt_registry.registry.get_prompt_metadata"):
                with patch(
                    "soda_curation.qc.prompt_registry.registry.get_prompt",
                    return_value="Test prompt",
                ):
                    for test_name, analyzer_class in test_cases:
                        analyzer = QCAnalyzerFactory.create_analyzer(test_name, config)
                        assert isinstance(analyzer, analyzer_class)
