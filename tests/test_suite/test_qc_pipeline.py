# tests/test_suite/test_qc_pipeline.py
import json
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from soda_curation.pipeline.manuscript_structure.manuscript_structure import (
    ZipStructure,
)
from soda_curation.qc.data_models import QCPipelineResult, QCResult
from soda_curation.qc.qc_pipeline import QCPipeline


@pytest.fixture
def mock_zip_structure():
    """Create a mock ZipStructure with test data."""
    zip_structure = MagicMock(spec=ZipStructure)
    zip_structure.figures = [
        {"label": "Figure 1", "caption": "Test caption for Figure 1"},
        {"label": "Figure 2", "caption": "Test caption for Figure 2"},
    ]
    return zip_structure


@pytest.fixture
def test_config():
    """Create a test configuration."""
    return {
        "qc_version": "0.3.1",
        "qc_test_metadata": {
            "individual_data_points": {
                "name": "Individual Data Points",
                "description": "Checks if individual data points are displayed.",
                "prompt_version": 2,
                "checklist_type": "fig-checklist",
            },
            "error_bars_defined": {
                "name": "Error Bars Defined",
                "description": "Checks whether error bars are defined in the figure caption.",
                "prompt_version": 2,
                "checklist_type": "fig-checklist",
            },
        },
        "pipeline": {
            "individual_data_points": {
                "openai": {
                    "prompts": {
                        "system": "System prompt",
                        "user": "User prompt with $figure_caption",
                    }
                }
            },
            "error_bars_defined": {
                "openai": {
                    "prompts": {
                        "system": "System prompt",
                        "user": "User prompt with $figure_caption",
                    }
                }
            },
        },
    }


@pytest.fixture
def figure_data():
    """Create test figure data."""
    return [
        ("Figure 1", "encoded_image_data", "Test caption for Figure 1"),
        ("Figure 2", "encoded_image_data", "Test caption for Figure 2"),
    ]


class TestQCPipeline:
    """Test the QCPipeline class."""

    def test_initialization(self, test_config, tmp_path):
        """Test initialization of QCPipeline."""
        extract_dir = tmp_path / "extract"
        extract_dir.mkdir()

        with patch("importlib.import_module") as mock_import:
            mock_module = MagicMock()
            mock_import.return_value = mock_module
            mock_module.IndividualDataPointsAnalyzer = MagicMock()
            mock_module.ErrorBarsDefinedAnalyzer = MagicMock()

            pipeline = QCPipeline(test_config, extract_dir)

            assert pipeline.config == test_config
            assert pipeline.extract_dir == extract_dir
            assert len(pipeline.tests) > 0

    @patch(
        "soda_curation.qc.qc_tests.individual_data_points.IndividualDataPointsAnalyzer"
    )
    @patch("soda_curation.qc.qc_tests.error_bars_defined.ErrorBarsDefinedAnalyzer")
    def test_run_with_unified_output(
        self,
        mock_error_bars,
        mock_individual_data,
        test_config,
        mock_zip_structure,
        figure_data,
        tmp_path,
    ):
        """Test run method with unified output."""
        extract_dir = tmp_path / "extract"
        extract_dir.mkdir()

        # Mock analyzer instances
        mock_individual_data_instance = MagicMock()
        mock_error_bars_instance = MagicMock()

        # Configure mock analyzers to return test values
        mock_individual_data.return_value = mock_individual_data_instance
        mock_error_bars.return_value = mock_error_bars_instance

        # Setup response for analyze_figure
        mock_result_1 = MagicMock()
        mock_result_1.outputs = [
            MagicMock(panel_label="A", individual_values="yes", plot="yes")
        ]
        mock_individual_data_instance.analyze_figure.return_value = (
            True,
            mock_result_1,
        )

        mock_result_2 = MagicMock()
        mock_result_2.outputs = [
            MagicMock(panel_label="A", error_bar_defined_in_caption="yes")
        ]
        mock_error_bars_instance.analyze_figure.return_value = (True, mock_result_2)

        # Create pipeline and run test
        pipeline = QCPipeline(test_config, extract_dir)
        result = pipeline.run(mock_zip_structure, figure_data)

        # Verify results
        assert "qc_version" in result
        assert "figures" in result
        assert len(result["figures"]) == 2
        assert "panels" in result["figures"][0]
        assert len(result["figures"][0]["panels"]) > 0
        assert "qc_tests" in result["figures"][0]["panels"][0]

        # Verify analyze_figure was called with correct params
        mock_individual_data_instance.analyze_figure.assert_called()
        mock_error_bars_instance.analyze_figure.assert_called()

    @patch(
        "soda_curation.qc.qc_tests.individual_data_points.IndividualDataPointsAnalyzer"
    )
    @patch("soda_curation.qc.qc_tests.error_bars_defined.ErrorBarsDefinedAnalyzer")
    def test_run_legacy_output(
        self,
        mock_error_bars,
        mock_individual_data,
        test_config,
        mock_zip_structure,
        figure_data,
        tmp_path,
    ):
        """Test run method with legacy output."""
        extract_dir = tmp_path / "extract"
        extract_dir.mkdir()

        # Mock analyzer instances
        mock_individual_data_instance = MagicMock()
        mock_error_bars_instance = MagicMock()

        # Configure mock analyzers to return test values
        mock_individual_data.return_value = mock_individual_data_instance
        mock_error_bars.return_value = mock_error_bars_instance

        # Setup response for analyze_figure
        mock_result_1 = MagicMock()
        mock_result_1.outputs = [
            MagicMock(panel_label="A", individual_values="yes", plot="yes")
        ]
        mock_individual_data_instance.analyze_figure.return_value = (
            True,
            mock_result_1,
        )

        mock_result_2 = MagicMock()
        mock_result_2.outputs = [
            MagicMock(panel_label="A", error_bar_defined_in_caption="yes")
        ]
        mock_error_bars_instance.analyze_figure.return_value = (True, mock_result_2)

        # Create pipeline and run test
        pipeline = QCPipeline(test_config, extract_dir)
        result = pipeline.run(mock_zip_structure, figure_data, unified_output=False)

        # Verify results
        assert "qc_version" in result
        assert "qc_status" in result
        assert "figures_processed" in result
        assert "figure_results" in result
        assert len(result["figure_results"]) == 2
        assert "qc_checks" in result["figure_results"][0]

        # Verify analyze_figure was called with correct params
        mock_individual_data_instance.analyze_figure.assert_called()
        mock_error_bars_instance.analyze_figure.assert_called()

    @patch(
        "soda_curation.qc.qc_tests.individual_data_points.IndividualDataPointsAnalyzer"
    )
    def test_run_with_error_handling(
        self,
        mock_individual_data,
        test_config,
        mock_zip_structure,
        figure_data,
        tmp_path,
    ):
        """Test run method with error handling."""
        extract_dir = tmp_path / "extract"
        extract_dir.mkdir()

        # Mock analyzer instances
        mock_individual_data_instance = MagicMock()

        # Configure mock analyzers to raise exception
        mock_individual_data.return_value = mock_individual_data_instance
        mock_individual_data_instance.analyze_figure.side_effect = Exception(
            "Test error"
        )

        # Create pipeline and run test
        pipeline = QCPipeline(test_config, extract_dir)
        result = pipeline.run(mock_zip_structure, figure_data)

        # Verify results handle errors
        assert "qc_version" in result
        assert "figures" in result
        assert len(result["figures"]) == 2

        # Verify analyze_figure was called
        mock_individual_data_instance.analyze_figure.assert_called()
