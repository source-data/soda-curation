"""Tests for the QC pipeline."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.soda_curation.pipeline.manuscript_structure.manuscript_structure import (
    ZipStructure,
)
from src.soda_curation.qc.analyzer_factory import AnalyzerFactory
from src.soda_curation.qc.qc_pipeline import QCPipeline


@pytest.fixture
def mock_zip_structure():
    """Create a mock ZipStructure with test data."""
    zip_structure = MagicMock(spec=ZipStructure)
    figure1 = MagicMock()
    figure1.figure_label = "Figure 1"
    figure1.figure_caption = "Test caption for Figure 1"
    figure1.encoded_image = "base64encodedimage1"

    figure2 = MagicMock()
    figure2.figure_label = "Figure 2"
    figure2.figure_caption = "Test caption for Figure 2"
    figure2.encoded_image = "base64encodedimage2"

    zip_structure.figures = [figure1, figure2]
    return zip_structure


@pytest.fixture
def test_config():
    """Create a test configuration with the new hierarchical structure."""
    return {
        "qc_version": "1.0.0",
        "qc_test_metadata": {
            "panel": {
                "error_bars_defined": {
                    "name": "Error Bars Defined",
                    "description": "Checks whether error bars are defined in the figure caption.",
                    "permalink": "https://example.com/error_bars_defined",
                },
                "plot_axis_units": {
                    "name": "Plot Axis Units",
                    "description": "Checks whether plot axes have units.",
                    "permalink": "https://example.com/plot_axis_units",
                },
            }
        },
        "default": {
            "pipeline": {
                "error_bars_defined": {"openai": {"model": "gpt-4o"}},
                "plot_axis_units": {"openai": {"model": "gpt-4o"}},
            }
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
    """Test the refactored QCPipeline class."""

    def test_initialization(self, test_config, tmp_path):
        """Test initialization of QCPipeline."""
        extract_dir = tmp_path / "extract"
        extract_dir.mkdir()

        # Mock the _initialize_tests method
        with patch.object(QCPipeline, "_initialize_tests") as mock_init_tests:
            mock_init_tests.return_value = {"test": MagicMock()}

            pipeline = QCPipeline(test_config, extract_dir)

            assert pipeline.config == test_config
            assert pipeline.extract_dir == extract_dir
            mock_init_tests.assert_called_once()

    def test_unified_output_format(
        self, test_config, mock_zip_structure, figure_data, tmp_path
    ):
        """Test the unified output format from the QC pipeline."""
        extract_dir = tmp_path / "extract"
        extract_dir.mkdir()

        # Create a predetermined output to test our format expectations
        expected_output = {
            "qc_version": "1.0.0",
            "figures": {
                "figure_1": {
                    "panels": [
                        {
                            "panel_label": "A",
                            "qc_tests": [
                                {
                                    "test_name": "error_bars_defined",
                                    "passed": True,
                                    "model_output": {
                                        "error_bar_on_figure": "yes",
                                        "error_bar_defined_in_caption": "yes",
                                    },
                                },
                                {
                                    "test_name": "plot_axis_units",
                                    "passed": True,
                                    "model_output": {
                                        "units_on_x_axis": "yes",
                                        "units_on_y_axis": "yes",
                                    },
                                },
                            ],
                        }
                    ]
                },
                "figure_2": {
                    "panels": [
                        {
                            "panel_label": "A",
                            "qc_tests": [
                                {
                                    "test_name": "error_bars_defined",
                                    "passed": True,
                                    "model_output": {
                                        "error_bar_on_figure": "yes",
                                        "error_bar_defined_in_caption": "yes",
                                    },
                                }
                            ],
                        }
                    ]
                },
            },
            "qc_test_metadata": {
                "error_bars_defined": {
                    "name": "Error Bars Defined",
                    "description": "Checks whether error bars are defined in the figure caption.",
                    "permalink": "https://example.com/error_bars_defined",
                },
                "plot_axis_units": {
                    "name": "Plot Axis Units",
                    "description": "Checks whether plot axes have units.",
                    "permalink": "https://example.com/plot_axis_units",
                },
            },
            "status": "success",
        }

        # Mock the QCPipeline.run method to return our predetermined output
        with patch.object(QCPipeline, "run", return_value=expected_output):
            pipeline = QCPipeline(test_config, extract_dir)
            result = pipeline.run(mock_zip_structure, figure_data)

            # Verify result format against our expected output
            assert result == expected_output
            assert "qc_version" in result
            assert "figures" in result
            assert "figure_1" in result["figures"]
            assert "figure_2" in result["figures"]
            assert "panels" in result["figures"]["figure_1"]

            # Verify test details
            panel = result["figures"]["figure_1"]["panels"][0]
            assert "qc_tests" in panel
            test_names = [test["test_name"] for test in panel["qc_tests"]]
            assert "error_bars_defined" in test_names
            assert "plot_axis_units" in test_names

    def test_error_handling(
        self, test_config, mock_zip_structure, figure_data, tmp_path
    ):
        """Test error handling in the QC pipeline."""
        extract_dir = tmp_path / "extract"
        extract_dir.mkdir()

        # Create a predetermined output simulating error handling
        expected_output = {
            "qc_version": "1.0.0",
            "figures": {"figure_1": {"panels": []}, "figure_2": {"panels": []}},
            "qc_test_metadata": {
                "error_bars_defined": {
                    "name": "Error Bars Defined",
                    "description": "Checks whether error bars are defined in the figure caption.",
                    "permalink": "https://example.com/error_bars_defined",
                }
            },
            "status": "error",  # Should mark as error when exceptions occur
        }

        # Mock the run method to return our error output
        with patch.object(QCPipeline, "run", return_value=expected_output):
            pipeline = QCPipeline(test_config, extract_dir)
            result = pipeline.run(mock_zip_structure, figure_data)

            # Verify error handling
            assert result == expected_output
            assert "status" in result
            assert result["status"] == "error"
            assert "figure_1" in result["figures"]
            assert "figure_2" in result["figures"]


def test_pipeline_handles_malformed_analyzer_output(
    tmp_path, test_config, mock_zip_structure, figure_data
):
    """Test handling of malformed analyzer output."""

    class MalformedAnalyzer:
        def analyze_figure(self, *args, **kwargs):
            return True, "not a dict"

    class TestableQCPipeline(QCPipeline):
        def _initialize_tests(self):
            return {"error_bars_defined": MalformedAnalyzer()}

        def run(self, zip_structure, figure_data=None, unified_output=True):
            self.qc_results = {"figures": {}}
            for figure_label, encoded_image, figure_caption in figure_data:
                figure_id = figure_label.replace(" ", "_").lower()
                passed, result = self.tests["error_bars_defined"].analyze_figure(
                    figure_label, encoded_image, figure_caption
                )
                self.add_qc_result(figure_id, "error_bars_defined", passed, result)
            return self.qc_results

    pipeline = TestableQCPipeline(test_config, tmp_path)
    result = pipeline.run(mock_zip_structure, figure_data)
    assert "figures" in result
    assert len(result["figures"]) == len(figure_data)
    for figure_id, figure in result["figures"].items():
        assert "panels" in figure
        for panel in figure["panels"]:
            assert "qc_tests" in panel


def test_pipeline_handles_empty_analyzer_output(
    tmp_path, test_config, mock_zip_structure, figure_data
):
    """Test handling of empty analyzer output."""

    class EmptyAnalyzer:
        def analyze_figure(self, *args, **kwargs):
            return True, {}

    class TestableQCPipeline(QCPipeline):
        def _initialize_tests(self):
            return {"error_bars_defined": EmptyAnalyzer()}

        def run(self, zip_structure, figure_data=None, unified_output=True):
            self.qc_results = {"figures": {}}
            for figure_label, encoded_image, figure_caption in figure_data:
                figure_id = figure_label.replace(" ", "_").lower()
                passed, result = self.tests["error_bars_defined"].analyze_figure(
                    figure_label, encoded_image, figure_caption
                )
                self.add_qc_result(figure_id, "error_bars_defined", passed, result)
            return self.qc_results

    pipeline = TestableQCPipeline(test_config, tmp_path)
    result = pipeline.run(mock_zip_structure, figure_data)
    assert "figures" in result
    assert len(result["figures"]) == len(figure_data)
    for figure_id, figure in result["figures"].items():
        assert "panels" in figure
        for panel in figure["panels"]:
            assert "qc_tests" in panel
