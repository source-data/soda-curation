"""Tests for QC pipeline module."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.soda_curation.qc.data_types import QCPipelineResult, QCResult
from src.soda_curation.qc.qc_pipeline import QCPipeline


@pytest.fixture
def config():
    """Return a test configuration."""
    return {
        "pipeline": {
            "stats_test": {
                "openai": {
                    "model": "gpt-4o",
                    "temperature": 0.1,
                    "prompts": {
                        "system": "Test system prompt",
                        "user": "Test user prompt",
                    },
                }
            }
        }
    }


@pytest.fixture
def figure_data():
    """Return sample figure data."""
    return [
        ("Figure 1", "base64encodedimage1", "Caption for Figure 1"),
        ("Figure 2", "base64encodedimage2", "Caption for Figure 2"),
    ]


@pytest.fixture
def zip_structure():
    """Return a mock zip structure."""
    mock = MagicMock()
    mock.figures = []
    return mock


@pytest.fixture
def extract_dir():
    """Return a temporary directory for extraction."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


# Fix 1: Patch the actual module import in QCPipeline
@patch("importlib.import_module")
def test_initialize_tests(mock_import_module, config, extract_dir):
    """Test that QC pipeline correctly initializes test modules."""
    # Set up mock module and class
    mock_module = MagicMock()
    mock_analyzer_class = MagicMock()
    mock_analyzer_instance = MagicMock()
    mock_analyzer_class.return_value = mock_analyzer_instance
    mock_module.StatsTestAnalyzer = mock_analyzer_class

    # Configure import_module to return our mock module
    def side_effect(name, package):
        if name == ".qc_tests.stats_test":
            return mock_module
        raise ModuleNotFoundError(f"No module named '{name}'")

    mock_import_module.side_effect = side_effect

    # Initialize QC pipeline
    qc_pipeline = QCPipeline(config, extract_dir)

    # Assert that the correct module was imported
    mock_import_module.assert_any_call(
        ".qc_tests.stats_test", package="soda_curation.qc"
    )

    # Assert that StatsTestAnalyzer was initialized correctly
    mock_analyzer_class.assert_called_once_with(config)
    assert "stats_test" in qc_pipeline.tests
    assert qc_pipeline.tests["stats_test"] == mock_analyzer_instance


@patch("importlib.import_module")
def test_qc_pipeline_run(
    mock_import_module, config, extract_dir, zip_structure, figure_data
):
    """Test QC pipeline run with mock analyzers."""
    # Set up mock module and class
    mock_module = MagicMock()
    mock_analyzer_class = MagicMock()
    mock_analyzer_instance = MagicMock()
    mock_analyzer_class.return_value = mock_analyzer_instance
    mock_module.StatsTestAnalyzer = mock_analyzer_class

    # Configure import_module to return our mock module
    def side_effect(name, package):
        if name == ".qc_tests.stats_test":
            return mock_module
        raise ModuleNotFoundError(f"No module named '{name}'")

    mock_import_module.side_effect = side_effect

    # Use a dictionary instead of trying to create a PanelStatsTest instance directly
    stats_result = {
        "outputs": [
            {
                "panel_label": "A",
                "is_a_plot": "yes",
                "statistical_test_needed": "yes",
                "statistical_test_mentioned": "yes",
                "justify_why_test_is_missing": "",
                "from_the_caption": "Statistical significance was tested using t-test.",
            }
        ]
    }

    # Set up the mock to return a tuple of (passed, result)
    mock_analyzer_instance.analyze_figure.return_value = (True, stats_result)

    # Initialize and run QC pipeline
    qc_pipeline = QCPipeline(config, extract_dir)
    results = qc_pipeline.run(zip_structure, figure_data)

    # Assert that the analyzer was called correctly
    assert mock_analyzer_instance.analyze_figure.call_count == 2

    # Verify results structure
    assert "qc_status" in results
    assert "figures_processed" in results
    assert results["figures_processed"] == 2
    assert "figure_results" in results
    assert len(results["figure_results"]) == 2


def test_save_and_load_results(extract_dir):
    """Test saving and loading QC results."""
    # Create a simple result using dictionaries
    qc_result = QCResult(
        figure_label="Figure 1",
        qc_checks={
            "stats_test": {
                "passed": True,
                "result": {
                    "outputs": [
                        {
                            "panel_label": "A",
                            "is_a_plot": "yes",
                            "statistical_test_needed": "yes",
                            "statistical_test_mentioned": "yes",
                            "justify_why_test_is_missing": "",
                            "from_the_caption": "Test was performed.",
                        }
                    ]
                },
            }
        },
        qc_status="passed",
    )

    pipeline_result = QCPipelineResult(
        qc_version="0.1.0",
        qc_status="passed",
        figures_processed=1,
        figure_results=[qc_result],
    )

    # Save to a temporary file
    output_file = os.path.join(extract_dir, "test_results.json")
    with open(output_file, "w") as f:
        json.dump(
            vars(pipeline_result),
            f,
            default=lambda o: vars(o) if hasattr(o, "__dict__") else o,
        )

    # Read it back
    with open(output_file, "r") as f:
        loaded_results = json.load(f)

    # Verify data was preserved
    assert loaded_results["qc_status"] == "passed"
    assert loaded_results["figures_processed"] == 1
    assert len(loaded_results["figure_results"]) == 1
    assert loaded_results["figure_results"][0]["figure_label"] == "Figure 1"


@patch("importlib.import_module")
def test_dynamic_module_loading(mock_import_module, extract_dir):
    """Test dynamic loading of test modules."""
    # Create config with multiple test modules
    config = {
        "pipeline": {
            "stats_test": {"openai": {}},
            "data_test": {"openai": {}},
            "invalid_test": {"openai": {}},
        }
    }

    # Set up mock modules
    mock_stats_module = MagicMock()
    mock_stats_analyzer = MagicMock()
    mock_stats_module.StatsTestAnalyzer = mock_stats_analyzer

    mock_data_module = MagicMock()
    mock_data_analyzer = MagicMock()
    mock_data_module.DataTestAnalyzer = mock_data_analyzer

    # Configure import_module mock
    def side_effect(name, package):
        if name == ".qc_tests.stats_test":
            return mock_stats_module
        elif name == ".qc_tests.data_test":
            return mock_data_module
        else:
            raise ModuleNotFoundError(f"No module named '{name}'")

    mock_import_module.side_effect = side_effect

    # Initialize QC pipeline
    qc_pipeline = QCPipeline(config, extract_dir)

    # Verify correct modules were loaded
    assert len(qc_pipeline.tests) == 2
    assert "stats_test" in qc_pipeline.tests
    assert "data_test" in qc_pipeline.tests
    assert "invalid_test" not in qc_pipeline.tests
