"""Integration tests for the QC pipeline."""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.soda_curation.pipeline.manuscript_structure.manuscript_structure import (
    ZipStructure,
)
from src.soda_curation.qc.analyzer_factory import AnalyzerFactory
from src.soda_curation.qc.main import main
from src.soda_curation.qc.prompt_registry import registry
from src.soda_curation.qc.qc_pipeline import QCPipeline


class TestQCIntegration:
    """Integration tests for the QC pipeline."""

    @pytest.fixture
    def mock_zip_structure(self):
        """Create a mock zip structure with test figures."""
        mock_zip = MagicMock(spec=ZipStructure)
        # Create sample figure data
        figure1 = MagicMock()
        figure1.figure_label = "Figure 1"
        figure1.figure_caption = "This is a test caption for Figure 1"
        figure1.encoded_image = "base64encodedimage1"

        figure2 = MagicMock()
        figure2.figure_label = "Figure 2"
        figure2.figure_caption = "This is a test caption for Figure 2"
        figure2.encoded_image = "base64encodedimage2"

        mock_zip.figures = [figure1, figure2]
        return mock_zip

    @pytest.fixture
    def mock_figure_data(self):
        """Create mock figure data in the format expected by the pipeline."""
        return [
            ("Figure 1", "base64encodedimage1", "This is a test caption for Figure 1"),
            ("Figure 2", "base64encodedimage2", "This is a test caption for Figure 2"),
        ]

    @patch("src.soda_curation.qc.model_api.ModelAPI.generate_response")
    @patch("src.soda_curation.qc.prompt_registry.registry.get_prompt")
    def test_qc_pipeline_run(
        self,
        mock_get_prompt,
        mock_generate_response,
        mock_zip_structure,
        mock_figure_data,
    ):
        """Test the QC pipeline run method."""
        # Setup mocks
        mock_get_prompt.return_value = "test prompt"
        mock_generate_response.return_value = {
            "outputs": [
                {
                    "panel_label": "A",
                    "error_bar_on_figure": "no",
                    "error_bar_defined_in_caption": "not needed",
                },
                {
                    "panel_label": "B",
                    "error_bar_on_figure": "yes",
                    "error_bar_defined_in_caption": "yes",
                },
            ]
        }

        # Create test config
        config = {
            "qc_version": "1.0.0",
            "qc_test_metadata": {
                "panel": {
                    "error_bars_defined": {
                        "name": "Error Bars Defined",
                        "description": "Checks if error bars are defined in the figure.",
                        "permalink": "https://example.com/error_bars_defined",
                    }
                }
            },
            "default": {
                "pipeline": {"error_bars_defined": {"openai": {"model": "gpt-4o"}}}
            },
        }

        # Initialize pipeline
        qc_pipeline = QCPipeline(config, "extract_dir")

        # Run pipeline
        results = qc_pipeline.run(mock_zip_structure, mock_figure_data)

        # Verify results
        assert "qc_version" in results
        assert results["qc_version"] == "1.0.0"
        assert "figures" in results
        assert len(results["figures"]) > 0
        assert "qc_test_metadata" in results

        # Figure keys are normalized to lowercase with underscores
        assert "figure_1" in results["figures"]
        assert "panels" in results["figures"]["figure_1"]

    @patch("src.soda_curation.qc.main.load_zip_structure")
    @patch("src.soda_curation.qc.main.load_figure_data")
    @patch("src.soda_curation.qc.main.QCPipeline")
    @patch("yaml.safe_load")
    def test_main_function(
        self,
        mock_yaml_load,
        mock_qc_pipeline,
        mock_load_figure_data,
        mock_load_zip_structure,
        mock_zip_structure,
        mock_figure_data,
        tmp_path,
    ):
        """Test the main function configuration loading and pipeline setup."""
        # Setup mocks - ensure both figure_data and zip_structure are returned properly
        mock_load_zip_structure.return_value = mock_zip_structure
        mock_load_figure_data.return_value = mock_figure_data

        # Mock the QCPipeline instance and its run method
        mock_pipeline_instance = MagicMock()
        mock_qc_pipeline.return_value = mock_pipeline_instance
        mock_pipeline_instance.run.return_value = {
            "qc_version": "1.0.0",
            "figures": {"figure_1": {"panels": []}},
            "qc_test_metadata": {"error_bars_defined": {}},
        }

        # Mock the YAML config loading
        test_config = {
            "qc_version": "1.0.0",
            "qc_test_metadata": {
                "panel": {
                    "error_bars_defined": {
                        "name": "Error Bars Defined",
                        "description": "Test description",
                        "permalink": "https://example.com/test",
                    }
                }
            },
            "default": {
                "pipeline": {"error_bars_defined": {"openai": {"model": "gpt-4o"}}}
            },
        }
        mock_yaml_load.return_value = test_config

        # Create output file path
        output_file = tmp_path / "qc_results.json"

        # Add a fake zip_structure file and figure_data file path for command line args
        zip_structure_file = tmp_path / "dummy_zip_structure.json"
        figure_data_file = tmp_path / "dummy_figure_data.json"

        # Mock file operations
        with patch("builtins.open", MagicMock()):
            with patch("json.dump") as mock_json_dump:
                # Run main with complete arguments including zip_structure and figure_data
                with patch(
                    "sys.argv",
                    [
                        "qc_main.py",
                        "--config",
                        "dummy_config.yaml",
                        "--extract-dir",
                        str(tmp_path),
                        "--output",
                        str(output_file),
                        "--zip-structure",
                        str(zip_structure_file),
                        "--figure-data",
                        str(figure_data_file),
                    ],
                ):
                    main()

        # Verify the pipeline was created with the right parameters
        mock_qc_pipeline.assert_called_once()
        # Verify the run method was called
        mock_pipeline_instance.run.assert_called_once()
        # Verify results were saved
        mock_json_dump.assert_called_once()
