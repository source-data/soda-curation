"""Tests for the QC main module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from src.soda_curation.qc.main import main


class TestQCMain:
    """Tests for the QC main module functions."""

    @patch("argparse.ArgumentParser.parse_args")
    def test_argument_parsing(self, mock_parse_args):
        """Test argument parsing through the mock."""
        # Create a mock args object with all the expected attributes
        mock_args = MagicMock()
        mock_args.config = "config.yaml"
        mock_args.extract_dir = "extract"
        mock_args.output = "output.json"
        mock_args.zip_structure = "structure.json"
        mock_args.figure_data = "figure_data.json"

        # Make the mock return our predefined args
        mock_parse_args.return_value = mock_args

        # Mock all other functions to avoid actual execution
        with patch("src.soda_curation.qc.main.load_figure_data") as mock_load_figure:
            with patch("src.soda_curation.qc.main.load_zip_structure") as mock_load_zip:
                with patch("src.soda_curation.qc.main.QCPipeline") as mock_pipeline:
                    with patch("builtins.open", MagicMock()):
                        with patch("yaml.safe_load") as mock_yaml_load:
                            with patch("json.dump"):
                                # Mock the external dependencies to prevent actual execution
                                mock_load_figure.return_value = [
                                    ("Figure 1", "base64", "caption")
                                ]
                                mock_load_zip.return_value = MagicMock()
                                mock_pipeline_instance = MagicMock()
                                mock_pipeline.return_value = mock_pipeline_instance
                                mock_pipeline_instance.run.return_value = {
                                    "qc_version": "1.0.0",
                                    "figures": {},
                                }
                                mock_yaml_load.return_value = {"qc_version": "1.0.0"}

                                # Execute main, but it will use our mocked args
                                main()

        # Verify our mock was called (meaning argparse.parse_args was called)
        mock_parse_args.assert_called_once()

        # Verify the args were used correctly
        mock_load_figure.assert_called_once_with(mock_args.figure_data)
        mock_load_zip.assert_called_once_with(mock_args.zip_structure)
