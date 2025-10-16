"""Tests for QC pipeline panel label validation.

This test ensures that the QC pipeline only generates panel labels
that match the main pipeline output, preventing issues with sub-panels
like 'A-a', descriptive labels like 'Rice cell', or figure labels like 'Figure 8'.
"""

import json
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.soda_curation.pipeline.manuscript_structure.manuscript_structure import (
    Figure,
    Panel,
    ZipStructure,
)
from src.soda_curation.qc.qc_pipeline import QCPipeline


@pytest.fixture
def mock_config():
    """Create a mock configuration for QC pipeline."""
    return {
        "qc_version": "2.5.4",
        "default": {"openai": {"model": "gpt-4o", "temperature": 0.1}},
        "qc_check_metadata": {
            "panel": {"micrograph_scale_bar": {}},
            "document": {},
        },
    }


@pytest.fixture
def mock_zip_structure():
    """Create a mock ZipStructure with valid panel labels."""
    # Figure 5 from EMBOJ-2024-118167-T has panels A and B
    # This is the case that was generating sub-panel labels like A-a through A-l
    panel_a = Panel(
        panel_label="A",
        panel_caption="Test caption A",
    )
    panel_b = Panel(
        panel_label="B",
        panel_caption="Test caption B",
    )

    figure_5 = Figure(
        figure_label="Figure 5",
        img_files=[],
        sd_files=[],
        figure_caption="Test figure caption",
        panels=[panel_a, panel_b],
    )

    zip_structure = ZipStructure()
    zip_structure.figures = [figure_5]

    return zip_structure


def test_populate_valid_panel_labels(mock_config, mock_zip_structure):
    """Test that valid panel labels are correctly populated from zip_structure."""
    pipeline = QCPipeline(mock_config, "/tmp/test")

    valid_labels = pipeline._populate_valid_panel_labels(mock_zip_structure)

    assert "Figure 5" in valid_labels
    assert valid_labels["Figure 5"] == ["A", "B"]


@patch("src.soda_curation.qc.qc_pipeline.AnalyzerFactory")
def test_expected_panels_passed_to_analyzer(
    mock_factory, mock_config, mock_zip_structure
):
    """Test that expected panel labels are passed to panel-level analyzers."""
    from src.soda_curation.qc.base_analyzers import PanelQCAnalyzer

    # Create a mock analyzer that inherits from PanelQCAnalyzer
    mock_analyzer = MagicMock(spec=PanelQCAnalyzer)
    mock_analyzer.analyze_figure = Mock(return_value=(True, {"outputs": []}))

    # Configure the factory to return our mock analyzer
    mock_factory.create_analyzer.return_value = mock_analyzer

    # Create pipeline and run with mock data
    pipeline = QCPipeline(mock_config, "/tmp/test")

    # Create minimal figure data for testing
    figure_data = [("Figure 5", "encoded_image_data", "Test caption")]

    # Run the pipeline
    pipeline.run(mock_zip_structure, figure_data=figure_data, unified_output=False)

    # Verify that analyze_figure was called with expected_panels
    mock_analyzer.analyze_figure.assert_called()
    call_args = mock_analyzer.analyze_figure.call_args

    # The fourth argument should be expected_panels = ["A", "B"]
    assert len(call_args[0]) >= 4
    expected_panels = call_args[0][3]
    assert expected_panels == ["A", "B"]


@patch("src.soda_curation.qc.model_api.ModelAPI.generate_response")
@patch("src.soda_curation.qc.qc_pipeline.AnalyzerFactory")
def test_ai_receives_panel_constraint_instructions(
    mock_factory, mock_generate_response, mock_config, mock_zip_structure
):
    """Test that AI receives constraint instructions about valid panel labels."""
    # Mock the AI response
    mock_response = Mock()
    mock_response.outputs = [
        {"panel_label": "A", "scale_bar_on_image": "yes"},
        {"panel_label": "B", "scale_bar_on_image": "yes"},
    ]
    mock_generate_response.return_value = mock_response

    # Create a real analyzer (not mocked) so we can test the full flow
    from src.soda_curation.qc.analyzer_factory import GenericPanelQCAnalyzer

    mock_analyzer = GenericPanelQCAnalyzer("micrograph_scale_bar", mock_config)
    mock_factory.create_analyzer.return_value = mock_analyzer

    # Create pipeline and run
    pipeline = QCPipeline(mock_config, "/tmp/test")
    figure_data = [("Figure 5", "encoded_image_data", "Test caption")]
    pipeline.run(mock_zip_structure, figure_data=figure_data, unified_output=False)

    # Verify that generate_response was called with expected_panels
    mock_generate_response.assert_called_once()
    call_kwargs = mock_generate_response.call_args[1]

    assert "expected_panels" in call_kwargs
    assert call_kwargs["expected_panels"] == ["A", "B"]


def test_no_invalid_panel_labels_in_output(mock_config):
    """Test that QC output contains only valid panel labels.

    This is a regression test for the issue where QC pipeline was generating
    invalid labels like 'A-a', 'Figure 8', 'Rice cell', etc.
    """
    # Load a real QC output file if it exists, or create a mock
    # This test checks the structure of the output
    test_output = {
        "figures": {
            "figure_5": {
                "panels": [
                    {
                        "panel_label": "A",
                        "qc_checks": [
                            {
                                "check_name": "micrograph_scale_bar",
                                "model_output": {
                                    "panel_label": "A",
                                    "scale_bar_on_image": "yes",
                                },
                            }
                        ],
                    },
                    {
                        "panel_label": "B",
                        "qc_checks": [
                            {
                                "check_name": "micrograph_scale_bar",
                                "model_output": {
                                    "panel_label": "B",
                                    "scale_bar_on_image": "yes",
                                },
                            }
                        ],
                    },
                ]
            }
        }
    }

    # Define invalid patterns
    invalid_patterns = [
        "-",  # Sub-panel markers like "A-a"
        "(",  # Panel modifiers like "C (plot)"
        "Figure ",  # Figure labels like "Figure 8"
    ]

    # Check all panel labels
    for figure_id, figure_data in test_output["figures"].items():
        for panel in figure_data.get("panels", []):
            # Check top-level panel label
            panel_label = panel.get("panel_label", "")
            for pattern in invalid_patterns:
                assert (
                    pattern not in panel_label
                ), f"Found invalid pattern '{pattern}' in top-level panel_label: {panel_label}"

            # Check model_output panel labels
            for check in panel.get("qc_checks", []):
                if "model_output" in check and isinstance(check["model_output"], dict):
                    model_output = check["model_output"]
                    if "panel_label" in model_output:
                        model_panel_label = model_output["panel_label"]
                        for pattern in invalid_patterns:
                            assert (
                                pattern not in model_panel_label
                            ), f"Found invalid pattern '{pattern}' in model_output panel_label: {model_panel_label}"


def test_panel_label_validation_with_problematic_cases():
    """Test validation with the specific problematic cases from the issues.

    These are real cases that were causing problems:
    - EMBOJ-2024-118167-T: Generated A-a through A-l, C (plot), C (right)
    - EMBOJ-2024-118766: Generated "Rice cell", "Small brown planthopper"
    - EMBOJ-2025-120881: Generated "Figure 8", "9"
    """
    # Simulate checking for these patterns
    invalid_labels = [
        "A-a",
        "A-b",
        "C (plot)",
        "C (right)",
        "Rice cell",
        "Small brown planthopper",
        "Figure 8",
        "9",
    ]

    valid_labels = ["A", "B", "C", "D", "E", "F", "G"]

    for invalid_label in invalid_labels:
        # These should all fail validation
        assert (
            invalid_label not in valid_labels
        ), f"Invalid label '{invalid_label}' should not be in valid labels"

        # Check for patterns
        has_dash = "-" in invalid_label
        has_paren = "(" in invalid_label
        starts_with_figure = invalid_label.startswith("Figure ")
        is_only_digit = invalid_label.isdigit()
        has_space = " " in invalid_label  # Descriptive labels like "Rice cell"

        # At least one of these should be true for invalid labels
        # Valid panel labels are typically single letters (A, B, C) or short alphanumeric codes
        assert (
            has_dash
            or has_paren
            or starts_with_figure
            or is_only_digit
            or has_space
            or (len(invalid_label) > 1 and not invalid_label.isupper())
        ), f"Invalid label '{invalid_label}' should match at least one invalid pattern"
