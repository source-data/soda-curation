"""Tests for stats_test module."""

from unittest.mock import Mock, patch

import pytest

from src.soda_curation.qc.qc_tests.stats_test import StatsTestAnalyzer


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
                        "user": "Test user prompt with $figure_caption placeholder",
                    },
                }
            }
        }
    }


@pytest.fixture
def sample_figure():
    """Return sample figure data."""
    return (
        "Figure 1",
        "base64encodedimagedata",
        "This is a sample figure caption showing statistical significance (p<0.05).",
    )


# Patch at the class level where it's used
@patch("src.soda_curation.qc.qc_tests.stats_test.ModelAPI")
def test_stats_test_analyzer_init(mock_model_api_class, config):
    """Test initialization of StatsTestAnalyzer."""
    # Create analyzer, this should call ModelAPI
    analyzer = StatsTestAnalyzer(config)

    # Verify the analyzer was initialized with the correct config
    assert analyzer.config == config
    # Verify ModelAPI was initialized
    mock_model_api_class.assert_called_once()


@patch("src.soda_curation.qc.qc_tests.stats_test.StatsTestAnalyzer.analyze_figure")
def test_stats_test_analyzer_analyze_figure(mock_analyze_figure, config, sample_figure):
    """Test analyze_figure method with a passing result."""
    # Create a mock result that would pass
    mock_result = Mock()
    mock_result.outputs = [
        Mock(
            panel_label="A",
            is_a_plot="yes",
            statistical_test_needed="yes",
            statistical_test_mentioned="yes",
            justify_why_test_is_missing="",
            from_the_caption="Statistical significance was tested using t-test (p<0.05).",
        )
    ]

    # Set up the mock to return (True, mock_result)
    mock_analyze_figure.return_value = (True, mock_result)

    # Create analyzer and call analyze_figure
    analyzer = StatsTestAnalyzer(config)
    figure_label, encoded_image, figure_caption = sample_figure
    passed, result = analyzer.analyze_figure(
        figure_label, encoded_image, figure_caption
    )

    # Verify the result
    assert passed is True
    assert result is mock_result
    assert hasattr(result, "outputs")
    assert result.outputs[0].panel_label == "A"
    assert result.outputs[0].statistical_test_mentioned == "yes"


@patch("src.soda_curation.qc.qc_tests.stats_test.StatsTestAnalyzer.analyze_figure")
def test_stats_test_analyzer_with_missing_test(
    mock_analyze_figure, config, sample_figure
):
    """Test analyze_figure method with missing statistical test."""
    # Create a mock result that would fail
    mock_result = Mock()
    mock_result.outputs = [
        Mock(
            panel_label="A",
            is_a_plot="yes",
            statistical_test_needed="yes",
            statistical_test_mentioned="no",
            justify_why_test_is_missing="The figure shows statistical significance with p<0.05 but doesn't mention which test was used.",
            from_the_caption="",
        )
    ]

    # Set up the mock to return (False, mock_result)
    mock_analyze_figure.return_value = (False, mock_result)

    # Create analyzer and call analyze_figure
    analyzer = StatsTestAnalyzer(config)
    figure_label, encoded_image, figure_caption = sample_figure
    passed, result = analyzer.analyze_figure(
        figure_label, encoded_image, figure_caption
    )

    # Verify result
    assert passed is False
    assert result is mock_result
    assert hasattr(result, "outputs")
    assert result.outputs[0].statistical_test_mentioned == "no"
    assert result.outputs[0].justify_why_test_is_missing != ""


@patch("src.soda_curation.qc.qc_tests.stats_test.StatsTestAnalyzer.analyze_figure")
def test_stats_test_analyzer_no_test_needed(mock_analyze_figure, config, sample_figure):
    """Test analyze_figure method when no statistical test is needed."""
    # Create a mock result where no test is needed
    mock_result = Mock()
    mock_result.outputs = [
        Mock(
            panel_label="A",
            is_a_plot="no",
            statistical_test_needed="no",
            statistical_test_mentioned="not needed",
            justify_why_test_is_missing="",
            from_the_caption="",
        )
    ]

    # Set up the mock to return (True, mock_result)
    mock_analyze_figure.return_value = (True, mock_result)

    # Create analyzer and call analyze_figure
    analyzer = StatsTestAnalyzer(config)
    figure_label, encoded_image, figure_caption = sample_figure
    passed, result = analyzer.analyze_figure(
        figure_label, encoded_image, figure_caption
    )

    # Verify result
    assert passed is True
    assert result is mock_result
    assert hasattr(result, "outputs")
    assert result.outputs[0].is_a_plot == "no"
    assert result.outputs[0].statistical_test_needed == "no"


@patch("src.soda_curation.qc.qc_tests.stats_test.ModelAPI")
def test_stats_test_analyzer_error_handling(
    mock_model_api_class, config, sample_figure
):
    """Test error handling in analyze_figure method."""
    # Set up mock to raise an exception
    mock_instance = Mock()
    mock_model_api_class.return_value = mock_instance
    mock_instance.generate_response.side_effect = Exception("API error")

    # Create analyzer and analyze figure
    analyzer = StatsTestAnalyzer(config)
    figure_label, encoded_image, figure_caption = sample_figure

    with pytest.raises(Exception):
        passed, result = analyzer.analyze_figure(
            figure_label, encoded_image, figure_caption
        )
