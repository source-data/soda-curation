"""Tests for StatsSignificanceLevelAnalyzer module."""

from unittest.mock import Mock, patch

import pytest

from soda_curation.qc.qc_tests.stats_significance_level import (
    StatsSignificanceLevelAnalyzer,
)


@pytest.fixture
def config():
    return {
        "pipeline": {
            "stats_significance_level": {
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
    return (
        "Figure 1",
        "base64encodedimagedata",
        "This is a sample figure caption with *p<0.05 and ****p<0.0001 significance.",
    )


@patch("soda_curation.qc.qc_tests.stats_significance_level.ModelAPI")
def test_stats_significance_level_analyzer_init(mock_model_api_class, config):
    analyzer = StatsSignificanceLevelAnalyzer(config)
    assert analyzer.config == config
    mock_model_api_class.assert_called_once()


@patch("soda_curation.qc.qc_tests.stats_significance_level.ModelAPI")
def test_analyze_figure_all_symbols_defined(
    mock_model_api_class, config, sample_figure
):
    # Mock the response from ModelAPI
    mock_instance = Mock()
    mock_model_api_class.return_value = mock_instance
    # Simulate a valid JSON response (from_the_caption as string)
    response_json = '{"outputs": [{"panel_label": "A", "is_a_plot": "yes", "significance_level_symbols_on_image": ["*", "****"], "significance_level_symbols_defined_in_caption": ["yes", "yes"], "from_the_caption": "*p<0.05; ****p<0.0001"}]}'
    mock_instance.generate_response.return_value = response_json

    analyzer = StatsSignificanceLevelAnalyzer(config)
    figure_label, encoded_image, figure_caption = sample_figure
    passed, result = analyzer.analyze_figure(
        figure_label, encoded_image, figure_caption
    )
    assert passed is True
    assert hasattr(result, "outputs")
    assert result.outputs[0].panel_label == "A"
    assert result.outputs[0].is_a_plot == "yes"
    assert result.outputs[0].significance_level_symbols_defined_in_caption == [
        "yes",
        "yes",
    ]


@patch("soda_curation.qc.qc_tests.stats_significance_level.ModelAPI")
def test_analyze_figure_some_symbols_missing(
    mock_model_api_class, config, sample_figure
):
    mock_instance = Mock()
    mock_model_api_class.return_value = mock_instance
    # Simulate a response where not all symbols are defined (from_the_caption as string)
    response_json = '{"outputs": [{"panel_label": "A", "is_a_plot": "yes", "significance_level_symbols_on_image": ["*", "****"], "significance_level_symbols_defined_in_caption": ["yes", "no"], "from_the_caption": "*p<0.05"}]}'
    mock_instance.generate_response.return_value = response_json

    analyzer = StatsSignificanceLevelAnalyzer(config)
    figure_label, encoded_image, figure_caption = sample_figure
    passed, result = analyzer.analyze_figure(
        figure_label, encoded_image, figure_caption
    )
    assert passed is False
    assert hasattr(result, "outputs")
    assert result.outputs[0].significance_level_symbols_defined_in_caption == [
        "yes",
        "no",
    ]


@patch("soda_curation.qc.qc_tests.stats_significance_level.ModelAPI")
def test_analyze_figure_no_plot(mock_model_api_class, config, sample_figure):
    mock_instance = Mock()
    mock_model_api_class.return_value = mock_instance
    # Simulate a response for a non-plot panel (from_the_caption as string)
    response_json = '{"outputs": [{"panel_label": "A", "is_a_plot": "no", "significance_level_symbols_on_image": [], "significance_level_symbols_defined_in_caption": [], "from_the_caption": ""}]}'
    mock_instance.generate_response.return_value = response_json

    analyzer = StatsSignificanceLevelAnalyzer(config)
    figure_label, encoded_image, figure_caption = sample_figure
    passed, result = analyzer.analyze_figure(
        figure_label, encoded_image, figure_caption
    )
    assert passed is True
    assert hasattr(result, "outputs")
    assert result.outputs[0].is_a_plot == "no"
    assert result.outputs[0].significance_level_symbols_on_image == []


@patch("soda_curation.qc.qc_tests.stats_significance_level.ModelAPI")
def test_stats_significance_level_analyzer_error_handling(
    mock_model_api_class, config, sample_figure
):
    mock_instance = Mock()
    mock_model_api_class.return_value = mock_instance
    mock_instance.generate_response.side_effect = Exception("API error")
    analyzer = StatsSignificanceLevelAnalyzer(config)
    figure_label, encoded_image, figure_caption = sample_figure
    with pytest.raises(Exception):
        analyzer.analyze_figure(figure_label, encoded_image, figure_caption)
