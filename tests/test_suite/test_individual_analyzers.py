# tests/test_suite/test_individual_analyzers.py
import json
from typing import Any, Dict
from unittest.mock import MagicMock, patch

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


# Create fixtures for common test data
@pytest.fixture
def figure_data():
    return {
        "caption": "This is a test figure caption with error bars representing standard deviation.",
        "panels": [{"label": "A", "image_url": "http://example.com/image.png"}],
    }


@pytest.fixture
def test_config():
    return {
        "model": "test_model",
        "pipeline": {
            "individual_data_points": {
                "openai": {"prompts": {"system": "", "user": ""}}
            },
            "error_bars_defined": {"openai": {"prompts": {"system": "", "user": ""}}},
            "micrograph_scale_bar": {"openai": {"prompts": {"system": "", "user": ""}}},
            "micrograph_symbols_defined": {
                "openai": {"prompts": {"system": "", "user": ""}}
            },
            "plot_axis_units": {"openai": {"prompts": {"system": "", "user": ""}}},
            "plot_gap_labeling": {"openai": {"prompts": {"system": "", "user": ""}}},
            "replicates_defined": {"openai": {"prompts": {"system": "", "user": ""}}},
            "stat_significance_level": {
                "openai": {"prompts": {"system": "", "user": ""}}
            },
            "stat_test": {"openai": {"prompts": {"system": "", "user": ""}}},
        },
    }


# Mock for metadata and models that can be reused
@pytest.fixture
def mock_metadata():
    return type(
        "obj",
        (object,),
        {
            "name": "Test Name",
            "description": "Test description",
            "permalink": "https://github.com/example/repo/path",
            "version": "1.0",
            "prompt_number": 1,
        },
    )


# Test individual_data_points analyzer
@patch("soda_curation.qc.model_api.ModelAPI.generate_response")
@patch("soda_curation.qc.prompt_registry.registry.get_prompt_metadata")
@patch("soda_curation.qc.prompt_registry.registry.get_pydantic_model")
@patch("soda_curation.qc.prompt_registry.registry.get_prompt")
@patch("soda_curation.qc.prompt_registry.registry.get_schema")
def test_individual_data_points_analyzer(
    mock_get_schema,
    mock_get_prompt,
    mock_get_model,
    mock_get_metadata,
    mock_generate_response,
    figure_data,
    test_config,
    mock_metadata,
):
    # Set up mocks
    mock_get_metadata.return_value = mock_metadata

    # Create a mock for the model result
    class MockOutput:
        def __init__(self):
            self.panel_label = "A"
            self.plot = "yes"
            self.average_values = "yes"
            self.individual_values = "yes"

    class MockResult:
        def __init__(self):
            self.outputs = [MockOutput()]
            self.metadata = {}

        def model_dump(self):
            return {"outputs": [{"panel_label": "A", "individual_values": "yes"}]}

    # Set up the model mock
    class MockModel:
        def model_validate_json(self, json_str):
            return MockResult()

    mock_get_model.return_value = MockModel()
    mock_get_prompt.return_value = "Test prompt"
    mock_get_schema.return_value = {}

    # Set up response from model
    mock_generate_response.return_value = json.dumps(
        {
            "outputs": [
                {
                    "panel_label": "A",
                    "plot": "yes",
                    "average_values": "yes",
                    "individual_values": "yes",
                }
            ]
        }
    )

    # Create analyzer and call the method correctly
    analyzer = IndividualDataPointsAnalyzer(test_config)
    panel = figure_data["panels"][0]
    passed, result = analyzer.analyze_figure(
        panel["label"], "dummy_encoded_image", figure_data["caption"]
    )

    # Check the result
    assert hasattr(result, "outputs")
    assert result.outputs[0].individual_values == "yes"


# Test error_bars_defined analyzer
@patch("soda_curation.qc.model_api.ModelAPI.generate_response")
@patch("soda_curation.qc.prompt_registry.registry.get_prompt_metadata")
@patch("soda_curation.qc.prompt_registry.registry.get_pydantic_model")
@patch("soda_curation.qc.prompt_registry.registry.get_prompt")
@patch("soda_curation.qc.prompt_registry.registry.get_schema")
def test_error_bars_defined_analyzer(
    mock_get_schema,
    mock_get_prompt,
    mock_get_model,
    mock_get_metadata,
    mock_generate_response,
    figure_data,
    test_config,
    mock_metadata,
):
    # Set up mocks
    mock_get_metadata.return_value = mock_metadata

    # Create a mock for the model result
    class MockOutput:
        def __init__(self):
            self.panel_label = "A"
            self.error_bar_on_figure = "yes"
            self.error_bar_defined_in_caption = "yes"
            self.from_the_caption = "Error bars represent standard deviation."

    class MockResult:
        def __init__(self):
            self.outputs = [MockOutput()]
            self.metadata = {}

        def model_dump(self):
            return {
                "outputs": [{"panel_label": "A", "error_bar_defined_in_caption": "yes"}]
            }

    # Set up the model mock
    class MockModel:
        def model_validate_json(self, json_str):
            return MockResult()

    mock_get_model.return_value = MockModel()
    mock_get_prompt.return_value = "Test prompt"
    mock_get_schema.return_value = {}

    # Set up response from model
    mock_generate_response.return_value = json.dumps(
        {
            "outputs": [
                {
                    "panel_label": "A",
                    "error_bar_on_figure": "yes",
                    "error_bar_defined_in_caption": "yes",
                    "from_the_caption": "Error bars represent standard deviation.",
                }
            ]
        }
    )

    # Create analyzer and call the method correctly
    analyzer = ErrorBarsDefinedAnalyzer(test_config)
    panel = figure_data["panels"][0]
    passed, result = analyzer.analyze_figure(
        panel["label"], "dummy_encoded_image", figure_data["caption"]
    )

    # Check the result
    assert hasattr(result, "outputs")
    assert result.outputs[0].error_bar_defined_in_caption == "yes"


# Test micrograph_scale_bar analyzer
@patch("soda_curation.qc.model_api.ModelAPI.generate_response")
@patch("soda_curation.qc.prompt_registry.registry.get_prompt_metadata")
@patch("soda_curation.qc.prompt_registry.registry.get_pydantic_model")
@patch("soda_curation.qc.prompt_registry.registry.get_prompt")
@patch("soda_curation.qc.prompt_registry.registry.get_schema")
def test_micrograph_scale_bar_analyzer(
    mock_get_schema,
    mock_get_prompt,
    mock_get_model,
    mock_get_metadata,
    mock_generate_response,
    figure_data,
    test_config,
    mock_metadata,
):
    # Set up mocks
    mock_get_metadata.return_value = mock_metadata

    # Create a mock for the model result
    class MockOutput:
        def __init__(self):
            self.panel_label = "A"
            self.micrograph = "yes"
            self.scale_bar_on_image = "yes"
            self.scale_bar_defined_in_caption = "yes"
            self.from_the_caption = "Scale bar: 10 μm."

    class MockResult:
        def __init__(self):
            self.outputs = [MockOutput()]
            self.metadata = {}

        def model_dump(self):
            return {"outputs": [{"panel_label": "A", "scale_bar_on_image": "yes"}]}

    # Set up the model mock
    class MockModel:
        def model_validate_json(self, json_str):
            return MockResult()

    mock_get_model.return_value = MockModel()
    mock_get_prompt.return_value = "Test prompt"
    mock_get_schema.return_value = {}

    # Set up response from model
    mock_generate_response.return_value = json.dumps(
        {
            "outputs": [
                {
                    "panel_label": "A",
                    "micrograph": "yes",
                    "scale_bar_on_image": "yes",
                    "scale_bar_defined_in_caption": "yes",
                    "from_the_caption": "Scale bar: 10 μm.",
                }
            ]
        }
    )

    # Create analyzer and call the method correctly
    analyzer = MicrographScaleBarAnalyzer(test_config)
    panel = figure_data["panels"][0]
    passed, result = analyzer.analyze_figure(
        panel["label"], "dummy_encoded_image", figure_data["caption"]
    )

    # Check the result
    assert hasattr(result, "outputs")
    assert result.outputs[0].scale_bar_on_image == "yes"


# Test micrograph_symbols_defined analyzer
@patch("soda_curation.qc.model_api.ModelAPI.generate_response")
@patch("soda_curation.qc.prompt_registry.registry.get_prompt_metadata")
@patch("soda_curation.qc.prompt_registry.registry.get_pydantic_model")
@patch("soda_curation.qc.prompt_registry.registry.get_prompt")
@patch("soda_curation.qc.prompt_registry.registry.get_schema")
def test_micrograph_symbols_defined_analyzer(
    mock_get_schema,
    mock_get_prompt,
    mock_get_model,
    mock_get_metadata,
    mock_generate_response,
    figure_data,
    test_config,
    mock_metadata,
):
    # Set up mocks
    mock_get_metadata.return_value = mock_metadata

    # Create a mock for the model result
    class MockOutput:
        def __init__(self):
            self.panel_label = "A"
            self.micrograph = "yes"
            self.symbols = "yes"
            self.symbols_defined_in_caption = "yes"
            self.from_the_caption = "Red arrows indicate cell nuclei."

    class MockResult:
        def __init__(self):
            self.outputs = [MockOutput()]
            self.metadata = {}

        def model_dump(self):
            return {
                "outputs": [{"panel_label": "A", "symbols_defined_in_caption": "yes"}]
            }

    # Set up the model mock
    class MockModel:
        def model_validate_json(self, json_str):
            return MockResult()

    mock_get_model.return_value = MockModel()
    mock_get_prompt.return_value = "Test prompt"
    mock_get_schema.return_value = {}

    # Set up response from model
    mock_generate_response.return_value = json.dumps(
        {
            "outputs": [
                {
                    "panel_label": "A",
                    "micrograph": "yes",
                    "symbols": "yes",
                    "symbols_defined_in_caption": "yes",
                    "from_the_caption": "Red arrows indicate cell nuclei.",
                }
            ]
        }
    )

    # Create analyzer and call the method correctly
    analyzer = MicrographSymbolsDefinedAnalyzer(test_config)
    panel = figure_data["panels"][0]
    passed, result = analyzer.analyze_figure(
        panel["label"], "dummy_encoded_image", figure_data["caption"]
    )

    # Check the result
    assert hasattr(result, "outputs")
    assert result.outputs[0].symbols_defined_in_caption == "yes"


# Test plot_axis_units analyzer
@patch("soda_curation.qc.model_api.ModelAPI.generate_response")
@patch("soda_curation.qc.prompt_registry.registry.get_prompt_metadata")
@patch("soda_curation.qc.prompt_registry.registry.get_pydantic_model")
@patch("soda_curation.qc.prompt_registry.registry.get_prompt")
@patch("soda_curation.qc.prompt_registry.registry.get_schema")
def test_plot_axis_units_analyzer(
    mock_get_schema,
    mock_get_prompt,
    mock_get_model,
    mock_get_metadata,
    mock_generate_response,
    figure_data,
    test_config,
    mock_metadata,
):
    # Set up mocks
    mock_get_metadata.return_value = mock_metadata

    # Create a mock for the model result
    class MockOutput:
        def __init__(self):
            self.panel_label = "A"
            self.is_a_plot = "yes"
            self.units_provided = "yes"
            self.justify_why_units_are_missing = ""
            self.unit_definition_as_provided = (
                "x-axis: time (min), y-axis: absorbance (AU)"
            )

    class MockResult:
        def __init__(self):
            self.outputs = [MockOutput()]
            self.metadata = {}

        def model_dump(self):
            return {"outputs": [{"panel_label": "A", "units_provided": "yes"}]}

    # Set up the model mock
    class MockModel:
        def model_validate_json(self, json_str):
            return MockResult()

    mock_get_model.return_value = MockModel()
    mock_get_prompt.return_value = "Test prompt"
    mock_get_schema.return_value = {}

    # Set up response from model
    mock_generate_response.return_value = json.dumps(
        {
            "outputs": [
                {
                    "panel_label": "A",
                    "is_a_plot": "yes",
                    "units_provided": "yes",
                    "justify_why_units_are_missing": "",
                    "unit_definition_as_provided": "x-axis: time (min), y-axis: absorbance (AU)",
                }
            ]
        }
    )

    # Create analyzer and call the method correctly
    analyzer = PlotAxisUnitsAnalyzer(test_config)
    panel = figure_data["panels"][0]
    passed, result = analyzer.analyze_figure(
        panel["label"], "dummy_encoded_image", figure_data["caption"]
    )

    # Check the result
    assert hasattr(result, "outputs")
    assert result.outputs[0].units_provided == "yes"


# Test plot_gap_labeling analyzer
@patch("soda_curation.qc.model_api.ModelAPI.generate_response")
@patch("soda_curation.qc.prompt_registry.registry.get_prompt_metadata")
@patch("soda_curation.qc.prompt_registry.registry.get_pydantic_model")
@patch("soda_curation.qc.prompt_registry.registry.get_prompt")
@patch("soda_curation.qc.prompt_registry.registry.get_schema")
def test_plot_gap_labeling_analyzer(
    mock_get_schema,
    mock_get_prompt,
    mock_get_model,
    mock_get_metadata,
    mock_generate_response,
    figure_data,
    test_config,
    mock_metadata,
):
    # Set up mocks
    mock_get_metadata.return_value = mock_metadata

    # Create a mock for the model result
    class MockOutput:
        def __init__(self):
            self.panel_label = "A"
            self.is_a_plot = "yes"
            self.gaps_defined = "yes"
            self.gap_description = "The y-axis has a break indicated by zigzag lines."
            self.justify_why_gaps_are_missing = ""

    class MockResult:
        def __init__(self):
            self.outputs = [MockOutput()]
            self.metadata = {}

        def model_dump(self):
            return {"outputs": [{"panel_label": "A", "gaps_defined": "yes"}]}

    # Set up the model mock
    class MockModel:
        def model_validate_json(self, json_str):
            return MockResult()

    mock_get_model.return_value = MockModel()
    mock_get_prompt.return_value = "Test prompt"
    mock_get_schema.return_value = {}

    # Set up response from model
    mock_generate_response.return_value = json.dumps(
        {
            "outputs": [
                {
                    "panel_label": "A",
                    "is_a_plot": "yes",
                    "gaps_defined": "yes",
                    "gap_description": "The y-axis has a break indicated by zigzag lines.",
                    "justify_why_gaps_are_missing": "",
                }
            ]
        }
    )

    # Create analyzer and call the method correctly
    analyzer = PlotGapLabelingAnalyzer(test_config)
    panel = figure_data["panels"][0]
    passed, result = analyzer.analyze_figure(
        panel["label"], "dummy_encoded_image", figure_data["caption"]
    )

    # Check the result
    assert hasattr(result, "outputs")
    assert result.outputs[0].gaps_defined == "yes"


# Test replicates_defined analyzer
@patch("soda_curation.qc.model_api.ModelAPI.generate_response")
@patch("soda_curation.qc.prompt_registry.registry.get_prompt_metadata")
@patch("soda_curation.qc.prompt_registry.registry.get_pydantic_model")
@patch("soda_curation.qc.prompt_registry.registry.get_prompt")
@patch("soda_curation.qc.prompt_registry.registry.get_schema")
def test_replicates_defined_analyzer(
    mock_get_schema,
    mock_get_prompt,
    mock_get_model,
    mock_get_metadata,
    mock_generate_response,
    figure_data,
    test_config,
    mock_metadata,
):
    # Set up mocks
    mock_get_metadata.return_value = mock_metadata

    # Create a mock for the model result
    class MockOutput:
        def __init__(self):
            self.panel_label = "A"
            self.involves_replicates = "yes"
            self.number_of_replicates = "3"
            self.type_of_replicates = "biological"

    class MockResult:
        def __init__(self):
            self.outputs = [MockOutput()]
            self.metadata = {}

        def model_dump(self):
            return {"outputs": [{"panel_label": "A", "number_of_replicates": "3"}]}

    # Set up the model mock
    class MockModel:
        def model_validate_json(self, json_str):
            return MockResult()

    mock_get_model.return_value = MockModel()
    mock_get_prompt.return_value = "Test prompt"
    mock_get_schema.return_value = {}

    # Set up response from model
    mock_generate_response.return_value = json.dumps(
        {
            "outputs": [
                {
                    "panel_label": "A",
                    "involves_replicates": "yes",
                    "number_of_replicates": "3",
                    "type_of_replicates": "biological",
                }
            ]
        }
    )

    # Create analyzer and call the method correctly
    analyzer = ReplicatesDefinedAnalyzer(test_config)
    panel = figure_data["panels"][0]
    passed, result = analyzer.analyze_figure(
        panel["label"], "dummy_encoded_image", figure_data["caption"]
    )

    # Check the result
    assert hasattr(result, "outputs")
    assert result.outputs[0].number_of_replicates == "3"


# Test stat_significance_level analyzer
@patch("soda_curation.qc.model_api.ModelAPI.generate_response")
@patch("soda_curation.qc.prompt_registry.registry.get_prompt_metadata")
@patch("soda_curation.qc.prompt_registry.registry.get_pydantic_model")
@patch("soda_curation.qc.prompt_registry.registry.get_prompt")
@patch("soda_curation.qc.prompt_registry.registry.get_schema")
def test_stat_significance_level_analyzer(
    mock_get_schema,
    mock_get_prompt,
    mock_get_model,
    mock_get_metadata,
    mock_generate_response,
    figure_data,
    test_config,
    mock_metadata,
):
    # Set up mocks
    mock_get_metadata.return_value = mock_metadata

    # Create a mock for the model result
    class MockOutput:
        def __init__(self):
            self.panel_label = "A"
            self.is_a_plot = "yes"
            self.significance_level_symbols_on_image = "yes"
            self.significance_level_symbols_defined_in_caption = "yes"
            self.from_the_caption = "* p < 0.05, ** p < 0.01."

    class MockResult:
        def __init__(self):
            self.outputs = [MockOutput()]
            self.metadata = {}

        def model_dump(self):
            return {
                "outputs": [
                    {
                        "panel_label": "A",
                        "significance_level_symbols_defined_in_caption": "yes",
                    }
                ]
            }

    # Set up the model mock
    class MockModel:
        def model_validate_json(self, json_str):
            return MockResult()

    mock_get_model.return_value = MockModel()
    mock_get_prompt.return_value = "Test prompt"
    mock_get_schema.return_value = {}

    # Set up response from model
    mock_generate_response.return_value = json.dumps(
        {
            "outputs": [
                {
                    "panel_label": "A",
                    "is_a_plot": "yes",
                    "significance_level_symbols_on_image": "yes",
                    "significance_level_symbols_defined_in_caption": "yes",
                    "from_the_caption": "* p < 0.05, ** p < 0.01.",
                }
            ]
        }
    )

    # Create analyzer and call the method correctly
    analyzer = StatsSignificanceLevelAnalyzer(test_config)
    panel = figure_data["panels"][0]
    passed, result = analyzer.analyze_figure(
        panel["label"], "dummy_encoded_image", figure_data["caption"]
    )

    # Check the result
    assert hasattr(result, "outputs")
    assert result.outputs[0].significance_level_symbols_defined_in_caption == "yes"


# Test stat_test analyzer
@patch("soda_curation.qc.model_api.ModelAPI.generate_response")
@patch("soda_curation.qc.prompt_registry.registry.get_prompt_metadata")
@patch("soda_curation.qc.prompt_registry.registry.get_pydantic_model")
@patch("soda_curation.qc.prompt_registry.registry.get_prompt")
@patch("soda_curation.qc.prompt_registry.registry.get_schema")
def test_stat_test_analyzer(
    mock_get_schema,
    mock_get_prompt,
    mock_get_model,
    mock_get_metadata,
    mock_generate_response,
    figure_data,
    test_config,
    mock_metadata,
):
    # Set up mocks
    mock_get_metadata.return_value = mock_metadata

    # Create a mock for the model result
    class MockOutput:
        def __init__(self):
            self.panel_label = "A"
            self.is_a_plot = "yes"
            self.statistical_test_needed = "yes"
            self.statistical_test_mentioned = "yes"
            self.justify_why_test_is_missing = ""
            self.from_the_caption = (
                "Statistical significance was determined using Student's t-test."
            )

    class MockResult:
        def __init__(self):
            self.outputs = [MockOutput()]
            self.metadata = {}

        def model_dump(self):
            return {
                "outputs": [{"panel_label": "A", "statistical_test_mentioned": "yes"}]
            }

    # Set up the model mock
    class MockModel:
        def model_validate_json(self, json_str):
            return MockResult()

    mock_get_model.return_value = MockModel()
    mock_get_prompt.return_value = "Test prompt"
    mock_get_schema.return_value = {}

    # Set up response from model
    mock_generate_response.return_value = json.dumps(
        {
            "outputs": [
                {
                    "panel_label": "A",
                    "is_a_plot": "yes",
                    "statistical_test_needed": "yes",
                    "statistical_test_mentioned": "yes",
                    "justify_why_test_is_missing": "",
                    "from_the_caption": "Statistical significance was determined using Student's t-test.",
                }
            ]
        }
    )

    # Create analyzer and call the method correctly
    analyzer = StatsTestAnalyzer(test_config)
    panel = figure_data["panels"][0]
    passed, result = analyzer.analyze_figure(
        panel["label"], "dummy_encoded_image", figure_data["caption"]
    )

    # Check the result
    assert hasattr(result, "outputs")
    assert result.outputs[0].statistical_test_mentioned == "yes"
