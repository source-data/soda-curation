# tests/test_qc_runner.py
import json
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from soda_curation.qc.model_api import ModelAPI
from soda_curation.qc.prompt_registry import registry


# QCRunner class with all analyzers
class QCRunner:
    """Runner for executing multiple QC tests on a figure."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.test_names = config.get("qc_tests", [])

    def run_tests(self, figure_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run all configured tests on the figure data."""
        from soda_curation.qc.qc_tests.error_bars_defined import (
            ErrorBarsDefinedAnalyzer,
        )
        from soda_curation.qc.qc_tests.individual_data_points import (
            IndividualDataPointsAnalyzer,
        )
        from soda_curation.qc.qc_tests.micrograph_scale_bar import (
            MicrographScaleBarAnalyzer,
        )
        from soda_curation.qc.qc_tests.micrograph_symbols_defined import (
            MicrographSymbolsDefinedAnalyzer,
        )
        from soda_curation.qc.qc_tests.plot_axis_units import PlotAxisUnitsAnalyzer
        from soda_curation.qc.qc_tests.plot_gap_labeling import PlotGapLabelingAnalyzer
        from soda_curation.qc.qc_tests.replicates_defined import (
            ReplicatesDefinedAnalyzer,
        )
        from soda_curation.qc.qc_tests.stat_significance_level import (
            StatsSignificanceLevelAnalyzer,
        )
        from soda_curation.qc.qc_tests.stat_test import StatsTestAnalyzer

        # Map test names to analyzer classes
        analyzers = {
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

        results = {}
        for test_name in self.test_names:
            if test_name not in analyzers:
                raise ValueError(f"Unknown test: {test_name}")

            analyzer = analyzers[test_name](self.config)

            # Extract the required arguments from figure_data
            panel = figure_data["panels"][0]  # Use the first panel
            figure_label = panel["label"]
            # Get image URL or use a dummy value if not available
            encoded_image = panel.get("image_url", "dummy_encoded_image")
            figure_caption = figure_data["caption"]

            # Call analyze_figure with the correct arguments
            passed, result = analyzer.analyze_figure(
                figure_label, encoded_image, figure_caption
            )

            results[test_name] = result

        return results


# Create a mock for Pydantic model
class MockPydanticModel:
    def __init__(self, outputs):
        self.outputs = outputs

    def model_dump(self):
        return {"outputs": self.outputs}


# Mock for registry
class MockMetadata:
    def __init__(self):
        self.name = "Test Name"
        self.description = "Test Description"
        self.permalink = "https://github.com/example/repo/path"
        self.version = "1.0"
        self.prompt_number = 1


# Test QCRunner with multiple tests
@patch("soda_curation.qc.model_api.ModelAPI.generate_response")
@patch("soda_curation.qc.prompt_registry.registry.get_schema")
@patch("soda_curation.qc.prompt_registry.registry.get_pydantic_model")
@patch("soda_curation.qc.prompt_registry.registry.get_prompt_metadata")
@patch("soda_curation.qc.prompt_registry.registry.get_prompt")
def test_qc_runner_multiple_tests(
    mock_get_prompt,
    mock_get_metadata,
    mock_get_model,
    mock_get_schema,
    mock_generate_response,
):
    # Set up registry mocks
    mock_get_schema.return_value = {"type": "object"}
    mock_get_metadata.return_value = MockMetadata()
    mock_get_prompt.return_value = "Test prompt"

    # Mock the Pydantic model
    class MockModel:
        def model_validate_json(self, json_str):
            data = json.loads(json_str)
            outputs = []
            for output in data["outputs"]:
                mock_output = MagicMock()
                for key, value in output.items():
                    setattr(mock_output, key, value)
                outputs.append(mock_output)

            result = MagicMock()
            result.outputs = outputs
            result.model_dump = lambda: data
            return result

    mock_get_model.return_value = MockModel()

    # Mock different returns for different tests
    # Update the mock function to use kwargs instead of args
    def generate_response_side_effect(*args, **kwargs):
        # Get the caption from kwargs
        prompt_config = kwargs.get("prompt_config", {})

        # Get the system prompt text
        system_prompt = prompt_config.get("prompts", {}).get("system", "")

        # Decide which response to return based on context
        if (
            "individual data points" in system_prompt.lower()
            or "individual_data_points" in str(kwargs)
        ):
            return json.dumps(
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
        elif "error bars" in system_prompt.lower() or "error_bars_defined" in str(
            kwargs
        ):
            return json.dumps(
                {
                    "outputs": [
                        {
                            "panel_label": "A",
                            "error_bar_on_figure": "yes",
                            "error_bar_defined_in_caption": "yes",
                            "from_the_caption": "Error bars represent SD.",
                        }
                    ]
                }
            )
        else:
            return json.dumps({"outputs": [{"panel_label": "A", "result": "default"}]})

    mock_generate_response.side_effect = generate_response_side_effect

    # Configure the runner with multiple tests
    config = {
        "model": "test_model",
        "qc_tests": ["individual_data_points", "error_bars_defined"],
        "pipeline": {
            "individual_data_points": {
                "openai": {"prompts": {"system": "", "user": ""}}
            },
            "error_bars_defined": {"openai": {"prompts": {"system": "", "user": ""}}},
        },
    }

    runner = QCRunner(config)

    figure_data = {
        "caption": "This is a test figure with error bars representing SD.",
        "panels": [{"label": "A", "image_url": "http://example.com/image.png"}],
    }

    results = runner.run_tests(figure_data)

    # Verify the results
    assert "individual_data_points" in results
    assert "error_bars_defined" in results
    assert mock_generate_response.call_count == 2  # Called once for each test
