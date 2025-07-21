# tests/test_prompt_registry.py
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import openai
import pytest

from soda_curation.qc.prompt_registry import create_registry
from src.soda_curation.qc.model_api import ModelAPI
from src.soda_curation.qc.qc_pipeline import QCPipeline


@pytest.fixture
def test_config():
    return {
        "qc_version": "1.0.0",
        "qc_test_metadata": {
            "panel": {
                "error_bars_defined": {
                    "name": "Error Bars Defined",
                    "description": "Test description",
                    "permalink": "https://example.com/permalink",
                }
            }
        },
        "default": {
            "pipeline": {"error_bars_defined": {"openai": {"model": "gpt-4o"}}}
        },
    }


@pytest.fixture
def mock_zip_structure():
    mock_zip = MagicMock()
    figure1 = MagicMock()
    figure1.figure_label = "Figure 1"
    figure1.figure_caption = "Test caption"
    figure1.encoded_image = "base64image"
    mock_zip.figures = [figure1]
    return mock_zip


@pytest.fixture
def figure_data():
    return [("Figure 1", "base64image", "Test caption")]


@pytest.fixture
def model_config():
    return {"default": {"openai": {"model": "gpt-4o"}}}


def test_registry_initialization():
    registry = create_registry()
    assert registry is not None
    assert registry.config is not None


def test_get_prompt_metadata():
    # Check if test exists in config
    registry = create_registry()
    metadata = registry.get_prompt_metadata("individual_data_points")
    assert metadata.name is not None
    assert metadata.description is not None
    assert metadata.permalink is not None
    assert metadata.version is not None  # Changed from prompt_version to version
    assert metadata.prompt_number is not None  # Add check for prompt_number


def test_get_prompt():
    registry = create_registry()

    # Mock the path.exists method to return True
    with patch("pathlib.Path.exists", return_value=True):
        # Mock the read_text method to return a test prompt
        with patch("pathlib.Path.read_text", return_value="This is a test prompt"):
            prompt = registry.get_prompt("individual_data_points")
            assert prompt == "This is a test prompt"


def test_get_schema():
    registry = create_registry()

    # Mock the path.exists method to return True
    with patch("pathlib.Path.exists", return_value=True):
        # Mock the read_text method to return a test schema
        with patch("pathlib.Path.read_text", return_value='{"type": "object"}'):
            # Path.read_text returns a string, but json.loads converts it to a dict
            with patch("json.loads", return_value={"type": "object"}):
                schema = registry.get_schema("individual_data_points")
                assert schema == {"type": "object"}


def test_get_pydantic_model():
    registry = create_registry()

    # Create a sample schema
    sample_schema = {
        "type": "object",
        "properties": {
            "outputs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"panel_label": {"type": "string"}},
                },
            }
        },
    }

    # Mock get_schema to return our sample schema
    with patch.object(registry, "get_schema", return_value=sample_schema):
        # Mock generate_pydantic_model_from_schema to return a simple model class
        class MockModel:
            def model_validate_json(self, json_str):
                return "validated_model"

        with patch.object(
            registry, "generate_pydantic_model_from_schema", return_value=MockModel
        ):
            model = registry.get_pydantic_model("individual_data_points")
            assert model == MockModel


def test_nonexistent_test():
    registry = create_registry()
    metadata = registry.get_prompt_metadata("nonexistent_test")
    # Check that the returned metadata is a PromptMetadata with default values
    assert metadata.name == "Nonexistent Test"
    assert metadata.description == ""
    assert metadata.permalink == ""
    assert metadata.version == "latest"
    assert metadata.prompt_number == 1


def test_pipeline_handles_malformed_analyzer_output(
    tmp_path, test_config, mock_zip_structure, figure_data
):
    class MalformedAnalyzer:
        def analyze_figure(self, *args, **kwargs):
            return True, "not a dict"

    class TestableQCPipeline(QCPipeline):
        def _initialize_tests(self):
            return {"error_bars_defined": MalformedAnalyzer()}

    pipeline = TestableQCPipeline(test_config, tmp_path)
    result = pipeline.run(mock_zip_structure, figure_data)
    assert "qc_version" in result
    assert "figures" in result
    # Should still produce a result, but panels may be empty or have error info


@patch("src.soda_curation.qc.model_api.openai.OpenAI")
def test_generate_response_openai_error(mock_openai, model_config):
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    mock_client.beta.chat.completions.parse.side_effect = openai.OpenAIError(
        "API error"
    )
    api = ModelAPI(model_config)
    with pytest.raises(openai.OpenAIError):
        api.generate_response(
            encoded_image="base64",
            caption="caption",
            prompt_config={"prompts": {"system": "", "user": ""}},
        )


@patch("src.soda_curation.qc.model_api.openai.OpenAI")
def test_generate_response_invalid_json(mock_openai, model_config):
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="{not: valid json"))]
    mock_client.beta.chat.completions.parse.return_value = mock_response
    api = ModelAPI(model_config)
    with pytest.raises(json.JSONDecodeError):
        api.generate_response(
            encoded_image="base64",
            caption="caption",
            prompt_config={"prompts": {"system": "", "user": ""}},
        )


def test_get_schema_file_not_found():
    registry = create_registry()
    with pytest.raises(FileNotFoundError):
        registry.get_schema("nonexistent_test")


def test_get_prompt_file_not_found():
    registry = create_registry()
    with patch("pathlib.Path.exists", return_value=False):
        with pytest.raises(FileNotFoundError):
            registry.get_prompt("nonexistent_test")


def test_registry_with_malformed_config(tmp_path):
    # Write a malformed YAML config
    config_path = tmp_path / "bad_config.yaml"
    config_path.write_text("not: [valid, yaml")
    with pytest.raises(Exception):
        create_registry(str(config_path))
