# tests/test_prompt_registry.py
"""Tests for the Langfuse-backed PromptRegistry."""

import json
from unittest.mock import MagicMock, patch

import pytest

from soda_curation.qc.prompt_registry import PromptMetadata, create_registry
from src.soda_curation.qc.qc_pipeline import QCPipeline

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def test_config():
    return {
        "qc_version": "3.0.0",
        "qc_check_metadata": {
            "panel": {
                "error_bars_defined": {
                    "name": "Error Bars Defined",
                    "description": "Test description",
                    "checklist_type": "fig-checklist",
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


# ---------------------------------------------------------------------------
# PromptRegistry core tests
# ---------------------------------------------------------------------------


def test_registry_initialization():
    registry = create_registry()
    assert registry is not None
    assert registry.config is not None


def test_get_prompt_metadata_returns_namedtuple():
    """get_prompt_metadata always returns a PromptMetadata, even without Langfuse."""
    registry = create_registry()
    # Mock Langfuse as unavailable so we exercise the fallback path
    with patch.object(
        registry,
        "_get_langfuse_prompt",
        side_effect=RuntimeError("Langfuse unavailable"),
    ):
        metadata = registry.get_prompt_metadata("individual_data_points")

    assert isinstance(metadata, PromptMetadata)
    assert metadata.name is not None
    assert metadata.description is not None
    assert metadata.permalink == ""  # Fallback has empty permalink
    assert metadata.version == "latest"  # Fallback version
    assert metadata.prompt_file == ""  # Langfuse-backed: no file


def test_get_prompt_metadata_from_langfuse():
    """get_prompt_metadata returns Langfuse data when available."""
    registry = create_registry()
    mock_lf_prompt = MagicMock()
    mock_lf_prompt.config = {
        "name": "Individual Data Points",
        "description": "Checks individual data points.",
    }
    mock_lf_prompt.version = 3

    with patch.object(registry, "_get_langfuse_prompt", return_value=mock_lf_prompt):
        # Local config name ("Individual Data Points Displayed") takes priority
        metadata = registry.get_prompt_metadata("individual_data_points")

    assert metadata.name == "Individual Data Points Displayed"
    assert metadata.description == "Checks individual data points."
    assert metadata.version == "3"
    assert "individual-data-points" in metadata.permalink
    assert metadata.prompt_file == ""


def test_get_prompt_text():
    """get_prompt returns the text prompt string from Langfuse."""
    registry = create_registry()
    mock_lf_prompt = MagicMock()
    mock_lf_prompt.prompt = "This is a test system prompt."

    with patch.object(registry, "_get_langfuse_prompt", return_value=mock_lf_prompt):
        prompt = registry.get_prompt("individual_data_points")

    assert prompt == "This is a test system prompt."


def test_get_prompt_chat_format():
    """get_prompt extracts system role from a chat-format prompt."""
    registry = create_registry()
    mock_lf_prompt = MagicMock()
    mock_lf_prompt.prompt = [
        {"role": "system", "content": "System instruction here."},
        {"role": "user", "content": "User template {{text}}"},
    ]

    with patch.object(registry, "_get_langfuse_prompt", return_value=mock_lf_prompt):
        prompt = registry.get_prompt("individual_data_points")

    assert prompt == "System instruction here."


def test_get_runtime_config_merges_hints():
    registry = create_registry()
    mock_lf_prompt = MagicMock()
    mock_lf_prompt.config = {
        "agentic": True,
        "model_config": {"tools": [{"type": "web_search_preview"}]},
    }
    with patch.object(registry, "_get_langfuse_prompt", return_value=mock_lf_prompt):
        hints = registry.get_runtime_config("individual_data_points")

    assert hints["agentic"] is True
    assert "model_config" in hints
    assert hints["model_config"]["tools"][0]["type"] == "web_search_preview"


def test_get_schema():
    """get_schema returns the schema dict from Langfuse prompt config."""
    registry = create_registry()
    mock_lf_prompt = MagicMock()
    mock_lf_prompt.config = {"schema": {"type": "object", "properties": {}}}

    with patch.object(registry, "_get_langfuse_prompt", return_value=mock_lf_prompt):
        schema = registry.get_schema("individual_data_points")

    assert schema == {"type": "object", "properties": {}}


def test_get_schema_raises_when_not_in_config():
    """get_schema raises FileNotFoundError when prompt config has no schema."""
    registry = create_registry()
    mock_lf_prompt = MagicMock()
    mock_lf_prompt.config = {}  # Neither 'schema' nor 'output_schema'

    with patch.object(registry, "_get_langfuse_prompt", return_value=mock_lf_prompt):
        with pytest.raises(FileNotFoundError):
            registry.get_schema("individual_data_points")


def test_get_schema_from_output_schema():
    """get_schema understands the Langfuse output_schema.format.schema nesting."""
    registry = create_registry()
    inner_schema = {"type": "object", "properties": {"panel_label": {"type": "string"}}}
    mock_lf_prompt = MagicMock()
    mock_lf_prompt.config = {
        "output_schema": {"format": {"type": "json_schema", "schema": inner_schema}}
    }

    with patch.object(registry, "_get_langfuse_prompt", return_value=mock_lf_prompt):
        schema = registry.get_schema("individual_data_points")

    assert schema == inner_schema


def test_get_schema_raises_when_langfuse_unavailable():
    """get_schema raises FileNotFoundError when Langfuse cannot be reached."""
    registry = create_registry()
    with patch.object(
        registry,
        "_get_langfuse_prompt",
        side_effect=RuntimeError("Langfuse not available"),
    ):
        with pytest.raises(FileNotFoundError):
            registry.get_schema("nonexistent_test")


def test_get_pydantic_model():
    """get_pydantic_model generates a model from the Langfuse schema."""
    registry = create_registry()

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

    with patch.object(registry, "get_schema", return_value=sample_schema):

        class MockModel:
            def model_validate_json(self, json_str):
                return "validated_model"

        with patch.object(
            registry, "generate_pydantic_model_from_schema", return_value=MockModel
        ):
            model = registry.get_pydantic_model("individual_data_points")
            assert model == MockModel


def test_nonexistent_test_fallback():
    """get_prompt_metadata returns sensible fallback for unknown tests."""
    registry = create_registry()

    # Ensure Langfuse is unavailable so we exercise the pure-fallback path
    with patch.object(
        registry,
        "_get_langfuse_prompt",
        side_effect=RuntimeError("unavailable"),
    ):
        metadata = registry.get_prompt_metadata("nonexistent_test")

    assert metadata.name == "Nonexistent Test"
    assert metadata.description == ""
    assert metadata.permalink == ""
    assert metadata.version == "latest"
    assert metadata.prompt_file == ""  # Langfuse-backed: no file path


def test_get_prompt_raises_when_langfuse_unavailable():
    """get_prompt propagates the error when Langfuse is not reachable."""
    registry = create_registry()
    with patch.object(
        registry,
        "_get_langfuse_prompt",
        side_effect=RuntimeError("Not found"),
    ):
        with pytest.raises(RuntimeError):
            registry.get_prompt("nonexistent_test")


# ---------------------------------------------------------------------------
# Pipeline integration tests (no real Langfuse calls)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Config / YAML edge cases
# ---------------------------------------------------------------------------


def test_registry_with_malformed_config(tmp_path):
    config_path = tmp_path / "bad_config.yaml"
    config_path.write_text("not: [valid, yaml")
    with pytest.raises(Exception):
        create_registry(str(config_path))
