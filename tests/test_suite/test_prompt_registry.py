# tests/test_prompt_registry.py
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from soda_curation.qc.prompt_registry import create_registry


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
    with pytest.raises(ValueError):
        registry.get_prompt_metadata("nonexistent_test")
