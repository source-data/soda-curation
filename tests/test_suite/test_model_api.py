"""Tests for the ModelAPI."""

import json
from unittest.mock import MagicMock, patch

import openai
import pytest

from src.soda_curation.qc.model_api import ModelAPI


class TestModelAPI:
    """Tests for the ModelAPI class."""

    @pytest.fixture
    def model_config(self):
        """Create a test model configuration."""
        return {"default": {"openai": {"model": "gpt-4o"}}}

    @patch("src.soda_curation.qc.model_api.openai.OpenAI")
    def test_generate_response(self, mock_openai, model_config):
        """Test generating a response."""
        # Setup mock for beta.chat.completions.parse
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        mock_beta = MagicMock()
        mock_client.beta = mock_beta

        mock_chat = MagicMock()
        mock_beta.chat = mock_chat

        mock_completions = MagicMock()
        mock_chat.completions = mock_completions

        # Mock the parse method return value
        mock_response = MagicMock()
        mock_completions.parse.return_value = mock_response

        # Mock the choices and content
        mock_choice = MagicMock()
        mock_response.choices = [mock_choice]

        mock_message = MagicMock()
        mock_choice.message = mock_message

        # Set the actual content value
        mock_message.content = {"test": "value"}

        # Create API and generate response
        api = ModelAPI(model_config)
        response = api.generate_response(
            encoded_image="base64encodedimage",
            caption="Test caption",
            prompt_config={
                "prompts": {
                    "system": "System prompt",
                    "user": "User prompt with $figure_caption",
                },
                "openai": {"model": "gpt-4o"},
            },
        )

        # Verify response
        assert response == {"test": "value"}
        # Verify the correct method was called
        mock_completions.parse.assert_called_once()

    @patch("src.soda_curation.qc.model_api.openai.OpenAI")
    def test_generate_response_openai_error(self, mock_openai, model_config):
        """Test handling of OpenAI API errors."""
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
    def test_generate_response_invalid_json(self, mock_openai, model_config):
        """Test handling of invalid JSON responses."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="{not: valid json"))
        ]
        mock_client.beta.chat.completions.parse.return_value = mock_response
        api = ModelAPI(model_config)
        with pytest.raises(json.JSONDecodeError):
            api.generate_response(
                encoded_image="base64",
                caption="caption",
                prompt_config={"prompts": {"system": "", "user": ""}},
            )
