"""Tests for OpenAI utility functions with GPT-5 fallback support."""

import json
from unittest.mock import MagicMock, patch

import openai
import pytest

from src.soda_curation.pipeline.openai_utils import (
    GPT5_MODEL,
    MODELS_WITHOUT_PARAMETERS,
    call_openai_with_fallback,
    is_context_length_error,
    prepare_model_params,
    validate_model_config,
)


class TestContextLengthErrorDetection:
    """Test context length error detection."""

    def test_is_context_length_error_true(self):
        """Test that context length errors are correctly identified."""
        error_messages = [
            "maximum context length exceeded",
            "context length is too long",
            "token limit exceeded",
            "input too long for model",
            "context window exceeded",
            "maximum tokens exceeded",
            "input too long",
        ]

        for message in error_messages:
            error = Exception(message)
            assert is_context_length_error(
                error
            ), f"Should detect context error: {message}"

    def test_is_context_length_error_false(self):
        """Test that non-context length errors are correctly identified."""
        error_messages = [
            "API key invalid",
            "rate limit exceeded",
            "authentication failed",
            "network error",
            "invalid request",
        ]

        for message in error_messages:
            error = Exception(message)
            assert not is_context_length_error(
                error
            ), f"Should not detect context error: {message}"

    def test_is_context_length_error_case_insensitive(self):
        """Test that context length error detection is case insensitive."""
        error = Exception("MAXIMUM CONTEXT LENGTH EXCEEDED")
        assert is_context_length_error(error)


class TestModelParameters:
    """Test model parameter preparation."""

    def test_prepare_model_params_standard_model(self):
        """Test parameter preparation for standard models."""
        messages = [{"role": "user", "content": "test"}]
        params = prepare_model_params(
            model="gpt-4o",
            messages=messages,
            temperature=0.5,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.2,
            max_tokens=1000,
            json_mode=False,  # Disable default JSON mode
        )

        expected = {
            "model": "gpt-4o",
            "messages": messages,
            "temperature": 0.5,
            "top_p": 0.9,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.2,
            "max_tokens": 1000,
        }
        assert params == expected

    def test_prepare_model_params_gpt5_model(self):
        """Test parameter preparation for GPT-5 model (no additional parameters)."""
        messages = [{"role": "user", "content": "test"}]
        params = prepare_model_params(
            model=GPT5_MODEL,
            messages=messages,
            temperature=0.5,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.2,
            max_tokens=1000,
            json_mode=False,  # Disable default JSON mode
        )

        expected = {
            "model": GPT5_MODEL,
            "messages": messages,
        }
        assert params == expected

    def test_prepare_model_params_with_response_format(self):
        """Test parameter preparation with response format."""
        messages = [{"role": "user", "content": "test"}]
        response_format = {"type": "json_object"}

        params = prepare_model_params(
            model="gpt-4o",
            messages=messages,
            response_format=response_format,
        )

        assert params["response_format"] == response_format

    def test_prepare_model_params_with_pydantic_model(self):
        """Test parameter preparation with Pydantic model response format."""
        messages = [{"role": "user", "content": "test"}]

        class TestModel:
            pass

        params = prepare_model_params(
            model="gpt-4o",
            messages=messages,
            response_format=TestModel,
        )

        assert params["response_format"] == TestModel

    def test_prepare_model_params_json_mode(self):
        """Test parameter preparation with JSON mode."""
        messages = [{"role": "user", "content": "test"}]

        params = prepare_model_params(
            model="gpt-4o",
            messages=messages,
            json_mode=True,
        )

        assert params["response_format"] == {"type": "json_object"}

    def test_prepare_model_params_no_json_mode(self):
        """Test parameter preparation without JSON mode."""
        messages = [{"role": "user", "content": "test"}]

        params = prepare_model_params(
            model="gpt-4o",
            messages=messages,
            json_mode=False,
        )

        assert "response_format" not in params


class TestModelValidation:
    """Test model configuration validation."""

    def test_validate_model_config_standard_model(self):
        """Test validation for standard models."""
        config = {
            "temperature": 0.5,
            "top_p": 0.9,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.2,
        }

        # Should not raise any exception
        validate_model_config("gpt-4o", config)

    def test_validate_model_config_gpt5_model(self):
        """Test validation for GPT-5 model (warns about unsupported parameters)."""
        config = {
            "temperature": 0.5,
            "top_p": 0.9,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.2,
        }

        # Should not raise any exception, but should log warnings
        with patch("src.soda_curation.pipeline.openai_utils.logger") as mock_logger:
            validate_model_config(GPT5_MODEL, config)
            mock_logger.warning.assert_called()

    def test_validate_model_config_invalid_temperature(self):
        """Test validation with invalid temperature."""
        config = {"temperature": 3.0}

        with pytest.raises(ValueError, match="Temperature must be between 0 and 2"):
            validate_model_config("gpt-4o", config)

    def test_validate_model_config_invalid_top_p(self):
        """Test validation with invalid top_p."""
        config = {"top_p": 2.0}

        with pytest.raises(ValueError, match="Top_p must be between 0 and 1"):
            validate_model_config("gpt-4o", config)

    def test_validate_model_config_invalid_frequency_penalty(self):
        """Test validation with invalid frequency_penalty."""
        config = {"frequency_penalty": 3.0}

        with pytest.raises(
            ValueError, match="Frequency penalty must be between -2 and 2"
        ):
            validate_model_config("gpt-4o", config)

    def test_validate_model_config_invalid_presence_penalty(self):
        """Test validation with invalid presence_penalty."""
        config = {"presence_penalty": 3.0}

        with pytest.raises(
            ValueError, match="Presence penalty must be between -2 and 2"
        ):
            validate_model_config("gpt-4o", config)


class TestOpenAICallWithFallback:
    """Test OpenAI API call with fallback functionality."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock OpenAI client."""
        return MagicMock()

    @pytest.fixture
    def mock_response(self):
        """Create a mock OpenAI response."""
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = '{"test": "response"}'
        response.usage = MagicMock()
        response.usage.prompt_tokens = 10
        response.usage.completion_tokens = 5
        response.usage.total_tokens = 15
        return response

    def test_call_openai_with_fallback_success_primary_model(
        self, mock_client, mock_response
    ):
        """Test successful call with primary model."""
        mock_client.beta.chat.completions.parse.return_value = mock_response

        messages = [{"role": "user", "content": "test"}]

        response = call_openai_with_fallback(
            client=mock_client,
            model="gpt-4o",
            messages=messages,
            temperature=0.1,
        )

        assert response == mock_response
        mock_client.beta.chat.completions.parse.assert_called_once()

    def test_call_openai_with_fallback_context_error_fallback_success(
        self, mock_client, mock_response
    ):
        """Test fallback to GPT-5 on context length error."""
        # First call fails with context length error, second succeeds
        context_error = openai.OpenAIError("maximum context length exceeded")
        mock_client.beta.chat.completions.parse.side_effect = [
            context_error,
            mock_response,
        ]

        messages = [{"role": "user", "content": "test"}]

        with patch("src.soda_curation.pipeline.openai_utils.logger") as mock_logger:
            response = call_openai_with_fallback(
                client=mock_client,
                model="gpt-4o",
                messages=messages,
                temperature=0.1,
            )

        assert response == mock_response
        assert mock_client.beta.chat.completions.parse.call_count == 2

        # Check that fallback was logged
        mock_logger.warning.assert_called()
        mock_logger.info.assert_called()

    def test_call_openai_with_fallback_non_context_error_raises(self, mock_client):
        """Test that non-context errors are re-raised."""
        api_error = openai.OpenAIError("API key invalid")
        mock_client.beta.chat.completions.parse.side_effect = api_error

        messages = [{"role": "user", "content": "test"}]

        with pytest.raises(openai.OpenAIError, match="API key invalid"):
            call_openai_with_fallback(
                client=mock_client,
                model="gpt-4o",
                messages=messages,
            )

    def test_call_openai_with_fallback_both_models_fail(self, mock_client):
        """Test that both primary and fallback model failures are handled."""
        context_error = openai.OpenAIError("maximum context length exceeded")
        fallback_error = openai.OpenAIError("fallback model failed")
        mock_client.beta.chat.completions.parse.side_effect = [
            context_error,
            fallback_error,
        ]

        messages = [{"role": "user", "content": "test"}]

        with pytest.raises(openai.OpenAIError, match="fallback model failed"):
            call_openai_with_fallback(
                client=mock_client,
                model="gpt-4o",
                messages=messages,
            )

    def test_call_openai_with_fallback_gpt5_parameters_ignored(
        self, mock_client, mock_response
    ):
        """Test that GPT-5 parameters are properly ignored."""
        mock_client.beta.chat.completions.parse.return_value = mock_response

        messages = [{"role": "user", "content": "test"}]

        response = call_openai_with_fallback(
            client=mock_client,
            model=GPT5_MODEL,
            messages=messages,
            temperature=0.5,  # Should be ignored
            top_p=0.9,  # Should be ignored
            frequency_penalty=0.1,  # Should be ignored
            presence_penalty=0.2,  # Should be ignored
            max_tokens=1000,  # Should be ignored
        )

        assert response == mock_response

        # Check that the call was made with only basic parameters
        call_args = mock_client.beta.chat.completions.parse.call_args[1]
        assert call_args["model"] == GPT5_MODEL
        assert call_args["messages"] == messages
        assert "temperature" not in call_args
        assert "top_p" not in call_args
        assert "frequency_penalty" not in call_args
        assert "presence_penalty" not in call_args
        assert "max_tokens" not in call_args

    def test_call_openai_with_fallback_response_format(
        self, mock_client, mock_response
    ):
        """Test that response format is properly handled."""
        mock_client.beta.chat.completions.parse.return_value = mock_response

        messages = [{"role": "user", "content": "test"}]
        response_format = {"type": "json_object"}

        response = call_openai_with_fallback(
            client=mock_client,
            model="gpt-4o",
            messages=messages,
            response_format=response_format,
        )

        assert response == mock_response

        # Check that response format was included
        call_args = mock_client.beta.chat.completions.parse.call_args[1]
        assert call_args["response_format"] == response_format

    def test_call_openai_with_fallback_custom_fallback_model(
        self, mock_client, mock_response
    ):
        """Test custom fallback model."""
        context_error = openai.OpenAIError("maximum context length exceeded")
        mock_client.beta.chat.completions.parse.side_effect = [
            context_error,
            mock_response,
        ]

        messages = [{"role": "user", "content": "test"}]

        response = call_openai_with_fallback(
            client=mock_client,
            model="gpt-4o",
            messages=messages,
            fallback_model="custom-model",
        )

        assert response == mock_response
        assert mock_client.beta.chat.completions.parse.call_count == 2

        # Check that custom fallback model was used
        second_call_args = mock_client.beta.chat.completions.parse.call_args_list[1][1]
        assert second_call_args["model"] == "custom-model"


class TestConstants:
    """Test module constants."""

    def test_gpt5_model_constant(self):
        """Test GPT-5 model constant."""
        assert GPT5_MODEL == "gpt-5"

    def test_models_without_parameters_constant(self):
        """Test models without parameters constant."""
        assert GPT5_MODEL in MODELS_WITHOUT_PARAMETERS
        assert "gpt-4o" not in MODELS_WITHOUT_PARAMETERS
        assert "gpt-4o-mini" not in MODELS_WITHOUT_PARAMETERS
