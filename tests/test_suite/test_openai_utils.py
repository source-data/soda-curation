"""Tests for OpenAI utility helpers (GPT-5–first; chunking on context limits)."""

import json
from unittest.mock import MagicMock, patch

import openai
import pytest
from pydantic import ValidationError

from src.soda_curation.pipeline.extract_captions.extract_captions_openai import (
    CaptionExtraction,
)
from src.soda_curation.pipeline.openai_utils import (
    GPT5_MODEL,
    MODELS_WITHOUT_PARAMETERS,
    call_openai,
    extract_first_json_value,
    is_context_length_error,
    prepare_model_params,
    validate_model_config,
)


class TestExtractFirstJsonValue:
    """Decode first JSON object when models append junk after valid JSON."""

    def test_accepts_trailing_text_after_json_object(self):
        payload = '{"figure_label": "Figure 1", "caption_title": "", "figure_caption": "x", "is_verbatim": false}'
        raw = payload + "\n\nAdditional commentary."
        assert (
            extract_first_json_value(raw, operation="test.op")["figure_label"]
            == "Figure 1"
        )

    def test_strips_markdown_fence(self):
        inner = '{"a": 1}'
        raw = "```json\n" + inner + "\n```"
        assert extract_first_json_value(raw)["a"] == 1

    def test_trailing_chars_trigger_strict_parse_failure_then_lenient_succeeds(self):
        """Reproduce pydantic json_invalid trailing characters; lenient path validates."""
        bad = (
            '{"figure_label":"F","caption_title":"","figure_caption":"c","is_verbatim":false}'
            "\nextras"
        )
        with pytest.raises(ValidationError):
            CaptionExtraction.model_validate_json(bad)
        parsed = CaptionExtraction.model_validate(extract_first_json_value(bad))
        assert parsed.figure_label == "F"


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

    def test_prepare_model_params_gpt5_dot_release_no_sampling(self):
        """gpt-5.4-mini style ids must not send max_tokens (API uses different knobs)."""
        messages = [{"role": "user", "content": "test"}]
        params = prepare_model_params(
            model="gpt-5.4-mini",
            messages=messages,
            temperature=0.1,
            max_tokens=4096,
            json_mode=False,
        )
        assert params == {"model": "gpt-5.4-mini", "messages": messages}

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
        """GPT-5 family skips sampling validation; YAML may still list unused keys."""
        config = {
            "temperature": 0.5,
            "top_p": 0.9,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.2,
        }

        with patch("src.soda_curation.pipeline.openai_utils.logger") as mock_logger:
            validate_model_config(GPT5_MODEL, config)
            mock_logger.warning.assert_not_called()

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


class TestCallOpenAI:
    """Exercise ``call_openai`` (single request + chunking on context overflow)."""

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

    def test_call_openai_success_primary_model(self, mock_client, mock_response):
        """Successful call with the configured model."""
        mock_client.beta.chat.completions.parse.return_value = mock_response

        messages = [{"role": "user", "content": "test"}]

        response = call_openai(
            client=mock_client,
            model="gpt-4o",
            messages=messages,
            temperature=0.1,
        )

        assert response == mock_response
        mock_client.beta.chat.completions.parse.assert_called_once()

    def test_call_openai_context_error_invokes_chunking(
        self, mock_client, mock_response
    ):
        """Context-length API errors delegate to chunked execution (same model)."""
        context_error = openai.OpenAIError("maximum context length exceeded")
        mock_client.beta.chat.completions.parse.side_effect = [context_error]

        messages = [{"role": "user", "content": "test"}]

        with patch(
            "src.soda_curation.pipeline.openai_utils._call_openai_with_chunking",
            return_value=mock_response,
        ) as mock_chunk:
            with patch("src.soda_curation.pipeline.openai_utils.logger"):
                response = call_openai(
                    client=mock_client,
                    model="gpt-4o",
                    messages=messages,
                    temperature=0.1,
                )

        assert response == mock_response
        mock_client.beta.chat.completions.parse.assert_called_once()
        mock_chunk.assert_called_once()

    def test_call_openai_non_context_error_raises(self, mock_client):
        """Non-context errors are re-raised."""
        api_error = openai.OpenAIError("API key invalid")
        mock_client.beta.chat.completions.parse.side_effect = api_error

        messages = [{"role": "user", "content": "test"}]

        with pytest.raises(openai.OpenAIError, match="API key invalid"):
            call_openai(
                client=mock_client,
                model="gpt-4o",
                messages=messages,
            )

    def test_call_openai_context_error_chunking_propagates(self, mock_client):
        """If chunking fails after a context-length error, the chunking error propagates."""
        context_error = openai.OpenAIError("maximum context length exceeded")
        mock_client.beta.chat.completions.parse.side_effect = [context_error]
        chunk_error = openai.OpenAIError("chunking failed")

        messages = [{"role": "user", "content": "test"}]

        with patch(
            "src.soda_curation.pipeline.openai_utils._call_openai_with_chunking",
            side_effect=chunk_error,
        ):
            with pytest.raises(openai.OpenAIError, match="chunking failed"):
                call_openai(
                    client=mock_client,
                    model="gpt-4o",
                    messages=messages,
                )

    def test_call_openai_gpt5_parameters_ignored(self, mock_client, mock_response):
        """GPT-5 family omits sampling / max_tokens in the API payload."""
        mock_client.beta.chat.completions.parse.return_value = mock_response

        messages = [{"role": "user", "content": "test"}]

        response = call_openai(
            client=mock_client,
            model=GPT5_MODEL,
            messages=messages,
            temperature=0.5,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.2,
            max_tokens=1000,
        )

        assert response == mock_response

        call_args = mock_client.beta.chat.completions.parse.call_args[1]
        assert call_args["model"] == GPT5_MODEL
        assert call_args["messages"] == messages
        assert "temperature" not in call_args
        assert "top_p" not in call_args
        assert "frequency_penalty" not in call_args
        assert "presence_penalty" not in call_args
        assert "max_tokens" not in call_args

    def test_call_openai_response_format(self, mock_client, mock_response):
        """Response format is forwarded to the client."""
        mock_client.beta.chat.completions.parse.return_value = mock_response

        messages = [{"role": "user", "content": "test"}]
        response_format = {"type": "json_object"}

        response = call_openai(
            client=mock_client,
            model="gpt-4o",
            messages=messages,
            response_format=response_format,
        )

        assert response == mock_response

        call_args = mock_client.beta.chat.completions.parse.call_args[1]
        assert call_args["response_format"] == response_format

    def test_call_openai_lenient_when_strict_parse_validation_error(self, mock_client):
        """If ``parse()`` fails on assistant text, retry via ``create`` + first-value JSON + schema."""
        try:
            CaptionExtraction.model_validate_json(
                '{"figure_label":"F","caption_title":"","figure_caption":"c","is_verbatim":false}'
                "\ntrailing"
            )
        except ValidationError as exc:
            parse_err = exc
        mock_client.beta.chat.completions.parse.side_effect = parse_err

        completion = MagicMock()
        completion.choices = [MagicMock()]
        completion.choices[0].message.content = (
            '{"figure_label":"Fig","caption_title":"","figure_caption":"body","is_verbatim":false}'
            "\n\nextra"
        )
        completion.usage = MagicMock()
        completion.usage.prompt_tokens = 1
        completion.usage.completion_tokens = 2
        completion.usage.total_tokens = 3
        mock_client.chat.completions.create.return_value = completion

        out = call_openai(
            client=mock_client,
            model="gpt-4o",
            messages=[{"role": "user", "content": "test"}],
            response_format=CaptionExtraction,
            enable_chunking=False,
            operation="test.op",
        )

        mock_client.chat.completions.create.assert_called_once()
        assert out.choices[0].message.parsed.figure_caption == "body"
        assert out.choices[0].message.parsed.figure_label == "Fig"


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
