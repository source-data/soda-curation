"""Tests for caption extraction functionality."""

from unittest.mock import MagicMock, patch

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from src.soda_curation.pipeline.extract_captions.extract_captions_openai import (
    FigureCaptionExtractorGpt,
)
from src.soda_curation.pipeline.manuscript_structure.manuscript_structure import (
    ProcessingCost,
    ZipStructure,
)

# Test data
VALID_CONFIG = {
    "openai": {
        "model": "gpt-4o",
        "temperature": 0.1,
        "top_p": 1.0,
        "max_tokens": 2048,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "seed": None,
        "stop": None,
        "user": None,
        "json_mode": True,
        "response_format": {"type": "text"},
    }
}

INVALID_MODEL_CONFIG = {
    "openai": {
        "model": "invalid-model",  # Invalid model
        "temperature": 0.1,
        "top_p": 1.0,
    }
}

INVALID_PARAMS_CONFIG = {
    "openai": {
        "model": "gpt-4o",
        "temperature": 3.0,  # Invalid temperature
        "top_p": 2.0,  # Invalid top_p
        "frequency_penalty": 3.0,  # Invalid frequency penalty
        "presence_penalty": -3.0,  # Invalid presence penalty
    }
}

# Mock API responses
MOCK_LOCATE_RESPONSE = """Here are the figure captions:
Figure 1: Test caption for figure 1. A) Panel A description. B) Panel B description.
Figure 2: Test caption for figure 2. Multiple panels showing different aspects."""

# Different formats of extract responses
MOCK_EXTRACT_RESPONSE_CLEAN = """```json
{
    "Figure 1": {
        "title": "Test caption for figure 1",
        "caption": "A) Panel A description. B) Panel B description."
    },
    "Figure 2": {
        "title": "Test caption for figure 2",
        "caption": "Multiple panels showing different aspects."
    }
}
```"""

MOCK_EXTRACT_RESPONSE_WITH_TEXT = """Some extra text here ```json
{
    "Figure 1": {
        "title": "Test caption for figure 1",
        "caption": "A) Panel A description. B) Panel B description."
    },
    "Figure 2": {
        "title": "Test caption for figure 2",
        "caption": "Multiple panels showing different aspects."
    }
}
```"""

MOCK_EXTRACT_RESPONSE_NO_CODEBLOCK = """
{
    "Figure 1": {
        "title": "Test caption for figure 1",
        "caption": "A) Panel A description. B) Panel B description."
    },
    "Figure 2": {
        "title": "Test caption for figure 2",
        "caption": "Multiple panels showing different aspects."
    }
}
"""

MOCK_EXTRACT_RESPONSE_WITH_TEXT_NO_CODEBLOCK = """Some extra text here
{
    "Figure 1": {
        "title": "Test caption for figure 1",
        "caption": "A) Panel A description. B) Panel B description."
    },
    "Figure 2": {
        "title": "Test caption for figure 2",
        "caption": "Multiple panels showing different aspects."
    }
}
"""

MOCK_EXTRACT_RESPONSE_UNPARSEABLE = (
    """Some text that is not JSON and cannot be parsed as such."""
)


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    with patch("openai.OpenAI") as mock_client:
        # Mock successful response
        mock_message = ChatCompletionMessage(
            content=MOCK_LOCATE_RESPONSE,
            role="assistant",
            function_call=None,
            tool_calls=None,
        )
        mock_choice = Choice(
            finish_reason="stop",
            index=0,
            message=mock_message,
        )
        mock_completion = ChatCompletion(
            id="123",
            choices=[mock_choice],
            created=1234567890,
            model="gpt-4o",
            object="chat.completion",
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        )

        mock_client.return_value.chat.completions.create.return_value = mock_completion
        yield mock_client


@pytest.fixture
def zip_structure():
    """Create a basic ZipStructure instance for testing."""
    return ZipStructure(
        manuscript_id="test_manuscript",
        xml="test.xml",
        docx="test.docx",
        pdf="test.pdf",
        figures=[],
        errors=[],
        cost=ProcessingCost(),
    )


class TestConfigValidation:
    """Test configuration validation functionality."""

    def test_valid_config(self):
        """Test that valid configuration is accepted."""
        extractor = FigureCaptionExtractorGpt(VALID_CONFIG)
        assert extractor.config == VALID_CONFIG
        assert extractor.model == "gpt-4o"

    def test_invalid_model(self):
        """Test that invalid model raises ValueError."""
        with pytest.raises(ValueError, match="Invalid model"):
            FigureCaptionExtractorGpt(INVALID_MODEL_CONFIG)

    def test_invalid_parameters(self):
        """Test that invalid parameters raise ValueError."""
        with pytest.raises(ValueError, match="Invalid parameter values"):
            FigureCaptionExtractorGpt(INVALID_PARAMS_CONFIG)

    def test_missing_config(self):
        """Test that missing configuration raises ValueError."""
        with pytest.raises(ValueError):
            FigureCaptionExtractorGpt({})


class TestAPIInteraction:
    """Test OpenAI API interaction functionality."""

    def test_successful_api_call(self, mock_openai_client, zip_structure):
        """Test successful API call for caption extraction."""
        extractor = FigureCaptionExtractorGpt(VALID_CONFIG)
        response = extractor._locate_figure_captions(
            "test content", 2, "Figure 1, Figure 2"
        )
        assert response == MOCK_LOCATE_RESPONSE
        mock_openai_client.assert_called_once()

    def test_api_error_handling(self, mock_openai_client, zip_structure):
        """Test handling of API errors."""
        mock_openai_client.return_value.chat.completions.create.side_effect = Exception(
            "API Error"
        )
        extractor = FigureCaptionExtractorGpt(VALID_CONFIG)

        with pytest.raises(Exception, match="API Error"):
            extractor._locate_figure_captions("test content", 2, "Figure 1, Figure 2")

    @patch("backoff.on_exception")  # Test retry mechanism
    def test_api_retry_mechanism(self, mock_backoff, mock_openai_client, zip_structure):
        """Test that API calls are retried on failure."""
        mock_openai_client.return_value.chat.completions.create.side_effect = [
            Exception("API Error"),
            MagicMock(
                choices=[MagicMock(message=MagicMock(content=MOCK_LOCATE_RESPONSE))]
            ),
        ]

        extractor = FigureCaptionExtractorGpt(VALID_CONFIG)
        response = extractor._locate_figure_captions(
            "test content", 2, "Figure 1, Figure 2"
        )
        assert response == MOCK_LOCATE_RESPONSE
        assert mock_openai_client.return_value.chat.completions.create.call_count == 2


class TestOutputParsing:
    """Test parsing of AI responses."""

    @pytest.mark.parametrize(
        "response",
        [
            MOCK_EXTRACT_RESPONSE_CLEAN,
            MOCK_EXTRACT_RESPONSE_WITH_TEXT,
            MOCK_EXTRACT_RESPONSE_NO_CODEBLOCK,
            MOCK_EXTRACT_RESPONSE_WITH_TEXT_NO_CODEBLOCK,
        ],
    )
    def test_parse_valid_responses(self, response):
        """Test parsing of various valid response formats."""
        extractor = FigureCaptionExtractorGpt(VALID_CONFIG)
        result = extractor._parse_response(response)
        assert isinstance(result, dict)
        assert "Figure 1" in result
        assert result["Figure 1"]["title"] == "Test caption for figure 1"
        assert (
            result["Figure 1"]["caption"]
            == "A) Panel A description. B) Panel B description."
        )

    def test_parse_unparseable_response(self, zip_structure):
        """Test handling of unparseable response."""
        extractor = FigureCaptionExtractorGpt(VALID_CONFIG)
        result = extractor._parse_response(MOCK_EXTRACT_RESPONSE_UNPARSEABLE)
        assert result == {}
        assert (
            "Failed to parse response" in zip_structure.errors[0]
            if zip_structure.errors
            else False
        )


class TestZipStructureUpdate:
    """Test updating of ZipStructure with extracted captions."""

    def test_update_zip_structure(self, zip_structure):
        """Test that ZipStructure is correctly updated with extracted captions."""
        extractor = FigureCaptionExtractorGpt(VALID_CONFIG)
        updated_structure = extractor.extract_captions(
            "test content", zip_structure, 2, "Figure 1, Figure 2"
        )

        assert updated_structure.ai_response_locate_captions is not None
        assert updated_structure.ai_response_extract_captions is not None
        assert len(updated_structure.errors) == 0

        # Verify cost tracking
        assert updated_structure.cost.extract_captions.prompt_tokens > 0
        assert updated_structure.cost.extract_captions.completion_tokens > 0

    def test_error_handling_in_structure(self, zip_structure, mock_openai_client):
        """Test that errors are properly recorded in ZipStructure."""
        mock_openai_client.return_value.chat.completions.create.side_effect = Exception(
            "Test Error"
        )

        extractor = FigureCaptionExtractorGpt(VALID_CONFIG)
        updated_structure = extractor.extract_captions(
            "test content", zip_structure, 2, "Figure 1, Figure 2"
        )

        assert len(updated_structure.errors) > 0
        assert "Test Error" in str(updated_structure.errors[0])

    def test_unparseable_response_error_recording(
        self, zip_structure, mock_openai_client
    ):
        """Test that unparseable responses are recorded as errors."""
        mock_openai_client.return_value.chat.completions.create.return_value = (
            MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(content=MOCK_EXTRACT_RESPONSE_UNPARSEABLE)
                    )
                ]
            )
        )

        extractor = FigureCaptionExtractorGpt(VALID_CONFIG)
        updated_structure = extractor.extract_captions(
            "test content", zip_structure, 2, "Figure 1, Figure 2"
        )

        assert len(updated_structure.errors) > 0
        assert "Failed to parse response" in str(updated_structure.errors[0])


class TestEndToEnd:
    """End-to-end tests for caption extraction."""

    @pytest.mark.parametrize(
        "extract_response",
        [
            MOCK_EXTRACT_RESPONSE_CLEAN,
            MOCK_EXTRACT_RESPONSE_WITH_TEXT,
            MOCK_EXTRACT_RESPONSE_NO_CODEBLOCK,
            MOCK_EXTRACT_RESPONSE_WITH_TEXT_NO_CODEBLOCK,
        ],
    )
    def test_full_extraction_process(
        self, zip_structure, mock_openai_client, extract_response
    ):
        """Test the complete caption extraction process with different response formats."""
        extractor = FigureCaptionExtractorGpt(VALID_CONFIG)

        # Configure mock responses
        mock_openai_client.return_value.chat.completions.create.side_effect = [
            MagicMock(
                choices=[MagicMock(message=MagicMock(content=MOCK_LOCATE_RESPONSE))]
            ),
            MagicMock(choices=[MagicMock(message=MagicMock(content=extract_response))]),
        ]

        updated_structure = extractor.extract_captions(
            "test content", zip_structure, 2, "Figure 1, Figure 2"
        )

        # Verify the complete process
        assert updated_structure.ai_response_locate_captions == MOCK_LOCATE_RESPONSE
        assert updated_structure.ai_response_extract_captions is not None
        assert len(updated_structure.errors) == 0
        assert updated_structure.cost.extract_captions.total_tokens > 0
