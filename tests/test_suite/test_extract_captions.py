"""Tests for caption extraction functionality."""

from unittest.mock import MagicMock, patch

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from src.soda_curation.pipeline.extract_captions.extract_captions_openai import (
    FigureCaptionExtractorOpenAI,
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
def mock_prompt_handler():
    """Create a mock prompt handler."""
    mock_handler = MagicMock()
    mock_handler.get_prompt.return_value = {
        "system": "System prompt",
        "user": "User prompt",
    }
    return mock_handler


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    with patch("openai.OpenAI") as mock_client:
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

    def test_valid_config(self, mock_prompt_handler):
        """Test that valid configuration is accepted."""
        extractor = FigureCaptionExtractorOpenAI(VALID_CONFIG, mock_prompt_handler)
        assert extractor.config == VALID_CONFIG
        assert extractor.model == "gpt-4o"

    def test_invalid_model(self, mock_prompt_handler):
        """Test that invalid model raises ValueError."""
        with pytest.raises(ValueError, match="Invalid model"):
            FigureCaptionExtractorOpenAI(INVALID_MODEL_CONFIG, mock_prompt_handler)

    def test_invalid_parameters(self, mock_prompt_handler):
        """Test that invalid parameters raise ValueError."""
        with pytest.raises(ValueError):
            FigureCaptionExtractorOpenAI(INVALID_PARAMS_CONFIG, mock_prompt_handler)


class TestLocateCaptions:
    """Test caption location functionality."""

    def test_successful_location(
        self, mock_openai_client, mock_prompt_handler, zip_structure
    ):
        """Test successful caption location."""
        extractor = FigureCaptionExtractorOpenAI(VALID_CONFIG, mock_prompt_handler)

        response = extractor.locate_captions(
            "test content", zip_structure, 2, "Figure 1, Figure 2"
        )

        assert response == MOCK_LOCATE_RESPONSE
        mock_prompt_handler.get_prompt.assert_called_once()
        assert mock_openai_client.return_value.chat.completions.create.called

    def test_location_error_handling(
        self, mock_openai_client, mock_prompt_handler, zip_structure
    ):
        """Test error handling during caption location."""
        mock_openai_client.return_value.chat.completions.create.side_effect = Exception(
            "API Error"
        )

        extractor = FigureCaptionExtractorOpenAI(VALID_CONFIG, mock_prompt_handler)
        with pytest.raises(Exception):
            extractor.locate_captions(
                "test content", zip_structure, 2, "Figure 1, Figure 2"
            )


class TestExtractIndividualCaptions:
    """Test individual caption extraction functionality."""

    @pytest.mark.parametrize(
        "response",
        [
            MOCK_EXTRACT_RESPONSE_CLEAN,
            MOCK_EXTRACT_RESPONSE_WITH_TEXT,
            MOCK_EXTRACT_RESPONSE_NO_CODEBLOCK,
            MOCK_EXTRACT_RESPONSE_WITH_TEXT_NO_CODEBLOCK,
        ],
    )
    def test_successful_extraction(
        self, mock_openai_client, mock_prompt_handler, zip_structure, response
    ):
        """Test successful extraction of individual captions."""
        # Configure mock to return different response formats
        mock_openai_client.return_value.chat.completions.create.return_value = (
            MagicMock(choices=[MagicMock(message=MagicMock(content=response))])
        )

        extractor = FigureCaptionExtractorOpenAI(VALID_CONFIG, mock_prompt_handler)
        result = extractor.extract_individual_captions(
            "caption section", zip_structure, 2, "Figure 1, Figure 2"
        )

        # Parse result to verify structure
        parsed_result = extractor._parse_captions(result)
        assert isinstance(parsed_result, dict)
        assert "Figure 1" in parsed_result
        assert parsed_result["Figure 1"]["title"] == "Test caption for figure 1"

    def test_extraction_error_handling(
        self, mock_openai_client, mock_prompt_handler, zip_structure
    ):
        """Test error handling during individual caption extraction."""
        mock_openai_client.return_value.chat.completions.create.side_effect = Exception(
            "API Error"
        )

        extractor = FigureCaptionExtractorOpenAI(VALID_CONFIG, mock_prompt_handler)
        with pytest.raises(Exception):
            extractor.extract_individual_captions(
                "caption section", zip_structure, 2, "Figure 1, Figure 2"
            )


class TestResponseParsing:
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
    def test_parse_valid_responses(self, mock_prompt_handler, response):
        """Test parsing of various valid response formats."""
        extractor = FigureCaptionExtractorOpenAI(VALID_CONFIG, mock_prompt_handler)
        result = extractor._parse_captions(response)

        assert isinstance(result, dict)
        assert "Figure 1" in result
        assert result["Figure 1"]["title"] == "Test caption for figure 1"
        assert (
            result["Figure 1"]["caption"]
            == "A) Panel A description. B) Panel B description."
        )

    def test_parse_unparseable_response(self, mock_prompt_handler):
        """Test handling of unparseable response."""
        extractor = FigureCaptionExtractorOpenAI(VALID_CONFIG, mock_prompt_handler)
        result = extractor._parse_captions(MOCK_EXTRACT_RESPONSE_UNPARSEABLE)
        assert result == {}
