"""Tests for caption extraction functionality."""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.soda_curation.pipeline.extract_captions.extract_captions_openai import (
    FigureCaptionExtractorOpenAI,
)
from src.soda_curation.pipeline.manuscript_structure.manuscript_structure import (
    ProcessingCost,
    ZipStructure,
)

# Test configurations
VALID_CONFIG = {
    "pipeline": {
        "extract_individual_captions": {
            "openai": {
                "model": "gpt-4o",
                "temperature": 0.1,
                "top_p": 1.0,
                "prompts": {"system": "System prompt", "user": "User prompt"},
            }
        }
    }
}

INVALID_MODEL_CONFIG = {
    "pipeline": {
        "extract_individual_captions": {"openai": {"model": "invalid-model"}},
    }
}

INVALID_PARAMS_CONFIG = {
    "pipeline": {
        "extract_individual_captions": {
            "openai": {
                "model": "gpt-4o",
                "temperature": 3.0,
                "top_p": 2.0,
            }
        }
    }
}

# Mock response content
MOCK_EXTRACT_RESPONSE = {
    "figures": [
        {
            "figure_label": "Figure 1",
            "caption_title": "Test caption for figure 1",
            "figure_caption": "A) Panel A description. B) Panel B description.",
        },
        {
            "figure_label": "Figure 2",
            "caption_title": "Test caption for figure 2",
            "figure_caption": "Multiple panels showing different aspects.",
        },
    ]
}


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
        # Create a properly structured mock response
        instance = mock_client.return_value
        instance.beta.chat.completions.parse.return_value = MagicMock(
            choices=[
                MagicMock(message=MagicMock(content=json.dumps(MOCK_EXTRACT_RESPONSE)))
            ],
            usage=MagicMock(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
            ),
        )
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
        ai_response_extract_individual_captions="",
    )


class TestConfigValidation:
    """Test configuration validation functionality."""

    def test_valid_config(self, mock_prompt_handler):
        """Test that valid configuration is accepted."""
        extractor = FigureCaptionExtractorOpenAI(VALID_CONFIG, mock_prompt_handler)
        assert extractor.config == VALID_CONFIG

    def test_invalid_model(self, mock_prompt_handler):
        """Test that invalid model raises ValueError."""
        with pytest.raises(ValueError, match="Invalid model"):
            FigureCaptionExtractorOpenAI(INVALID_MODEL_CONFIG, mock_prompt_handler)

    def test_invalid_parameters(self, mock_prompt_handler):
        """Test that invalid parameters raise ValueError."""
        with pytest.raises(ValueError):
            FigureCaptionExtractorOpenAI(INVALID_PARAMS_CONFIG, mock_prompt_handler)


class TestIndividualCaptionExtraction:
    """Test individual caption extraction functionality."""

    def test_successful_extraction(
        self, mock_openai_client, mock_prompt_handler, zip_structure
    ):
        """Test successful extraction of individual captions."""
        extractor = FigureCaptionExtractorOpenAI(VALID_CONFIG, mock_prompt_handler)
        result = extractor.extract_individual_captions("test content", zip_structure)

        assert isinstance(result, ZipStructure)
        assert result.ai_response_extract_individual_captions == json.dumps(
            MOCK_EXTRACT_RESPONSE
        )

        # Verify figures were updated correctly
        for figure_data in MOCK_EXTRACT_RESPONSE["figures"]:
            matching_figures = [
                f
                for f in result.figures
                if f.figure_label == figure_data["figure_label"]
            ]
            if matching_figures:
                assert matching_figures[0].caption_title == figure_data["caption_title"]
                assert (
                    matching_figures[0].figure_caption == figure_data["figure_caption"]
                )

    def test_api_error_handling(
        self, mock_openai_client, mock_prompt_handler, zip_structure
    ):
        """Test handling of API errors."""
        mock_openai_client.return_value.beta.chat.completions.parse.side_effect = (
            Exception("API Error")
        )

        extractor = FigureCaptionExtractorOpenAI(VALID_CONFIG, mock_prompt_handler)
        result = extractor.extract_individual_captions("test content", zip_structure)

        assert isinstance(result, ZipStructure)
        # Error cases should preserve empty string
        assert result.ai_response_extract_individual_captions == ""

    def test_empty_content(
        self, mock_openai_client, mock_prompt_handler, zip_structure
    ):
        """Test handling of empty content."""
        # Make sure empty content causes an error
        mock_openai_client.return_value.beta.chat.completions.parse.side_effect = (
            Exception("Empty content")
        )

        extractor = FigureCaptionExtractorOpenAI(VALID_CONFIG, mock_prompt_handler)
        result = extractor.extract_individual_captions("", zip_structure)

        assert isinstance(result, ZipStructure)
        # Error cases should preserve empty string
        assert result.ai_response_extract_individual_captions == ""


class TestResponseParsing:
    """Test parsing of AI responses."""

    def test_parse_valid_response(self, mock_prompt_handler):
        """Test parsing of valid response format."""
        extractor = FigureCaptionExtractorOpenAI(VALID_CONFIG, mock_prompt_handler)
        result = extractor._parse_response(json.dumps(MOCK_EXTRACT_RESPONSE))

        assert len(result) > 0
        assert "figures" in result
        assert isinstance(result["figures"], list)
        assert len(result["figures"]) == 2
        assert all(
            key in result["figures"][0]
            for key in ["figure_label", "caption_title", "figure_caption"]
        )

    def test_parse_invalid_response(self, mock_prompt_handler):
        """Test parsing of invalid response."""
        extractor = FigureCaptionExtractorOpenAI(VALID_CONFIG, mock_prompt_handler)
        result = extractor._parse_response("Invalid JSON content")

        assert result == {}
