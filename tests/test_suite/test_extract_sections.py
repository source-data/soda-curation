"""Tests for section extraction functionality."""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.soda_curation.pipeline.extract_sections.extract_sections_openai import (
    SectionExtractorOpenAI,
)
from src.soda_curation.pipeline.manuscript_structure.manuscript_structure import (
    ProcessingCost,
    ZipStructure,
)

# Test configurations
VALID_CONFIG = {
    "pipeline": {
        "extract_sections": {
            "openai": {
                "model": "gpt-4o",
                "temperature": 0.1,
                "top_p": 1.0,
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "prompts": {"system": "System prompt", "user": "User prompt"},
            }
        }
    }
}

INVALID_MODEL_CONFIG = {
    "pipeline": {
        "extract_sections": {"openai": {"model": "invalid-model"}},
    }
}

INVALID_PARAMS_CONFIG = {
    "pipeline": {
        "extract_sections": {
            "openai": {
                "model": "gpt-4o",
                "temperature": 3.0,
                "top_p": 2.0,
                "frequency_penalty": 3.0,
            }
        }
    }
}

# Mock response content
MOCK_SECTIONS_RESPONSE = {
    "figure_legends": "Figure 1: Test caption for figure 1. A) Panel A description. B) Panel B description.\nFigure 2: Test caption for figure 2. Multiple panels showing different aspects.",
    "data_availability": "Data Availability:\nThe data in this study are available under the following accession numbers:\n- RNA-seq data: GEO: GSE123456\n- Proteomics data: PRIDE: PXD987654\n- Code repository: GitHub: https://github.com/example/code",
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
    """Create a mock OpenAI client with properly structured responses."""
    with patch("openai.OpenAI") as mock_client:
        # Create a mock response that matches the expected structure
        instance = mock_client.return_value

        # Mock the structured response with parsed content
        mock_parsed = MagicMock()
        mock_parsed.figure_legends = MOCK_SECTIONS_RESPONSE["figure_legends"]
        mock_parsed.data_availability = MOCK_SECTIONS_RESPONSE["data_availability"]

        instance.beta.chat.completions.parse.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(parsed=mock_parsed))],
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
        ai_response_locate_captions="",  # Initialize with empty string
    )


class TestConfigValidation:
    """Test configuration validation functionality."""

    def test_valid_config(self, mock_prompt_handler):
        """Test that valid configuration is accepted."""
        extractor = SectionExtractorOpenAI(VALID_CONFIG, mock_prompt_handler)
        assert extractor.config == VALID_CONFIG

    def test_invalid_model(self, mock_prompt_handler):
        """Test that invalid model raises ValueError."""
        with pytest.raises(ValueError, match="Invalid model"):
            SectionExtractorOpenAI(INVALID_MODEL_CONFIG, mock_prompt_handler)

    def test_invalid_parameters(self, mock_prompt_handler):
        """Test that invalid parameters raise ValueError."""
        with pytest.raises(ValueError):
            SectionExtractorOpenAI(INVALID_PARAMS_CONFIG, mock_prompt_handler)


class TestSectionExtraction:
    """Test section extraction functionality."""

    def test_successful_extraction(
        self, mock_openai_client, mock_prompt_handler, zip_structure
    ):
        """Test successful extraction of both sections."""
        extractor = SectionExtractorOpenAI(VALID_CONFIG, mock_prompt_handler)
        figure_legends, data_availability, updated_zip = extractor.extract_sections(
            "test content", zip_structure
        )

        assert MOCK_SECTIONS_RESPONSE["figure_legends"] == figure_legends
        assert MOCK_SECTIONS_RESPONSE["data_availability"] == data_availability
        assert updated_zip.ai_response_locate_captions == figure_legends
        assert hasattr(updated_zip.cost.extract_sections, "total_tokens")

    def test_api_error_handling(
        self, mock_openai_client, mock_prompt_handler, zip_structure
    ):
        """Test handling of API errors."""
        mock_openai_client.return_value.beta.chat.completions.parse.side_effect = (
            Exception("API Error")
        )

        extractor = SectionExtractorOpenAI(VALID_CONFIG, mock_prompt_handler)

        with pytest.raises(Exception, match="API Error"):
            extractor.extract_sections("test content", zip_structure)

    def test_empty_content(
        self, mock_openai_client, mock_prompt_handler, zip_structure
    ):
        """Test handling of empty document content."""
        mock_openai_client.return_value.beta.chat.completions.parse.side_effect = (
            Exception("Empty content")
        )

        extractor = SectionExtractorOpenAI(VALID_CONFIG, mock_prompt_handler)

        with pytest.raises(Exception, match="Empty content"):
            extractor.extract_sections("", zip_structure)


class TestResponseParsing:
    """Test parsing of AI responses."""

    def test_parse_valid_response(self, mock_prompt_handler):
        """Test parsing of valid response format."""
        extractor = SectionExtractorOpenAI(VALID_CONFIG, mock_prompt_handler)
        result = extractor._parse_response(json.dumps(MOCK_SECTIONS_RESPONSE))

        assert "figure_legends" in result
        assert "data_availability" in result
        assert "Figure 1:" in result["figure_legends"]
        assert "Data Availability" in result["data_availability"]

    def test_parse_invalid_response(self, mock_prompt_handler):
        """Test parsing of invalid response."""
        extractor = SectionExtractorOpenAI(VALID_CONFIG, mock_prompt_handler)
        result = extractor._parse_response("Invalid JSON content")

        assert result == {
            "figure_legends": "",
            "data_availability": "",
        }
