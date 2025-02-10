"""Tests for data availability extraction functionality."""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.soda_curation.pipeline.data_availability.data_availability_openai import (
    DataAvailabilityExtractorOpenAI,
)
from src.soda_curation.pipeline.manuscript_structure.manuscript_structure import (
    ProcessingCost,
    ZipStructure,
)

# Test configurations
VALID_CONFIG = {
    "pipeline": {
        "locate_data_availability": {
            "openai": {
                "model": "gpt-4o",
                "temperature": 0.1,
                "top_p": 1.0,
                "prompts": {"system": "System prompt", "user": "User prompt"},
            }
        },
        "extract_data_sources": {
            "openai": {
                "model": "gpt-4o",
                "temperature": 0.1,
                "top_p": 1.0,
                "prompts": {"system": "System prompt", "user": "User prompt"},
            }
        },
    }
}

INVALID_MODEL_CONFIG = {
    "pipeline": {
        "locate_data_availability": {"openai": {"model": "invalid-model"}},
        "extract_data_sources": {"openai": {"model": "invalid-model"}},
    }
}

INVALID_PARAMS_CONFIG = {
    "pipeline": {
        "locate_data_availability": {
            "openai": {"model": "gpt-4o", "temperature": 3.0, "top_p": 2.0}
        },
        "extract_data_sources": {
            "openai": {"model": "gpt-4o", "temperature": 3.0, "top_p": 2.0}
        },
    }
}

# Mock API responses
MOCK_SECTION_RESPONSE = {
    "data_availability": """
    Data Availability:
    The data in this study are available under the following accession numbers:
    - RNA-seq data: GEO: GSE123456
    - Proteomics data: PRIDE: PXD987654
    - Code repository: GitHub: https://github.com/example/code
    """
}

MOCK_SOURCES_RESPONSE = {
    "data_sources": [
        {"database": "GEO", "accession_number": "GSE123456", "url": None},
        {"database": "PRIDE", "accession_number": "PXD987654", "url": None},
        {
            "database": "GitHub",
            "accession_number": "example/code",
            "url": "https://github.com/example/code",
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
        mock_resp1 = MagicMock()
        mock_resp1.choices[0].message.content = json.dumps(MOCK_SECTION_RESPONSE)
        mock_resp2 = MagicMock()
        mock_resp2.choices[0].message.content = json.dumps(MOCK_SOURCES_RESPONSE)

        mock_client.return_value.beta.chat.completions.parse.side_effect = [
            mock_resp1,
            mock_resp2,
        ]
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
        extractor = DataAvailabilityExtractorOpenAI(VALID_CONFIG, mock_prompt_handler)
        assert extractor.config == VALID_CONFIG

    def test_invalid_model(self, mock_prompt_handler):
        """Test that invalid model raises ValueError."""
        with pytest.raises(ValueError, match="Invalid model"):
            DataAvailabilityExtractorOpenAI(INVALID_MODEL_CONFIG, mock_prompt_handler)

    def test_invalid_parameters(self, mock_prompt_handler):
        """Test that invalid parameters raise ValueError."""
        with pytest.raises(ValueError):
            DataAvailabilityExtractorOpenAI(INVALID_PARAMS_CONFIG, mock_prompt_handler)


class TestDataAvailabilityExtraction:
    """Test data availability extraction functionality."""

    def test_successful_extraction(
        self, mock_openai_client, mock_prompt_handler, zip_structure
    ):
        """Test successful extraction of data availability information."""
        extractor = DataAvailabilityExtractorOpenAI(VALID_CONFIG, mock_prompt_handler)
        result = extractor.extract_data_availability("test content", zip_structure)

        assert isinstance(result, ZipStructure)
        assert "data_availability" in result.__dict__
        assert (
            result.data_availability["section_text"]
            == MOCK_SECTION_RESPONSE["data_availability"]
        )
        assert len(result.data_availability["data_sources"]) == 3
        assert result.data_availability["data_sources"][0]["database"] == "GEO"

    def test_no_section_found(
        self, mock_openai_client, mock_prompt_handler, zip_structure
    ):
        """Test handling when no data availability section is found."""
        mock_openai_client.return_value.beta.chat.completions.parse.side_effect = [
            MagicMock(choices=[MagicMock(message=MagicMock(content=""))]),
        ]

        extractor = DataAvailabilityExtractorOpenAI(VALID_CONFIG, mock_prompt_handler)
        result = extractor.extract_data_availability("test content", zip_structure)

        assert isinstance(result, ZipStructure)
        assert result.data_availability["section_text"] == ""
        assert result.data_availability["data_sources"] == []

    def test_api_error_handling(
        self, mock_openai_client, mock_prompt_handler, zip_structure
    ):
        """Test handling of API errors."""
        mock_openai_client.return_value.beta.chat.completions.parse.side_effect = (
            Exception("API Error")
        )

        extractor = DataAvailabilityExtractorOpenAI(VALID_CONFIG, mock_prompt_handler)
        result = extractor.extract_data_availability("test content", zip_structure)

        assert isinstance(result, ZipStructure)
        assert result.data_availability["section_text"] == ""
        assert result.data_availability["data_sources"] == []

    def test_token_usage_tracking(
        self, mock_openai_client, mock_prompt_handler, zip_structure
    ):
        """Test that token usage is properly tracked."""
        mock_resp = MagicMock()
        mock_resp.usage = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }
        mock_openai_client.return_value.beta.chat.completions.parse.return_value = (
            mock_resp
        )

        extractor = DataAvailabilityExtractorOpenAI(VALID_CONFIG, mock_prompt_handler)
        result = extractor.extract_data_availability("test content", zip_structure)

        assert result.cost.locate_data_availability.total_tokens > 0
        assert result.cost.extract_data_sources.total_tokens > 0


class TestResponseParsing:
    """Test parsing of AI responses."""

    def test_parse_valid_json_response(self, mock_prompt_handler):
        """Test parsing of valid JSON response."""
        extractor = DataAvailabilityExtractorOpenAI(VALID_CONFIG, mock_prompt_handler)
        response = """```json
        [{"database": "GEO", "accession_number": "GSE123456", "url": null}]
        ```"""
        result = extractor._parse_response(response)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["database"] == "GEO"

    def test_parse_response_no_code_block(self, mock_prompt_handler):
        """Test parsing of JSON response without code block."""
        extractor = DataAvailabilityExtractorOpenAI(VALID_CONFIG, mock_prompt_handler)
        response = '[{"database": "GEO", "accession_number": "GSE123456", "url": null}]'
        result = extractor._parse_response(response)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["database"] == "GEO"

    def test_parse_invalid_response(self, mock_prompt_handler):
        """Test handling of invalid response format."""
        extractor = DataAvailabilityExtractorOpenAI(VALID_CONFIG, mock_prompt_handler)
        response = "This is not JSON"
        result = extractor._parse_response(response)
        assert result == []
