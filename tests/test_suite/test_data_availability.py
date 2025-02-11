"""Tests for data availability extraction functionality."""

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
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "prompts": {"system": "System prompt", "user": "User prompt"},
            }
        },
        "extract_data_sources": {
            "openai": {
                "model": "gpt-4o",
                "temperature": 0.1,
                "top_p": 1.0,
                "frequency_penalty": 0,
                "presence_penalty": 0,
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
            "openai": {
                "model": "gpt-4o",
                "temperature": 3.0,
                "top_p": 2.0,
                "frequency_penalty": 3.0,
            }
        },
        "extract_data_sources": {
            "openai": {
                "model": "gpt-4o",
                "temperature": 3.0,
                "top_p": 2.0,
                "frequency_penalty": 3.0,
            }
        },
    }
}

# Mock response content
MOCK_SECTION = """```json
{
    "data_availability": "Data Availability:\\nThe data in this study are available under the following accession numbers:\\n- RNA-seq data: GEO: GSE123456\\n- Proteomics data: PRIDE: PXD987654\\n- Code repository: GitHub: https://github.com/example/code"
}
```"""

MOCK_SOURCES = """```json
[
    {
        "database": "GEO",
        "accession_number": "GSE123456",
        "url": ""
    },
    {
        "database": "PRIDE",
        "accession_number": "PXD987654",
        "url": ""
    },
    {
        "database": "GitHub",
        "accession_number": "example/code",
        "url": "https://github.com/example/code"
    }
]
```"""


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
        # Mock for locate_data_availability
        mock_resp1 = MagicMock()
        mock_resp1.choices = [MagicMock()]
        mock_resp1.choices[
            0
        ].message.content = """```json
        {
            "data_availability": "Data Availability:\\nThe data in this study are available under the following accession numbers:\\n- RNA-seq data: GEO: GSE123456\\n- Proteomics data: PRIDE: PXD987654\\n- Code repository: GitHub: https://github.com/example/code"
        }
        ```"""
        mock_resp1.usage = MagicMock(
            prompt_tokens=100, completion_tokens=50, total_tokens=150
        )

        # Mock for extract_data_sources
        mock_resp2 = MagicMock()
        mock_resp2.choices = [MagicMock()]
        mock_resp2.choices[
            0
        ].message.content = """```json
        [
            {
                "database": "GEO",
                "accession_number": "GSE123456",
                "url": null
            },
            {
                "database": "PRIDE",
                "accession_number": "PXD987654",
                "url": null
            },
            {
                "database": "GitHub",
                "accession_number": "example/code",
                "url": "https://github.com/example/code"
            }
        ]
        ```"""
        mock_resp2.usage = MagicMock(
            prompt_tokens=80, completion_tokens=40, total_tokens=120
        )

        mock_client.return_value.chat.completions.create.side_effect = [
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
        assert "GEO: GSE123456" in result.data_availability["section_text"]
        assert len(result.data_availability["data_sources"]) == 3
        assert result.data_availability["data_sources"][0]["database"] == "GEO"

    def test_no_section_found(
        self, mock_openai_client, mock_prompt_handler, zip_structure
    ):
        """Test handling when no data availability section is found."""
        # Create new mock with proper response format
        mock_empty_response = MagicMock()
        mock_empty_response.choices = [MagicMock()]

        # The response should directly match what the extractor will put in the structure
        mock_empty_response.choices[
            0
        ].message.content = (
            """<p>This study includes no data deposited in external repositories.</p>"""
        )

        # Set up token usage tracking
        from src.soda_curation.pipeline.manuscript_structure.manuscript_structure import (
            TokenUsage,
        )

        mock_empty_response.usage = TokenUsage(
            prompt_tokens=50, completion_tokens=10, total_tokens=60, cost=0.0
        )

        # Override the mock's return value
        mock_openai_client.return_value.chat.completions.create.reset_mock()
        mock_openai_client.return_value.chat.completions.create.side_effect = None
        mock_openai_client.return_value.chat.completions.create.return_value = (
            mock_empty_response
        )

        extractor = DataAvailabilityExtractorOpenAI(VALID_CONFIG, mock_prompt_handler)
        result = extractor.extract_data_availability("test content", zip_structure)

        assert isinstance(result, ZipStructure)

        # The data_availability structure should match the expected format
        assert "data_availability" in result.__dict__
        assert result.data_availability == {
            "section_text": "<p>This study includes no data deposited in external repositories.</p>",
            "data_sources": [],
        }

    def test_api_error_handling(
        self, mock_openai_client, mock_prompt_handler, zip_structure
    ):
        """Test handling of API errors."""
        mock_openai_client.return_value.chat.completions.create.side_effect = Exception(
            "API Error"
        )

        extractor = DataAvailabilityExtractorOpenAI(VALID_CONFIG, mock_prompt_handler)
        result = extractor.extract_data_availability("test content", zip_structure)

        assert isinstance(result, ZipStructure)
        # Fix the data structure access
        assert result.data_availability == {"section_text": "", "data_sources": []}
