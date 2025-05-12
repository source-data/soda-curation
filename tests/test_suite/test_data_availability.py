"""Tests for data source extraction functionality."""

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
        "extract_data_sources": {
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
        "extract_data_sources": {"openai": {"model": "invalid-model"}},
    }
}

INVALID_PARAMS_CONFIG = {
    "pipeline": {
        "extract_data_sources": {
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
MOCK_SOURCES_RESPONSE = {
    "sources": [
        {"database": "GEO", "accession_number": "GSE123456", "url": ""},
        {"database": "PRIDE", "accession_number": "PXD987654", "url": ""},
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
        instance = mock_client.return_value
        instance.beta.chat.completions.parse.return_value = MagicMock(
            choices=[
                MagicMock(message=MagicMock(content=json.dumps(MOCK_SOURCES_RESPONSE)))
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
        data_availability={
            "section_text": "",
            "data_sources": [],
        },
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


class TestDataSourceExtraction:
    """Test data source extraction functionality."""

    def test_successful_extraction(
        self, mock_openai_client, mock_prompt_handler, zip_structure
    ):
        """Test successful extraction of data sources."""
        extractor = DataAvailabilityExtractorOpenAI(VALID_CONFIG, mock_prompt_handler)
        result = extractor.extract_data_sources("test content", zip_structure)

        assert isinstance(result, ZipStructure)
        assert "section_text" in result.data_availability
        assert "data_sources" in result.data_availability
        assert len(result.data_availability["data_sources"]) == len(
            MOCK_SOURCES_RESPONSE["sources"]
        )

        # Expected normalized values
        expected_normalized = [
            {
                "database": "Gene Expression Omnibus",
                "accession_number": "GSE123456",
                "url": "https://identifiers.org/geo:GSE123456",
            },
            {
                "database": "Proteomics Identification database",
                "accession_number": "PXD987654",
                "url": "https://identifiers.org/pride.project:PXD987654",
            },
            {
                "database": "Github",
                "accession_number": "example/code",
                "url": "https://github.com/example/code",
            },
        ]

        # Compare with normalized values
        for actual, expected in zip(
            result.data_availability["data_sources"], expected_normalized
        ):
            assert actual["database"] == expected["database"]
            assert actual["accession_number"] == expected["accession_number"]
            assert actual["url"] == expected["url"]

    def test_no_sources_found(
        self, mock_openai_client, mock_prompt_handler, zip_structure
    ):
        """Test handling when no data sources are found."""
        mock_empty_response = json.dumps({"sources": []})
        mock_openai_client.return_value.beta.chat.completions.parse.return_value = (
            MagicMock(
                choices=[MagicMock(message=MagicMock(content=mock_empty_response))],
                usage=MagicMock(
                    prompt_tokens=50,
                    completion_tokens=10,
                    total_tokens=60,
                ),
            )
        )

        extractor = DataAvailabilityExtractorOpenAI(VALID_CONFIG, mock_prompt_handler)
        result = extractor.extract_data_sources("test content", zip_structure)

        assert isinstance(result, ZipStructure)
        assert result.data_availability["data_sources"] == []

    def test_api_error_handling(
        self, mock_openai_client, mock_prompt_handler, zip_structure
    ):
        """Test handling of API errors."""
        mock_openai_client.return_value.beta.chat.completions.parse.side_effect = (
            Exception("API Error")
        )

        extractor = DataAvailabilityExtractorOpenAI(VALID_CONFIG, mock_prompt_handler)
        result = extractor.extract_data_sources("test content", zip_structure)

        assert isinstance(result, ZipStructure)
        assert result.data_availability["data_sources"] == []


class TestResponseParsing:
    """Test parsing of AI responses."""

    def test_parse_valid_response(self, mock_prompt_handler):
        """Test parsing of valid response format."""
        extractor = DataAvailabilityExtractorOpenAI(VALID_CONFIG, mock_prompt_handler)
        result = extractor._parse_response(json.dumps(MOCK_SOURCES_RESPONSE))

        assert isinstance(result, list)
        assert len(result) == len(MOCK_SOURCES_RESPONSE["sources"])
        for source in result:
            assert all(key in source for key in ["database", "accession_number", "url"])

    def test_parse_invalid_response(self, mock_prompt_handler):
        """Test parsing of invalid response."""
        extractor = DataAvailabilityExtractorOpenAI(VALID_CONFIG, mock_prompt_handler)
        result = extractor._parse_response("Invalid JSON content")

        assert result == []


class TestDatabaseNormalization:
    """
    Test database name normalization and URL construction functionality.

    These tests verify that:
    1. Database names are properly normalized to standard forms
    2. Permanent URLs are constructed correctly using identifiers.org
    3. The normalization process is integrated with the extraction pipeline
    """

    def test_database_name_normalization(self, mock_prompt_handler):
        """Test normalization of database names."""
        extractor = DataAvailabilityExtractorOpenAI(VALID_CONFIG, mock_prompt_handler)

        # Test standard normalization cases
        test_cases = [
            ("geo", "Gene Expression Omnibus"),
            ("GEO", "Gene Expression Omnibus"),
            ("Gene Expression Omnibus", "Gene Expression Omnibus"),
            ("array express", "ArrayExpress"),
            ("arrayexpress", "ArrayExpress"),
            ("bioproject", "BioProject"),
            ("pride", "Proteomics Identification database"),
            ("PRIDE", "Proteomics Identification database"),
            ("github", "Github"),
            ("GitHUB", "Github"),
            # Test an unknown database name
            ("Unknown Database", "Unknown Database"),
            # Test edge cases
            ("", ""),
            ("  geo  ", "Gene Expression Omnibus"),
        ]

        for input_name, expected in test_cases:
            result = extractor.normalize_database_name(input_name)
            assert (
                result == expected
            ), f"Failed to normalize {input_name} to {expected}, got {result}"

    def test_permanent_url_construction(self, mock_prompt_handler):
        """Test construction of permanent URLs."""
        extractor = DataAvailabilityExtractorOpenAI(VALID_CONFIG, mock_prompt_handler)

        # Test URL construction cases
        test_cases = [
            # Standard identifiers.org databases
            (
                "Gene Expression Omnibus",
                "GSE123456",
                "",
                "https://identifiers.org/geo:GSE123456",
            ),
            (
                "Proteomics Identification database",
                "PXD000440",
                "",
                "https://identifiers.org/pride.project:PXD000440",
            ),
            ("BioProject", "PRJDB3", "", "https://identifiers.org/bioproject:PRJDB3"),
            # DOI cases
            ("Dryad", "10.5061/dryad.3rq87", "", "https://doi.org/10.5061/dryad.3rq87"),
            (
                "FigShare",
                "10.6084/m9.figshare.21780449.v1",
                "",
                "https://doi.org/10.6084/m9.figshare.21780449.v1",
            ),
            # GitHub repository
            ("Github", "source-data/sdash", "", "https://github.com/source-data/sdash"),
            # Test with existing URL
            (
                "Unknown Database",
                "ABC123",
                "https://example.com/ABC123",
                "https://example.com/ABC123",
            ),
            # Test edge cases
            ("", "123456", "", ""),
            ("Database", "", "", ""),
            ("Database", "", "https://original.url", "https://original.url"),
        ]

        for db, accession, original_url, expected in test_cases:
            result = extractor.construct_permanent_url(db, accession, original_url)
            assert (
                result == expected
            ), f"Failed to construct URL for {db}:{accession}, got {result}, expected {expected}"

    def test_data_source_normalization(self, mock_prompt_handler):
        """Test normalization of entire data sources list."""
        extractor = DataAvailabilityExtractorOpenAI(VALID_CONFIG, mock_prompt_handler)

        input_sources = [
            {"database": "geo", "accession_number": "GSE123456", "url": ""},
            {"database": "PRIDE", "accession_number": "PXD987654", "url": ""},
            {
                "database": "GitHub",
                "accession_number": "example/code",
                "url": "https://github.com/example/code",
            },
            # Include a source with incomplete data
            {"database": "incomplete", "url": ""},
            {"accession_number": "123456", "url": ""},
            # Source with all fields but unknown database
            {
                "database": "Unknown Database",
                "accession_number": "XYZ789",
                "url": "https://unknown-db.org/XYZ789",
            },
        ]

        result = extractor.normalize_data_sources(input_sources)

        # Check the expected number of sources
        assert len(result) == len(input_sources)

        # Verify the first source (geo)
        assert result[0]["database"] == "Gene Expression Omnibus"
        assert result[0]["accession_number"] == "GSE123456"
        assert result[0]["url"] == "https://identifiers.org/geo:GSE123456"

        # Verify the second source (PRIDE)
        assert result[1]["database"] == "Proteomics Identification database"
        assert result[1]["accession_number"] == "PXD987654"
        assert result[1]["url"] == "https://identifiers.org/pride.project:PXD987654"

        # Verify the third source (GitHub)
        assert result[2]["database"] == "Github"
        assert result[2]["accession_number"] == "example/code"
        assert result[2]["url"] == "https://github.com/example/code"

        # Verify the incomplete sources were preserved
        assert "database" in result[3]
        assert "accession_number" not in result[3]

        assert "database" not in result[4]
        assert "accession_number" in result[4]

        # Verify the unknown database case
        assert result[5]["database"] == "Unknown Database"
        assert result[5]["url"] == "https://unknown-db.org/XYZ789"

    def test_extraction_with_normalization(
        self, mock_openai_client, mock_prompt_handler, zip_structure
    ):
        """Test data source extraction with normalization."""
        # Create a custom mock response
        custom_sources = {
            "sources": [
                {"database": "geo", "accession_number": "GSE123456", "url": ""},
                {"database": "PRIDE", "accession_number": "PXD987654", "url": ""},
            ]
        }

        mock_openai_client.return_value.beta.chat.completions.parse.return_value = (
            MagicMock(
                choices=[
                    MagicMock(message=MagicMock(content=json.dumps(custom_sources)))
                ],
                usage=MagicMock(
                    prompt_tokens=100,
                    completion_tokens=50,
                    total_tokens=150,
                ),
            )
        )

        extractor = DataAvailabilityExtractorOpenAI(VALID_CONFIG, mock_prompt_handler)

        # Patch the normalize_data_sources method to verify it gets called
        original_method = extractor.normalize_data_sources
        try:
            # Create a mock to track calls
            normalized_data = [
                {
                    "database": "Gene Expression Omnibus",
                    "accession_number": "GSE123456",
                    "url": "https://identifiers.org/geo:GSE123456",
                },
                {
                    "database": "Proteomics Identification database",
                    "accession_number": "PXD987654",
                    "url": "https://identifiers.org/pride.project:PXD987654",
                },
            ]
            extractor.normalize_data_sources = MagicMock(return_value=normalized_data)

            # Call extract_data_sources which should call normalize_data_sources
            result = extractor.extract_data_sources("test content", zip_structure)

            # Verify normalize_data_sources was called
            extractor.normalize_data_sources.assert_called_once()

            # Verify the normalized data was used
            assert result.data_availability["data_sources"] == normalized_data

        finally:
            # Restore the original method
            extractor.normalize_data_sources = original_method
