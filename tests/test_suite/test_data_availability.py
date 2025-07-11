"""Tests for data source extraction functionality."""

import json
from unittest.mock import MagicMock, mock_open, patch

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
}

# Mock database registry content
MOCK_REGISTRY_CONTENT = """{
  "databases": [
    {
      "name": "Gene Expression Omnibus",
      "identifiers_pattern": "https://identifiers.org/geo:",
      "url_pattern": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={$id}",
      "sample_id": "GDS1234",
      "sample_identifiers_url": "https://identifiers.org/geo:GDS1234"
    },
    {
      "name": "Proteomics Identification database",
      "identifiers_pattern": "https://identifiers.org/pride.project:",
      "url_pattern": "https://www.ebi.ac.uk/pride/archive/projects/{$id}",
      "sample_id": "PXD000440",
      "sample_identifiers_url": "https://identifiers.org/pride.project:PXD000440"
    },
    {
      "name": "Github",
      "url_pattern": "https://github.com/",
      "identifiers_pattern": "",
      "sample_id": "source-data/sdash/",
      "sample_identifiers_url": ""
    }
  ]
}"""


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

    @patch("builtins.open", mock_open(read_data=MOCK_REGISTRY_CONTENT))
    def test_valid_config(self, mock_prompt_handler):
        """Test that valid configuration is accepted."""
        extractor = DataAvailabilityExtractorOpenAI(VALID_CONFIG, mock_prompt_handler)
        assert extractor.config == VALID_CONFIG

    @patch("builtins.open", mock_open(read_data=MOCK_REGISTRY_CONTENT))
    def test_invalid_model(self, mock_prompt_handler):
        """Test that invalid model raises ValueError."""
        with pytest.raises(ValueError, match="Invalid model"):
            DataAvailabilityExtractorOpenAI(INVALID_MODEL_CONFIG, mock_prompt_handler)

    @patch("builtins.open", mock_open(read_data=MOCK_REGISTRY_CONTENT))
    def test_invalid_parameters(self, mock_prompt_handler):
        """Test that invalid parameters raise ValueError."""
        with pytest.raises(ValueError):
            DataAvailabilityExtractorOpenAI(INVALID_PARAMS_CONFIG, mock_prompt_handler)


class TestDatabaseRegistry:
    """Test loading and formatting of database registry."""

    @patch("builtins.open", mock_open(read_data=MOCK_REGISTRY_CONTENT))
    def test_registry_loading(self, mock_prompt_handler):
        """Test that the database registry is loaded correctly."""
        extractor = DataAvailabilityExtractorOpenAI(VALID_CONFIG, mock_prompt_handler)
        assert "databases" in extractor.database_registry
        assert len(extractor.database_registry["databases"]) == 3

    @patch("builtins.open", mock_open(read_data=MOCK_REGISTRY_CONTENT))
    def test_registry_formatting(self, mock_prompt_handler):
        """Test that the registry is formatted correctly as a markdown table."""
        extractor = DataAvailabilityExtractorOpenAI(VALID_CONFIG, mock_prompt_handler)
        formatted = extractor._create_registry_info()

        # Check that it's a JSON string with the expected keys
        import json

        registry = json.loads(formatted)
        assert "databases" in registry
        assert any(
            db["name"] == "Gene Expression Omnibus" for db in registry["databases"]
        )
        assert any(
            db["name"] == "Proteomics Identification database"
            for db in registry["databases"]
        )
        assert any(db["name"] == "Github" for db in registry["databases"])
        assert "https://identifiers.org/geo:" in formatted
        assert "https://identifiers.org/pride.project:" in formatted

    @patch("builtins.open", side_effect=Exception("File not found"))
    def test_registry_loading_error(self, mock_file, mock_prompt_handler):
        """Test error handling when loading the registry fails."""
        extractor = DataAvailabilityExtractorOpenAI(VALID_CONFIG, mock_prompt_handler)
        assert "databases" in extractor.database_registry
        assert extractor.database_registry["databases"] == []


class TestDataSourceExtraction:
    """Test data source extraction functionality."""

    @patch("builtins.open", mock_open(read_data=MOCK_REGISTRY_CONTENT))
    def test_successful_extraction(
        self, mock_openai_client, mock_prompt_handler, zip_structure
    ):
        """Test successful extraction of data sources."""
        extractor = DataAvailabilityExtractorOpenAI(VALID_CONFIG, mock_prompt_handler)
        result = extractor.extract_data_sources("test content", zip_structure)

        assert isinstance(result, ZipStructure)
        assert "section_text" in result.data_availability
        assert "data_sources" in result.data_availability

        # Check that the data sources are extracted correctly
        assert len(result.data_availability["data_sources"]) == len(
            MOCK_SOURCES_RESPONSE["sources"]
        )

        # Verify the extracted sources match the mock response
        for i, source in enumerate(result.data_availability["data_sources"]):
            assert source["database"] == MOCK_SOURCES_RESPONSE["sources"][i]["database"]
            assert (
                source["accession_number"]
                == MOCK_SOURCES_RESPONSE["sources"][i]["accession_number"]
            )
            assert source["url"] == MOCK_SOURCES_RESPONSE["sources"][i]["url"]

    @patch("builtins.open", mock_open(read_data=MOCK_REGISTRY_CONTENT))
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

    @patch("builtins.open", mock_open(read_data=MOCK_REGISTRY_CONTENT))
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

    @patch("builtins.open", mock_open(read_data=MOCK_REGISTRY_CONTENT))
    def test_system_prompt_with_registry(self, mock_prompt_handler):
        """Test that the system prompt includes the registry information."""
        extractor = DataAvailabilityExtractorOpenAI(VALID_CONFIG, mock_prompt_handler)

        # Mock the prompt handler and _create_registry_info
        registry_info = "Sample registry info"
        extractor._create_registry_info = MagicMock(return_value=registry_info)

        with patch.object(extractor, "client") as mock_client:
            # Set up the mock client
            mock_response = MagicMock()
            mock_response.choices = [
                MagicMock(message=MagicMock(content=json.dumps({"sources": []})))
            ]
            mock_response.usage = MagicMock(
                prompt_tokens=0, completion_tokens=0, total_tokens=0
            )
            mock_client.beta.chat.completions.parse.return_value = mock_response

            # Call the method
            extractor.extract_data_sources("test", ZipStructure())

            # Check that the system prompt includes the registry info
            called_args = mock_client.beta.chat.completions.parse.call_args
            messages = called_args[1]["messages"]
            system_message = messages[0]["content"]

            assert registry_info in system_message


class TestResponseParsing:
    """Test parsing of AI responses."""

    @patch("builtins.open", mock_open(read_data=MOCK_REGISTRY_CONTENT))
    def test_parse_valid_response(self, mock_prompt_handler):
        """Test parsing of valid response format."""
        extractor = DataAvailabilityExtractorOpenAI(VALID_CONFIG, mock_prompt_handler)
        result = extractor._parse_response(json.dumps(MOCK_SOURCES_RESPONSE))

        assert isinstance(result, list)
        assert len(result) == len(MOCK_SOURCES_RESPONSE["sources"])
        for source in result:
            assert all(key in source for key in ["database", "accession_number", "url"])

    @patch("builtins.open", mock_open(read_data=MOCK_REGISTRY_CONTENT))
    def test_parse_invalid_response(self, mock_prompt_handler):
        """Test parsing of invalid response."""
        extractor = DataAvailabilityExtractorOpenAI(VALID_CONFIG, mock_prompt_handler)
        result = extractor._parse_response("Invalid JSON content")

        assert result == []
