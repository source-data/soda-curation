"""
This module contains unit tests for the Anthropic-based figure caption extraction functionality
in the soda_curation package.

It tests various scenarios of caption extraction using the Anthropic AI model, including
successful extractions, error handling, and edge cases.
"""

import pytest
from unittest.mock import Mock, patch, mock_open
import json
from soda_curation.pipeline.extract_captions.extract_captions_anthropic import FigureCaptionExtractorClaude
from soda_curation.pipeline.zip_structure.zip_structure_base import ZipStructure, Figure

@pytest.fixture
def mock_anthropic_client():
    """
    Fixture to mock the Anthropic client.

    This fixture patches the Anthropic class to allow controlled testing
    of Anthropic API interactions without making actual API calls.

    Yields:
        Mock: A mock object representing the Anthropic client.
    """
    with patch('soda_curation.pipeline.extract_captions.extract_captions_anthropic.Anthropic') as mock_client:
        yield mock_client

@pytest.fixture
def sample_zip_structure():
    """
    Fixture to provide a sample ZipStructure for testing.

    Returns:
        ZipStructure: A sample ZipStructure object with predefined attributes and figures.
    """
    return ZipStructure(
        manuscript_id="test_manuscript",
        xml="test.xml",
        docx="test.docx",
        pdf="test.pdf",
        appendix=[],
        figures=[
            Figure("Figure 1", ["image1.png"], ["data1.xlsx"], "TO BE ADDED IN LATER STEP", []),
            Figure("Figure 2", ["image2.png"], ["data2.xlsx"], "TO BE ADDED IN LATER STEP", [])
        ]
    )

@pytest.fixture
def sample_config():
    """
    Fixture to provide a sample configuration for FigureCaptionExtractorClaude.

    Returns:
        dict: A dictionary containing sample configuration values for Anthropic API.
    """
    return {
        'api_key': 'test_key',
        'model': 'claude-3-sonnet-20240229',
        'max_tokens_to_sample': 1000,
        'temperature': 0.7
    }

@pytest.fixture
def mock_file_operations():
    """
    Fixture to mock file operations.

    This fixture patches various file-related operations to simulate file handling
    without actual file system interactions.
    """
    with patch('builtins.open', mock_open(read_data="test content")), \
         patch('os.path.exists', return_value=True), \
         patch('shutil.copy2'), \
         patch('soda_curation.pipeline.extract_captions.extract_captions_anthropic.Document') as mock_document:
        mock_document.return_value.paragraphs = [Mock(text="Figure 1. Test caption")]
        yield

def test_claude_extract_captions_success(mock_anthropic_client, sample_zip_structure, sample_config, mock_file_operations):
    """
    Test successful caption extraction using the Anthropic Claude model.

    This test verifies that when given a valid response from the Anthropic API,
    the extractor correctly processes the response and updates the ZipStructure
    with the extracted captions.

    Args:
        mock_anthropic_client (Mock): Mocked Anthropic client.
        sample_zip_structure (ZipStructure): Sample ZIP structure for testing.
        sample_config (dict): Sample configuration for the extractor.
        mock_file_operations (fixture): Mocked file operations.
    """
    mock_response = Mock()
    mock_response.content = json.dumps({
        "Figure 1": "This is caption for Figure 1",
        "Figure 2": "This is caption for Figure 2"
    })
    mock_anthropic_client.return_value.messages.create.return_value = mock_response

    extractor = FigureCaptionExtractorClaude(sample_config)
    result = extractor.extract_captions("test.docx", sample_zip_structure)

    assert result.figures[0].figure_caption == "This is caption for Figure 1"
    assert result.figures[1].figure_caption == "This is caption for Figure 2"

def test_claude_extract_captions_api_error(mock_anthropic_client, sample_zip_structure, sample_config, mock_file_operations):
    """
    Test caption extraction when an API error occurs.

    This test verifies that the extractor handles API errors gracefully,
    setting default captions when the Anthropic API call fails.

    Args:
        mock_anthropic_client (Mock): Mocked Anthropic client.
        sample_zip_structure (ZipStructure): Sample ZIP structure for testing.
        sample_config (dict): Sample configuration for the extractor.
        mock_file_operations (fixture): Mocked file operations.
    """
    mock_anthropic_client.return_value.messages.create.side_effect = Exception("API Error")

    extractor = FigureCaptionExtractorClaude(sample_config)
    result = extractor.extract_captions("test.docx", sample_zip_structure)

    assert result.figures[0].figure_caption == "Figure caption not found."
    assert result.figures[1].figure_caption == "Figure caption not found."

def test_claude_extract_captions_invalid_json(mock_anthropic_client, sample_zip_structure, sample_config, mock_file_operations):
    """
    Test caption extraction when the API returns invalid JSON.

    This test checks if the extractor correctly handles cases where the Anthropic API
    returns a response that cannot be parsed as JSON.

    Args:
        mock_anthropic_client (Mock): Mocked Anthropic client.
        sample_zip_structure (ZipStructure): Sample ZIP structure for testing.
        sample_config (dict): Sample configuration for the extractor.
        mock_file_operations (fixture): Mocked file operations.
    """
    mock_response = Mock()
    mock_response.content = "Invalid JSON"
    mock_anthropic_client.return_value.messages.create.return_value = mock_response

    extractor = FigureCaptionExtractorClaude(sample_config)
    result = extractor.extract_captions("test.docx", sample_zip_structure)

    assert result.figures[0].figure_caption == "Figure caption not found."
    assert result.figures[1].figure_caption == "Figure caption not found."

def test_claude_extract_captions_missing_figures(mock_anthropic_client, sample_zip_structure, sample_config, mock_file_operations):
    """
    Test caption extraction when some figures are missing from the API response.

    This test verifies that the extractor correctly handles cases where the API response
    doesn't contain captions for all figures in the ZipStructure.

    Args:
        mock_anthropic_client (Mock): Mocked Anthropic client.
        sample_zip_structure (ZipStructure): Sample ZIP structure for testing.
        sample_config (dict): Sample configuration for the extractor.
        mock_file_operations (fixture): Mocked file operations.
    """
    mock_response = Mock()
    mock_response.content = json.dumps({
        "Figure 1": "This is caption for Figure 1"
    })
    mock_anthropic_client.return_value.messages.create.return_value = mock_response

    extractor = FigureCaptionExtractorClaude(sample_config)
    result = extractor.extract_captions("test.docx", sample_zip_structure)

    assert result.figures[0].figure_caption == "This is caption for Figure 1"
    assert result.figures[1].figure_caption == "Figure caption not found."

def test_claude_extract_captions_pdf_fallback(mock_anthropic_client, sample_zip_structure, sample_config, mock_file_operations):
    """
    Test caption extraction fallback to PDF when DOCX extraction fails.

    This test verifies that the extractor attempts to extract captions from a PDF file
    when DOCX extraction fails or is not available.

    Args:
        mock_anthropic_client (Mock): Mocked Anthropic client.
        sample_zip_structure (ZipStructure): Sample ZIP structure for testing.
        sample_config (dict): Sample configuration for the extractor.
        mock_file_operations (fixture): Mocked file operations.
    """
    mock_response = Mock()
    mock_response.content = json.dumps({
        "Figure 1": "This is caption for Figure 1 from PDF",
        "Figure 2": "This is caption for Figure 2 from PDF"
    })
    mock_anthropic_client.return_value.messages.create.return_value = mock_response

    extractor = FigureCaptionExtractorClaude(sample_config)
    result = extractor.extract_captions("test.pdf", sample_zip_structure)

    assert result.figures[0].figure_caption == "This is caption for Figure 1 from PDF"
    assert result.figures[1].figure_caption == "This is caption for Figure 2 from PDF"

def test_claude_extract_captions_empty_response(mock_anthropic_client, sample_zip_structure, sample_config, mock_file_operations):
    """
    Test caption extraction when the API returns an empty response.

    This test checks if the extractor correctly handles cases where the Anthropic API
    returns an empty response.

    Args:
        mock_anthropic_client (Mock): Mocked Anthropic client.
        sample_zip_structure (ZipStructure): Sample ZIP structure for testing.
        sample_config (dict): Sample configuration for the extractor.
        mock_file_operations (fixture): Mocked file operations.
    """
    mock_response = Mock()
    mock_response.content = ""
    mock_anthropic_client.return_value.messages.create.return_value = mock_response

    extractor = FigureCaptionExtractorClaude(sample_config)
    result = extractor.extract_captions("test.docx", sample_zip_structure)

    assert result.figures[0].figure_caption == "Figure caption not found."
    assert result.figures[1].figure_caption == "Figure caption not found."

def test_claude_extract_captions_malformed_json(mock_anthropic_client, sample_zip_structure, sample_config, mock_file_operations):
    """
    Test caption extraction when the API returns malformed JSON.

    This test verifies that the extractor gracefully handles cases where the Anthropic API
    returns a response with malformed JSON.

    Args:
        mock_anthropic_client (Mock): Mocked Anthropic client.
        sample_zip_structure (ZipStructure): Sample ZIP structure for testing.
        sample_config (dict): Sample configuration for the extractor.
        mock_file_operations (fixture): Mocked file operations.
    """
    mock_response = Mock()
    mock_response.content = '{"Figure 1": "Caption 1", "Figure 2": }'
    mock_anthropic_client.return_value.messages.create.return_value = mock_response

    extractor = FigureCaptionExtractorClaude(sample_config)
    result = extractor.extract_captions("test.docx", sample_zip_structure)

    assert result.figures[0].figure_caption == "Figure caption not found."
    assert result.figures[1].figure_caption == "Figure caption not found."

def test_claude_extract_captions_file_not_found(mock_anthropic_client, sample_zip_structure, sample_config):
    """
    Test caption extraction when the input file is not found.

    This test checks if the extractor correctly handles cases where the input file
    (DOCX or PDF) is not found or cannot be opened.

    Args:
        mock_anthropic_client (Mock): Mocked Anthropic client.
        sample_zip_structure (ZipStructure): Sample ZIP structure for testing.
        sample_config (dict): Sample configuration for the extractor.
    """
    with patch('soda_curation.pipeline.extract_captions.extract_captions_anthropic.Document', side_effect=FileNotFoundError("File not found")):
        extractor = FigureCaptionExtractorClaude(sample_config)
        result = extractor.extract_captions("nonexistent.docx", sample_zip_structure)

    assert result.figures[0].figure_caption == "Figure caption not found."
    assert result.figures[1].figure_caption == "Figure caption not found."

def test_claude_extract_captions_unexpected_error(mock_anthropic_client, sample_zip_structure, sample_config, mock_file_operations):
    """
    Test caption extraction when an unexpected error occurs.

    This test verifies that the extractor gracefully handles unexpected errors
    during the caption extraction process.

    Args:
        mock_anthropic_client (Mock): Mocked Anthropic client.
        sample_zip_structure (ZipStructure): Sample ZIP structure for testing.
        sample_config (dict): Sample configuration for the extractor.
        mock_file_operations (fixture): Mocked file operations.
    """
    mock_anthropic_client.return_value.messages.create.side_effect = ValueError("Unexpected error")

    extractor = FigureCaptionExtractorClaude(sample_config)
    result = extractor.extract_captions("test.docx", sample_zip_structure)

    assert result.figures[0].figure_caption == "Figure caption not found."
    assert result.figures[1].figure_caption == "Figure caption not found."
