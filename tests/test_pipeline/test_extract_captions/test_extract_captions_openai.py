"""
This module contains unit tests for the OpenAI-based figure caption extraction functionality
in the soda_curation package.

It tests various scenarios of caption extraction using OpenAI's GPT models, including
successful extractions, error handling, and edge cases specific to the OpenAI implementation.
"""

import pytest
from unittest.mock import Mock, patch
import json
from soda_curation.pipeline.extract_captions.extract_captions_openai import FigureCaptionExtractorGpt
from soda_curation.pipeline.manuscript_structure.manuscript_structure import ZipStructure, Figure

@pytest.fixture
def mock_openai_client():
    """
    Fixture to mock the OpenAI client.

    This fixture patches the OpenAI class to allow controlled testing
    of OpenAI API interactions without making actual API calls.

    Yields:
        Mock: A mock object representing the OpenAI client.
    """
    with patch('soda_curation.pipeline.extract_captions.extract_captions_openai.openai.OpenAI') as mock_client:
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
    Fixture to provide a sample configuration for FigureCaptionExtractorGpt.

    Returns:
        dict: A dictionary containing sample configuration values for OpenAI API.
    """
    return {
        'api_key': 'test_key',
        'model': 'gpt-4',
        'caption_extraction_assistant_id': 'test_assistant_id'
    }

def test_gpt_extract_captions_success(mock_openai_client, sample_zip_structure, sample_config):
    """
    Test successful caption extraction using OpenAI's GPT model.

    This test verifies that when given a valid response from the OpenAI API,
    the extractor correctly processes the response and updates the ZipStructure
    with the extracted captions.

    Args:
        mock_openai_client (Mock): Mocked OpenAI client.
        sample_zip_structure (ZipStructure): Sample ZIP structure for testing.
        sample_config (dict): Sample configuration for the extractor.
    """
    mock_openai_client.return_value.beta.threads.runs.create_and_poll.return_value = Mock(status='completed')
    mock_openai_client.return_value.beta.threads.messages.list.return_value = Mock(data=[
        Mock(content=[Mock(text=Mock(value=json.dumps({
            "Figure 1": "This is caption for Figure 1",
            "Figure 2": "This is caption for Figure 2"
        })))])
    ])

    extractor = FigureCaptionExtractorGpt(sample_config)

    with patch('builtins.open', Mock()):
        with patch('os.path.exists', return_value=True):
            with patch.object(extractor, '_upload_file', return_value=Mock(id='test_file_id')):
                result = extractor.extract_captions("test.docx", sample_zip_structure)

    assert result.figures[0].figure_caption == "This is caption for Figure 1"
    assert result.figures[1].figure_caption == "This is caption for Figure 2"

def test_gpt_extract_captions_api_error(mock_openai_client, sample_zip_structure, sample_config):
    """
    Test caption extraction when an API error occurs.

    This test verifies that the extractor handles API errors gracefully,
    returning the original ZipStructure when the OpenAI API call fails.

    Args:
        mock_openai_client (Mock): Mocked OpenAI client.
        sample_zip_structure (ZipStructure): Sample ZIP structure for testing.
        sample_config (dict): Sample configuration for the extractor.
    """
    mock_openai_client.return_value.beta.threads.runs.create_and_poll.side_effect = Exception("API Error")

    extractor = FigureCaptionExtractorGpt(sample_config)
    with patch('builtins.open', Mock()):
        with patch('os.path.exists', return_value=True):
            with patch.object(extractor, '_upload_file', return_value=Mock(id='test_file_id')):
                result = extractor.extract_captions("test.docx", sample_zip_structure)

    assert result == sample_zip_structure  # Should return original structure on error

def test_gpt_extract_captions_invalid_json(mock_openai_client, sample_zip_structure, sample_config):
    """
    Test caption extraction when the API returns invalid JSON.

    This test checks if the extractor correctly handles cases where the OpenAI API
    returns a response that cannot be parsed as JSON.

    Args:
        mock_openai_client (Mock): Mocked OpenAI client.
        sample_zip_structure (ZipStructure): Sample ZIP structure for testing.
        sample_config (dict): Sample configuration for the extractor.
    """
    mock_run = Mock()
    mock_run.status = 'completed'
    mock_message = Mock()
    mock_message.content = [Mock(text=Mock(value="Invalid JSON"))]
    mock_openai_client.return_value.beta.threads.runs.create_and_poll.return_value = mock_run
    mock_openai_client.return_value.beta.threads.messages.list.return_value = Mock(data=[mock_message])

    extractor = FigureCaptionExtractorGpt(sample_config)
    with patch('builtins.open', Mock()):
        with patch('os.path.exists', return_value=True):
            with patch.object(extractor, '_upload_file', return_value=Mock(id='test_file_id')):
                result = extractor.extract_captions("test.docx", sample_zip_structure)

    assert result.figures[0].figure_caption == "Figure caption not found."
    assert result.figures[1].figure_caption == "Figure caption not found."

def test_gpt_extract_captions_missing_figures(mock_openai_client, sample_zip_structure, sample_config):
    """
    Test caption extraction when some figures are missing from the API response.

    This test verifies that the extractor correctly handles cases where the API response
    doesn't contain captions for all figures in the ZipStructure.

    Args:
        mock_openai_client (Mock): Mocked OpenAI client.
        sample_zip_structure (ZipStructure): Sample ZIP structure for testing.
        sample_config (dict): Sample configuration for the extractor.
    """
    mock_run = Mock()
    mock_run.status = 'completed'
    mock_message = Mock()
    mock_message.content = [Mock(text=Mock(value=json.dumps({
        "Figure 1": "This is caption for Figure 1"
    })))]
    mock_openai_client.return_value.beta.threads.runs.create_and_poll.return_value = mock_run
    mock_openai_client.return_value.beta.threads.messages.list.return_value = Mock(data=[mock_message])

    extractor = FigureCaptionExtractorGpt(sample_config)
    with patch('builtins.open', Mock()):
        with patch('os.path.exists', return_value=True):
            with patch.object(extractor, '_upload_file', return_value=Mock(id='test_file_id')):
                result = extractor.extract_captions("test.docx", sample_zip_structure)

    assert result.figures[0].figure_caption == "This is caption for Figure 1"
    assert result.figures[1].figure_caption == "Figure caption not found."

def test_gpt_extract_captions_pdf_fallback(mock_openai_client, sample_zip_structure, sample_config):
    """
    Test caption extraction fallback to PDF when DOCX extraction fails.

    This test verifies that the extractor attempts to extract captions from a PDF file
    when DOCX extraction fails or is not available.

    Args:
        mock_openai_client (Mock): Mocked OpenAI client.
        sample_zip_structure (ZipStructure): Sample ZIP structure for testing.
        sample_config (dict): Sample configuration for the extractor.
    """
    mock_run = Mock()
    mock_run.status = 'completed'
    mock_message = Mock()
    mock_message.content = [Mock(text=Mock(value=json.dumps({
        "Figure 1": "This is caption for Figure 1 from PDF",
        "Figure 2": "This is caption for Figure 2 from PDF"
    })))]
    mock_openai_client.return_value.beta.threads.runs.create_and_poll.return_value = mock_run
    mock_openai_client.return_value.beta.threads.messages.list.return_value = Mock(data=[mock_message])

    extractor = FigureCaptionExtractorGpt(sample_config)
    with patch('builtins.open', Mock()):
        with patch('os.path.exists', return_value=True):
            with patch.object(extractor, '_upload_file', return_value=Mock(id='test_file_id')):
                result = extractor.extract_captions("test.pdf", sample_zip_structure)

    assert result.figures[0].figure_caption == "This is caption for Figure 1 from PDF"
    assert result.figures[1].figure_caption == "This is caption for Figure 2 from PDF"
