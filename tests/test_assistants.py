"""
Tests for the assistants module of the SODA Curation Tool.

This module contains unit tests for the functionality in assistants.py,
particularly the get_file_structure function.
"""

import pytest
from unittest.mock import patch, MagicMock
from src.assistants import get_file_structure

@pytest.fixture
def mock_anthropic():
    with patch('src.assistants.Anthropic') as mock:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = MagicMock(
            content=[
                MagicMock(text='''{
                    "manuscript": {
                        "id_": "JOURNAL-2023-12345",
                        "files": ["/path/to/manuscript.docx"],
                        "xml": "/path/to/manuscript.xml",
                        "figures": [
                            {
                                "figure_label": "Figure 1",
                                "img_file": ["/path/to/figure1.tif"],
                                "sd_file": ["/path/to/figure1-sd.zip"]
                            }
                        ]
                    }
                }''')
            ]
        )
        mock.return_value = mock_client
        yield mock

def test_get_file_structure_success(mock_anthropic):
    file_list = [
        "/path/to/manuscript.docx",
        "/path/to/manuscript.xml",
        "/path/to/figure1.tif",
        "/path/to/figure1-sd.zip"
    ]
    result = get_file_structure(file_list)
    assert isinstance(result, dict)
    assert "manuscript" in result
    assert result["manuscript"]["id_"] == "JOURNAL-2023-12345"
    assert len(result["manuscript"]["figures"]) == 1

@patch('src.assistants.os.getenv')
def test_get_file_structure_missing_api_key(mock_getenv):
    mock_getenv.return_value = None
    with pytest.raises(Exception) as exc_info:
        get_file_structure([])
    assert "ANTHROPIC_API_KEY environment variable is not set" in str(exc_info.value)

def test_get_file_structure_empty_list(mock_anthropic):
    mock_anthropic.return_value.messages.create.return_value = MagicMock(content=[MagicMock(text='{}')])
    result = get_file_structure([])
    assert isinstance(result, dict)
    assert result == {}

@patch('src.assistants.Anthropic')
def test_get_file_structure_api_error(mock_anthropic):
    mock_anthropic.return_value.messages.create.side_effect = Exception("API Error")
    result = get_file_structure(["file1.txt"])
    assert result == {}
    
@pytest.fixture
def mock_docx():
    with patch('src.assistants.docx.Document') as mock:
        mock.return_value.paragraphs = [MagicMock(text="Figure 1: This is a test caption.")]
        yield mock

@pytest.fixture
def mock_pypdf2():
    with patch('src.assistants.PyPDF2.PdfReader') as mock:
        mock.return_value.pages = [MagicMock(extract_text=MagicMock(return_value="Figure 2: This is another test caption."))]
        yield mock

def test_extract_figure_captions_docx_priority(mock_anthropic, mock_docx, mock_pypdf2):
    manuscript_files = ["/path/to/manuscript.docx", "/path/to/manuscript.pdf"]
    mock_anthropic.return_value.messages.create.return_value = MagicMock(
        content=[MagicMock(text='{"Figure 1": "This is a test caption from DOCX."}')]
    )
    
    result = extract_figure_captions(manuscript_files, "fake_api_key")
    
    assert isinstance(result, dict)
    assert len(result) == 1
    assert result["Figure 1"] == "This is a test caption from DOCX."
    mock_docx.assert_called_once()
    mock_pypdf2.assert_not_called()

def test_extract_figure_captions_pdf_fallback(mock_anthropic, mock_docx, mock_pypdf2):
    manuscript_files = ["/path/to/manuscript.docx", "/path/to/manuscript.pdf"]
    mock_anthropic.return_value.messages.create.side_effect = [
        Exception("DOCX processing failed"),
        MagicMock(content=[MagicMock(text='{"Figure 2": "This is a test caption from PDF."}')])
    ]
    
    result = extract_figure_captions(manuscript_files, "fake_api_key")
    
    assert isinstance(result, dict)
    assert len(result) == 1
    assert result["Figure 2"] == "This is a test caption from PDF."
    mock_docx.assert_called_once()
    mock_pypdf2.assert_called_once()

def test_get_file_structure_with_captions(mock_anthropic, mock_docx, mock_pypdf2):
    file_list = [
        "/path/to/manuscript.docx",
        "/path/to/manuscript.pdf",
        "/path/to/figure1.tif",
        "/path/to/figure1-sd.zip"
    ]
    
    mock_anthropic.return_value.messages.create.side_effect = [
        MagicMock(content=[MagicMock(text='''{
            "manuscript": {
                "id_": "JOURNAL-2023-12345",
                "files": ["/path/to/manuscript.docx", "/path/to/manuscript.pdf"],
                "xml": "/path/to/manuscript.xml",
                "figures": [
                    {
                        "figure_label": "Figure 1",
                        "img_file": ["/path/to/figure1.tif"],
                        "sd_file": ["/path/to/figure1-sd.zip"]
                    },
                    {
                        "figure_label": "Figure 2",
                        "img_file": ["/path/to/figure2.tif"],
                        "sd_file": ["/path/to/figure2-sd.zip"]
                    }
                ]
            }
        }''')]),
        MagicMock(content=[MagicMock(text='{"Figure 1": "This is a test caption.", "Figure 2": "This is another test caption."}')])
    ]

    result = get_file_structure(file_list)
    
    assert isinstance(result, dict)
    assert "manuscript" in result
    assert result["manuscript"]["id_"] == "JOURNAL-2023-12345"
    assert len(result["manuscript"]["figures"]) == 2
    assert "figure_caption" in result["manuscript"]["figures"][0]
    assert result["manuscript"]["figures"][0]["figure_caption"] == "This is a test caption."
    assert result["manuscript"]["figures"][1]["figure_caption"] == "This is another test caption."
