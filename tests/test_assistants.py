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