"""
Tests for the SODA Curation Tool Streamlit app.

This module contains unit tests for the main functionality of the SODA Curation Tool.
"""

import pytest
from streamlit.testing.v1 import AppTest
from unittest.mock import patch, MagicMock
import io

def create_mock_file():
    """Create a mock file object for testing purposes."""
    mock_file = MagicMock()
    mock_file.name = "test.zip"
    mock_file.type = "application/zip"
    mock_file.size = 1024
    mock_file.getvalue.return_value = b"test content"
    return mock_file

def test_app_title():
    """Test if the app title is set to 'SODA Curation Tool'."""
    at = AppTest.from_file("src/app.py")
    at.run()
    assert "SODA Curation Tool" in at.title[0].value

def test_upload_section_present():
    """Test if the upload section is present in the app."""
    at = AppTest.from_file("src/app.py")
    at.run()
    assert any("Upload ZIP File" in h.value for h in at.header)

def test_file_upload_feedback():
    """Test if the app provides feedback when a file is uploaded."""
    with patch('streamlit.file_uploader', return_value=create_mock_file()):
        at = AppTest.from_file("src/app.py")
        at.run()
        assert any("test.zip" in s.value for s in at.success)

def test_file_details_displayed():
    """Test if the file details are displayed after a file is uploaded."""
    with patch('streamlit.file_uploader', return_value=create_mock_file()):
        at = AppTest.from_file("src/app.py")
        at.run()
        all_text = ' '.join([t.value for t in at.text if hasattr(t, 'value')])
        assert "FileName" in all_text
        assert "FileType" in all_text
        assert "FileSize" in all_text

def test_file_processing_info():
    """Test if the information of the processed file is displayed."""
    with patch('streamlit.file_uploader', return_value=create_mock_file()):
        at = AppTest.from_file("src/app.py")
        at.run()
        assert any("processed" in i.value.lower() for i in at.info)

def test_temporary_file_info():
    """Test if the temporary file path is displayed."""
    with patch('streamlit.file_uploader', return_value=create_mock_file()):
        at = AppTest.from_file("src/app.py")
        at.run()
        assert any("temporary file path" in i.value.lower() for i in at.info)