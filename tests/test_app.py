"""
Tests for the SODA Curation Tool Streamlit app.

This module contains unit tests for the main functionality of the SODA Curation Tool.
"""

import pytest
from streamlit.testing.v1 import AppTest
from unittest.mock import patch, MagicMock
import io
import zipfile
import tempfile
import os

def create_mock_file(with_ejp=True):
    """Create a mock file object for testing purposes."""
    mock_file = MagicMock()
    mock_file.name = "test.zip"
    mock_file.type = "application/zip"
    
    # Create a real ZIP file in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        if with_ejp:
            zf.writestr('some_folder/eJP/file1.txt', 'content1')
            zf.writestr('some_folder/eJP/file2.txt', 'content2')
        else:
            zf.writestr('some_folder/file1.txt', 'content1')
    
    mock_file.getvalue.return_value = zip_buffer.getvalue()
    mock_file.size = len(zip_buffer.getvalue())
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

def test_ejp_folder_found():
    """Test if the eJP folder is found and its contents are displayed."""
    with patch('streamlit.file_uploader', return_value=create_mock_file()):
        at = AppTest.from_file("src/app.py")
        at.run()
        assert any("Found 2 files in the eJP folder" in s.value for s in at.success)
        assert any("file1.txt" in t.value for t in at.text)
        assert any("file2.txt" in t.value for t in at.text)

def test_ejp_folder_not_found():
    """Test the behavior when no eJP folder is found in the ZIP file."""
    with patch('streamlit.file_uploader', return_value=create_mock_file(with_ejp=False)):
        at = AppTest.from_file("src/app.py")
        at.run()
        assert any("No eJP folder found in the ZIP file" in w.value for w in at.warning)