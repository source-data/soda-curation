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
import sys
import json

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

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

@pytest.fixture
def mock_process_zip_file():
    with patch('app.process_zip_file') as mock:
        mock.return_value = (['file1.txt', 'file2.txt'], '/tmp/ejp_files')
        yield mock

def test_app_title():
    """Test if the app title is set to 'SODA Curation Tool'."""
    at = AppTest.from_file("src/app.py")
    at.run()
    assert any("SODA Curation Tool" in t.value for t in at.title)

def test_upload_section_present():
    """Test if the upload section is present in the app."""
    at = AppTest.from_file("src/app.py")
    at.run()
    assert any("Upload ZIP File" in h.value for h in at.header)

def test_file_upload_feedback(mock_process_zip_file):
    """Test if the app provides feedback when a file is uploaded."""
    with patch('streamlit.file_uploader', return_value=create_mock_file()):
        at = AppTest.from_file("src/app.py")
        at.run()
        assert any("test.zip" in s.value for s in at.success)

def test_file_details_displayed(mock_process_zip_file):
    """Test if the file details are displayed after a file is uploaded."""
    with patch('streamlit.file_uploader', return_value=create_mock_file()):
        at = AppTest.from_file("src/app.py")
        at.run(timeout=10)  # Increase timeout to 10 seconds
        all_text = ' '.join([t.value for t in at.text if hasattr(t, 'value')])
        assert "FileName" in all_text
        assert "FileType" in all_text
        assert "FileSize" in all_text

@pytest.fixture
def mock_get_file_structure():
    with patch('src.assistants.get_file_structure') as mock:
        mock.return_value = {
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
        }
        yield mock

def test_file_processing_info(mock_get_file_structure):
    """Test if the information of the processed file is displayed."""
    with patch('streamlit.file_uploader', return_value=create_mock_file()):
        at = AppTest.from_file("src/app.py")
        at.run(timeout=10)  # Increase timeout to 10 seconds
        
        # Print all success messages for debugging
        print("Success messages:")
        for s in at.success:
            print(s.value)
        
        # Print all text outputs for debugging
        print("Text outputs:")
        for t in at.text:
            print(t.value)
        
        # Check if the success message is displayed
        assert any("File structure processed successfully" in s.value for s in at.success), "Success message not found"
        
        # Check if the JSON output is displayed
        json_output = [t for t in at.text if "JOURNAL-2023-12345" in t.value]
        if not json_output:
            # If JSON output is not found in text, check in other elements
            json_output = [e for e in at._elements if hasattr(e, 'value') and "JOURNAL-2023-12345" in str(e.value)]
        
        assert len(json_output) > 0, f"JSON output not found. All elements: {at._elements}"
        
        # Verify the content of the JSON output
        json_str = json_output[0].value
        # Remove the debug prefix if it exists
        if json_str.startswith("Debug: Raw JSON output: "):
            json_str = json_str.replace("Debug: Raw JSON output: ", "", 1)
        json_content = json.loads(json_str)
        assert json_content["manuscript"]["id_"] == "JOURNAL-2023-12345", "Incorrect manuscript ID in JSON output"
        
def test_ejp_folder_found(mock_process_zip_file):
    """Test if the eJP folder is found and its contents are displayed."""
    with patch('streamlit.file_uploader', return_value=create_mock_file()):
        at = AppTest.from_file("src/app.py")
        at.run()
        assert any("Found 2 files in the eJP folder" in s.value for s in at.success)
        assert any("file1.txt" in t.value for t in at.text)
        assert any("file2.txt" in t.value for t in at.text)

def test_ejp_folder_not_found(mock_process_zip_file):
    """Test the behavior when no eJP folder is found in the ZIP file."""
    mock_process_zip_file.return_value = (None, None)
    with patch('streamlit.file_uploader', return_value=create_mock_file(with_ejp=False)):
        at = AppTest.from_file("src/app.py")
        at.run()
        assert any("No eJP folder found in the ZIP file" in w.value for w in at.warning)