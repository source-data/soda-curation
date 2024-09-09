import pytest
from unittest.mock import patch, MagicMock
import io
import zipfile
import os
import sys
from streamlit.testing.v1 import AppTest

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
    with patch('src.zip_processor.process_zip_file') as mock:
        mock.return_value = (['file1.txt', 'file2.txt'], '/tmp/ejp_files')
        yield mock

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

def test_app_title():
    """Test if the app title is set to 'SODA Curation Tool'."""
    at = AppTest.from_file("src/app.py")
    at.run(timeout=10)
    assert any("SODA Curation Tool" in t.value for t in at.title)

def test_upload_section_present():
    """Test if the upload section is present in the app."""
    at = AppTest.from_file("src/app.py")
    at.run(timeout=10)
    assert any("Upload ZIP File" in h.value for h in at.header)

def test_file_upload_feedback(mock_process_zip_file, mock_get_file_structure):
    """Test if the app provides feedback when a file is uploaded."""
    with patch('streamlit.file_uploader', return_value=create_mock_file()):
        at = AppTest.from_file("src/app.py")
        at.run(timeout=15)
        assert any("test.zip" in s.value for s in at.success)

def test_file_details_displayed(mock_process_zip_file, mock_get_file_structure):
    """Test if the file details are displayed after a file is uploaded."""
    with patch('streamlit.file_uploader', return_value=create_mock_file()):
        at = AppTest.from_file("src/app.py")
        at.run(timeout=15)
        all_text = ' '.join([t.value for t in at.text if hasattr(t, 'value')])
        assert "FileName" in all_text
        assert "FileType" in all_text
        assert "FileSize" in all_text

def test_ejp_folder_found(mock_process_zip_file, mock_get_file_structure):
    """Test if the eJP folder is found and processed correctly."""
    with patch('streamlit.file_uploader', return_value=create_mock_file()):
        at = AppTest.from_file("src/app.py")
        at.run(timeout=15)
        
        print("Success messages:")
        for s in at.success:
            print(s.value)
        
        print("Info messages:")
        for i in at.info:
            print(i.value)
        
        print("Subheaders:")
        for s in at.subheader:
            print(s.value)
        
        print("JSON outputs:")
        for j in at.json:
            print(j.value)
        
        # Check for eJP folder found message
        assert any("Found" in s.value and "files in the eJP folder" in s.value for s in at.success), "eJP folder found message not displayed"
        
        # Check for any indication that file was processed
        file_processed = len(at.json) > 0 or len(at.subheader) > 0
        assert file_processed, "No indication that file was processed (no JSON output or subheaders)"
        
        # Additional checks to ensure the file was processed
        assert len(at.json) > 0, "No JSON output found, file might not have been processed"
        assert len(at.subheader) > 0, "No subheaders found, file might not have been processed correctly"

def test_file_processing_info(mock_process_zip_file, mock_get_file_structure):
    """Test if the information of the processed file is displayed correctly."""
    with patch('streamlit.file_uploader', return_value=create_mock_file()):
        at = AppTest.from_file("src/app.py")
        at.run(timeout=15)
        
        print("Success messages:")
        for s in at.success:
            print(s.value)
        
        print("Info messages:")
        for i in at.info:
            print(i.value)
        
        print("Subheaders:")
        for s in at.subheader:
            print(s.value)
        
        print("All text:")
        for t in at.text:
            print(t.value)
        
        # Check for specific success messages
        success_messages = [s.value for s in at.success]
        assert any("successfully uploaded" in msg for msg in success_messages), "File upload success message not found"
        assert any("Found 2 files in the eJP folder" in msg for msg in success_messages), "eJP folder files message not found"
        
        # Check for JSON output
        json_outputs = at.json
        assert len(json_outputs) > 0, "JSON output not found"
        
        # Check for file structure information
        assert any("Initial file structure" in s.value for s in at.subheader), "Initial file structure header not found"
        
        # Check for tabs
        tab_labels = [tab.label for tab in at.tabs]
        assert "File Structure" in tab_labels, "File Structure tab not found"
        assert "Future Step 1" in tab_labels, "Future Step 1 tab not found"
        assert "Future Step 2" in tab_labels, "Future Step 2 tab not found"
