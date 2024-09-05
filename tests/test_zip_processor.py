
"""
Tests for the ZIP Processor module.

This module contains unit tests for the ZIP processing functionality.
"""

import pytest
import os
import zipfile
import tempfile
import sys

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from zip_processor import process_zip_file

def create_test_zip(tmp_path, with_ejp=True):
    """Create a test ZIP file."""
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        if with_ejp:
            zf.writestr('some_folder/eJP/file1.txt', 'content1')
            zf.writestr('some_folder/eJP/file2.txt', 'content2')
        else:
            zf.writestr('some_folder/file1.txt', 'content1')
    return zip_path

def test_process_zip_file_with_ejp(tmp_path):
    """Test processing a ZIP file with an eJP folder."""
    zip_path = create_test_zip(tmp_path)
    with tempfile.TemporaryDirectory() as extract_dir:
        ejp_files, ejp_process_dir = process_zip_file(zip_path, extract_dir)
        
        assert ejp_files is not None
        assert len(ejp_files) == 2
        assert 'file1.txt' in ejp_files
        assert 'file2.txt' in ejp_files
        assert os.path.isdir(ejp_process_dir)
        assert len(os.listdir(ejp_process_dir)) == 2

def test_process_zip_file_without_ejp(tmp_path):
    """Test processing a ZIP file without an eJP folder."""
    zip_path = create_test_zip(tmp_path, with_ejp=False)
    with tempfile.TemporaryDirectory() as extract_dir:
        ejp_files, ejp_process_dir = process_zip_file(zip_path, extract_dir)
        
        assert ejp_files is None
        assert ejp_process_dir is None