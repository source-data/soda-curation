"""Tests for main utility functions."""

import os
import shutil
import zipfile
from pathlib import Path

import pytest

from src.soda_curation._main_utils import (
    cleanup_extract_dir,
    setup_extract_dir,
    validate_paths,
    write_output,
)


@pytest.fixture
def mock_paths(tmp_path):
    """Create mock paths for testing."""
    return {
        "zip_path": str(tmp_path / "test.zip"),
        "config_path": str(tmp_path / "config.yaml"),
        "output_path": str(tmp_path / "output.json"),
    }


@pytest.fixture
def mock_zip_with_docx(tmp_path):
    """Create a test ZIP file with a DOCX in manuscript ID directory structure."""
    zip_path = tmp_path / "test.zip"
    manuscript_id = "TEST-2023-12345"
    docx_content = b"test docx content"

    with zipfile.ZipFile(zip_path, "w") as zf:
        # Add files preserving manuscript ID prefix
        zf.writestr(f"{manuscript_id}/Doc/manuscript.docx", docx_content)
        zf.writestr(f"{manuscript_id}.xml", "<xml>test</xml>")

    return zip_path, manuscript_id, docx_content


def test_validate_paths_missing_zip():
    """Test validation fails when no ZIP path provided."""
    with pytest.raises(ValueError, match="ZIP path must be provided"):
        validate_paths(None, "config.yaml")


def test_validate_paths_missing_config():
    """Test validation fails when no config path provided."""
    with pytest.raises(ValueError, match="config path must be provided"):
        validate_paths("test.zip", None)


def test_validate_paths_nonexistent_zip(mock_paths):
    """Test validation fails when ZIP file doesn't exist."""
    with pytest.raises(FileNotFoundError, match="ZIP file .* does not exist"):
        validate_paths(mock_paths["zip_path"], mock_paths["config_path"])


def test_validate_paths_nonexistent_config(mock_paths):
    """Test validation fails when config file doesn't exist."""
    Path(mock_paths["zip_path"]).touch()
    with pytest.raises(FileNotFoundError, match="Config file .* does not exist"):
        validate_paths(mock_paths["zip_path"], mock_paths["config_path"])


def test_validate_paths_creates_output_dir(mock_paths, tmp_path):
    """Test validation creates output directory if needed."""
    Path(mock_paths["zip_path"]).touch()
    Path(mock_paths["config_path"]).touch()

    nested_output = tmp_path / "nested" / "path" / "output.json"
    validate_paths(
        mock_paths["zip_path"], mock_paths["config_path"], str(nested_output)
    )

    assert nested_output.parent.exists()


def test_setup_extract_dir():
    """Test temporary directory creation."""
    extract_dir = setup_extract_dir()
    try:
        assert extract_dir.exists()
        assert extract_dir.is_dir()
        assert "soda_curation_" in extract_dir.name

        # Verify we can write to it
        test_file = extract_dir / "test.txt"
        test_file.write_text("test")
        assert test_file.exists()

    finally:
        cleanup_extract_dir(extract_dir)
        assert not extract_dir.exists()


def test_write_output_success(tmp_path):
    """Test successful output writing."""
    output_path = tmp_path / "output.json"
    test_json = '{"test": "data"}'

    write_output(test_json, str(output_path))

    assert output_path.exists()
    assert output_path.read_text() == test_json


def test_write_output_failure(tmp_path):
    """Test output writing failure."""
    invalid_path = tmp_path / "nonexistent" / "output.json"

    with pytest.raises(Exception):
        write_output('{"test": "data"}', str(invalid_path))


def test_cleanup_extract_dir(tmp_path):
    """Test extraction directory cleanup."""
    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    (extract_dir / "test.txt").touch()

    cleanup_extract_dir(extract_dir)

    assert not extract_dir.exists()


def test_cleanup_nonexistent_dir():
    """Test cleanup of nonexistent directory doesn't raise."""
    cleanup_extract_dir(Path("/nonexistent/path"))


def test_extract_and_find_docx(mock_zip_with_docx):
    """Test extracting ZIP and finding DOCX with manuscript ID prefix."""
    zip_path, manuscript_id, docx_content = mock_zip_with_docx
    extract_dir = setup_extract_dir()

    try:
        # Extract ZIP contents
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

        # Verify DOCX exists at expected path with manuscript ID
        docx_path = extract_dir / manuscript_id / "Doc/manuscript.docx"
        assert docx_path.exists()
        assert docx_path.read_bytes() == docx_content

        # Verify direct path from manuscript ID prefix works
        docx_relative = f"{manuscript_id}/Doc/manuscript.docx"
        docx_full = extract_dir / docx_relative
        assert docx_full.exists()
        assert docx_full.read_bytes() == docx_content

    finally:
        cleanup_extract_dir(extract_dir)
        assert not extract_dir.exists()


def test_extract_dir_cleanup_handles_permission_error(tmp_path):
    """Test cleanup handles permission errors gracefully."""
    extract_dir = setup_extract_dir()
    try:
        # Create nested structure
        nested_dir = extract_dir / "nested"
        nested_dir.mkdir()
        test_file = nested_dir / "test.txt"
        test_file.write_text("test")

        # Make directory read-only on Unix-like systems
        if os.name != "nt":  # Skip on Windows
            nested_dir.chmod(0o555)

        cleanup_extract_dir(extract_dir)
        # Even with permission error, should not raise

    finally:
        # Ensure cleanup in case of test failure
        if extract_dir.exists():
            # Reset permissions to allow cleanup
            if os.name != "nt":
                nested_dir.chmod(0o755)
            shutil.rmtree(extract_dir)
