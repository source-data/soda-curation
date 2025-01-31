"""Tests for main utility functions."""

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


def test_setup_extract_dir(tmp_path):
    """Test extract directory setup."""
    zip_path = tmp_path / "test.zip"
    extract_dir = setup_extract_dir(str(zip_path))

    assert extract_dir.name == "test"
    assert extract_dir.parent == zip_path.parent


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


def test_cleanup_extract_dir_nonexistent(tmp_path):
    """Test cleanup of nonexistent directory."""
    extract_dir = tmp_path / "nonexistent"
    cleanup_extract_dir(extract_dir)  # Should not raise
