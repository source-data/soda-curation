"""Tests for main module CLI arguments and basic functionality."""

import json
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.soda_curation.main import main
from src.soda_curation.pipeline.manuscript_structure.manuscript_structure import (
    ZipStructure,
)


@pytest.fixture
def mock_paths(tmp_path):
    """Create mock paths for testing."""
    return {
        "zip_path": str(tmp_path / "test.zip"),
        "config_path": str(tmp_path / "config.yaml"),
        "output_path": str(tmp_path / "output.json"),
    }


def test_main_missing_zip_path():
    """Test main fails when no ZIP path provided."""
    with pytest.raises(ValueError, match="ZIP path must be provided"):
        main(None, "config.yaml")


def test_main_missing_config_path():
    """Test main fails when no config path provided."""
    with pytest.raises(ValueError, match="config path must be provided"):
        main("test.zip", None)


def test_main_zip_not_found(mock_paths):
    """Test main fails when ZIP file doesn't exist."""
    with pytest.raises(FileNotFoundError, match="ZIP file .* does not exist"):
        main(mock_paths["zip_path"], mock_paths["config_path"])


def test_main_config_not_found(mock_paths, tmp_path):
    """Test main fails when config file doesn't exist."""
    # Create dummy ZIP
    Path(mock_paths["zip_path"]).touch()

    with pytest.raises(FileNotFoundError, match="Config file .* does not exist"):
        main(mock_paths["zip_path"], mock_paths["config_path"])


@pytest.fixture
def mock_zip_content(tmp_path):
    """Create a mock ZIP file with required structure."""
    zip_path = tmp_path / "test.zip"
    manuscript_id = "EMBOJ-DUMMY-ZIP"

    with zipfile.ZipFile(zip_path, "w") as zf:
        # Create required files and directories
        zf.writestr(f"{manuscript_id}.xml", "<xml>dummy content</xml>")
        zf.writestr(f"Doc/{manuscript_id}Manuscript_TextIG.docx", "dummy content")
        zf.writestr("graphic/", "")  # Empty string for directories
        zf.writestr("graphic/FIGURE 1.tif", "dummy image content")
        zf.writestr("graphic/FIGURE 2.tif", "dummy image content")
        zf.writestr("pdf/", "")
        zf.writestr(f"pdf/{manuscript_id}.pdf", "dummy pdf content")
        zf.writestr("suppl_data/", "")
        zf.writestr("suppl_data/Figure_3sd.zip", "dummy zip content")
        zf.writestr("prod_forms/", "")

    return str(zip_path)


@pytest.fixture
def mock_config(tmp_path):
    """Create a mock config file with all required sections."""
    config_path = tmp_path / "config.yaml"
    config_content = """
default: &default
    pipeline:
        locate_captions:
            openai:
                model: gpt-4o
                temperature: 0.1
                prompts:
                    system: "System prompt"
                    user: "User prompt"
        extract_individual_captions:
            openai:
                model: gpt-4o
                temperature: 0.1
                prompts:
                    system: "System prompt"
                    user: "User prompt"
        locate_data_availability:
            openai:
                model: gpt-4o
                temperature: 0.1
                prompts:
                    system: "System prompt"
                    user: "User prompt"
        extract_data_sources:
            openai:
                model: gpt-4o-mini
                temperature: 0.1
                prompts:
                    system: "System prompt"
                    user: "User prompt"
        object_detection:
            model_path: "data/models/panel_detection_model_no_labels.pt"
            confidence_threshold: 0.25
            iou_threshold: 0.1
            image_size: 512
            max_detections: 30

dev:
    <<: *default
"""
    config_path.write_text(config_content)
    return str(config_path)


@pytest.fixture
def mock_structure():
    """Create a mock ZipStructure that can be serialized."""
    return ZipStructure(
        manuscript_id="EMBOJ-DUMMY-ZIP",
        xml="EMBOJ-DUMMY-ZIP.xml",
        docx="Doc/EMBOJ-DUMMY-ZIPManuscript_TextIG.docx",
        pdf="pdf/EMBOJ-DUMMY-ZIP.pdf",
        figures=[],
        errors=[],
        appendix=[],
        non_associated_sd_files=[],
    )


def test_main_creates_output_directory(
    mock_zip_content, mock_config, mock_structure, tmp_path
):
    """Test main creates output directory if it doesn't exist."""
    # Create output path in nonexistent directory
    output_dir = tmp_path / "nonexistent" / "nested" / "path"
    output_path = str(output_dir / "result.json")

    # Add patch for temporary directory creation
    with patch("src.soda_curation.main.setup_extract_dir") as mock_setup_dir, patch(
        "src.soda_curation.main.XMLStructureExtractor"
    ) as mock_extractor:
        # Configure mocks
        extract_dir = tmp_path / "extract"
        mock_setup_dir.return_value = extract_dir
        mock_instance = MagicMock()
        mock_instance.extract_structure.return_value = mock_structure
        mock_extractor.return_value = mock_instance

        # Run main
        main(mock_zip_content, mock_config, output_path)

        # Check that directory and file were created
        assert output_dir.exists()
        assert Path(output_path).exists()

        # Verify mock was called correctly with the controlled temporary directory
        mock_extractor.assert_called_once_with(mock_zip_content, str(extract_dir))


def test_main_successful_run(mock_zip_content, mock_config, mock_structure):
    """Test successful execution of main function."""
    with patch("src.soda_curation.main.XMLStructureExtractor") as mock_extractor:
        # Configure mock
        mock_instance = MagicMock()
        mock_instance.extract_structure.return_value = mock_structure
        mock_extractor.return_value = mock_instance

        # Run main
        result = main(mock_zip_content, mock_config)

        # Verify output is valid JSON string
        assert isinstance(result, str)
        result_dict = json.loads(result)
        assert "manuscript_id" in result_dict
        assert result_dict["manuscript_id"] == "EMBOJ-DUMMY-ZIP"


def test_main_no_output_path_returns_json(
    mock_zip_content, mock_config, mock_structure
):
    """Test main returns JSON string when no output path provided."""
    with patch("src.soda_curation.main.XMLStructureExtractor") as mock_extractor:
        # Configure mock
        mock_instance = MagicMock()
        mock_instance.extract_structure.return_value = mock_structure
        mock_extractor.return_value = mock_instance

        result = main(mock_zip_content, mock_config)
        assert isinstance(result, str)
        assert json.loads(result)["manuscript_id"] == "EMBOJ-DUMMY-ZIP"
