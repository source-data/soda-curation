"""Tests for main module CLI arguments and basic functionality."""

import json
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.soda_curation.main import main
from src.soda_curation.pipeline.manuscript_structure.manuscript_structure import (
    ProcessingCost,
    TokenUsage,
    ZipStructure,
)

MOCK_CONFIG = {
    "pipeline": {
        "extract_sections": {
            "openai": {
                "model": "gpt-4o",
                "temperature": 0.1,
                "top_p": 1.0,
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "prompts": {"system": "System prompt", "user": "User prompt"},
            }
        },
        "extract_individual_captions": {
            "openai": {
                "model": "gpt-4o",
                "temperature": 0.1,
                "top_p": 1.0,
                "prompts": {"system": "System prompt", "user": "User prompt"},
            }
        },
        "extract_data_sources": {
            "openai": {
                "model": "gpt-4o",
                "temperature": 0.1,
                "top_p": 1.0,
                "prompts": {"system": "System prompt", "user": "User prompt"},
            }
        },
        "assign_panel_source": {
            "openai": {
                "model": "gpt-4o",
                "temperature": 0.1,
                "top_p": 1.0,
                "prompts": {"system": "System prompt", "user": "User prompt"},
            }
        },
        "match_caption_panel": {
            "openai": {
                "model": "gpt-4o",
                "temperature": 0.1,
                "top_p": 1.0,
                "prompts": {"system": "System prompt", "user": "User prompt"},
            }
        },
    }
}


@pytest.fixture
def mock_paths(tmp_path):
    """Create mock paths for testing."""
    return {
        "zip_path": str(tmp_path / "test.zip"),
        "config_path": str(tmp_path / "config.yaml"),
        "output_path": str(tmp_path / "output.json"),
    }


@pytest.fixture
def mock_zip_content(tmp_path):
    """Create a mock ZIP file with required structure."""
    zip_path = tmp_path / "test.zip"
    manuscript_id = "EMBOJ-DUMMY-ZIP"

    with zipfile.ZipFile(zip_path, "w") as zf:
        # Create required files and directories
        zf.writestr(f"{manuscript_id}.xml", "<xml>dummy content</xml>")
        zf.writestr(f"Doc/{manuscript_id}Manuscript_TextIG.docx", "dummy content")
        zf.writestr("graphic/FIGURE 1.tif", "dummy image content")
        zf.writestr("graphic/FIGURE 2.tif", "dummy image content")
        zf.writestr(f"pdf/{manuscript_id}.pdf", "dummy pdf content")
        zf.writestr("suppl_data/Figure_3sd.zip", "dummy zip content")

    return str(zip_path)


@pytest.fixture
def mock_config(tmp_path):
    """Create a mock config file."""
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        json.dump(MOCK_CONFIG, f)
    return str(config_path)


@pytest.fixture
def mock_structure():
    """Create a mock ZipStructure that can be serialized."""
    cost = ProcessingCost()
    cost.extract_sections = TokenUsage(
        prompt_tokens=0, completion_tokens=0, total_tokens=0, cost=0.0
    )

    structure = ZipStructure(
        manuscript_id="EMBOJ-DUMMY-ZIP",
        xml="EMBOJ-DUMMY-ZIP.xml",
        docx="Doc/EMBOJ-DUMMY-ZIPManuscript_TextIG.docx",
        pdf="pdf/EMBOJ-DUMMY-ZIP.pdf",
        figures=[],
        errors=[],
        appendix=[],
        non_associated_sd_files=[],
        cost=cost,
        data_availability={"section_text": "", "data_sources": []},
        ai_response_locate_captions="",
        ai_response_extract_individual_captions="",
    )

    # Ensure private attributes are not creating circular references
    structure._full_appendix = []
    structure._full_docx = ""
    structure._full_pdf = ""
    structure.ai_config = {}
    structure.ai_provider = ""

    return structure


@patch("yaml.safe_load")
def test_main_creates_output_directory(
    mock_yaml_load, mock_zip_content, mock_config, mock_structure, tmp_path
):
    """Test main creates output directory if it doesn't exist."""
    # Mock the yaml config loading
    mock_yaml_load.return_value = {"default": MOCK_CONFIG}

    # Create output path in nonexistent directory
    output_dir = tmp_path / "nonexistent" / "nested" / "path"
    output_path = str(output_dir / "result.json")

    # Add patches for pipeline components
    with patch("src.soda_curation.main.setup_extract_dir") as mock_setup_dir, patch(
        "src.soda_curation.main.XMLStructureExtractor"
    ) as mock_extractor, patch(
        "src.soda_curation.main.SectionExtractorOpenAI"
    ) as mock_section_extractor, patch(
        "src.soda_curation.main.FigureCaptionExtractorOpenAI"
    ) as mock_caption_extractor, patch(
        "src.soda_curation.main.DataAvailabilityExtractorOpenAI"
    ) as mock_data_extractor:
        # Configure mocks
        extract_dir = tmp_path / "extract"
        mock_setup_dir.return_value = extract_dir

        # XML extractor mock
        mock_instance = MagicMock()
        mock_instance.extract_structure.return_value = mock_structure
        mock_instance.extract_docx_content.return_value = "test content"
        mock_extractor.return_value = mock_instance

        # Section extractor mock
        mock_section_instance = MagicMock()
        mock_section_instance.extract_sections.return_value = (
            "test legends",
            "test data",
            mock_structure,
        )
        mock_section_extractor.return_value = mock_section_instance

        # Caption extractor mock
        mock_caption_instance = MagicMock()
        mock_caption_instance.extract_individual_captions.return_value = mock_structure
        mock_caption_extractor.return_value = mock_caption_instance

        # Data extractor mock
        mock_data_instance = MagicMock()
        mock_data_instance.extract_data_sources.return_value = mock_structure
        mock_data_extractor.return_value = mock_data_instance

        # Run main
        main(mock_zip_content, mock_config, output_path)

        # Check that directory and file were created
        assert output_dir.exists()
        assert Path(output_path).exists()


@patch("yaml.safe_load")
def test_main_successful_run(
    mock_yaml_load, mock_zip_content, mock_config, mock_structure
):
    """Test successful execution of main function."""
    # Mock the yaml config loading
    mock_yaml_load.return_value = {"default": MOCK_CONFIG}

    # Create a ZipStructure instance
    test_structure = ZipStructure(
        manuscript_id="EMBOJ-DUMMY-ZIP",
        xml="EMBOJ-DUMMY-ZIP.xml",
        docx="Doc/EMBOJ-DUMMY-ZIPManuscript_TextIG.docx",
        pdf="pdf/EMBOJ-DUMMY-ZIP.pdf",
        figures=[],
        errors=[],
        appendix=[],
        non_associated_sd_files=[],
        cost=ProcessingCost(),
        data_availability={"section_text": "", "data_sources": []},
        ai_response_locate_captions="",
        ai_response_extract_individual_captions="",
    )

    with patch("src.soda_curation.main.XMLStructureExtractor") as mock_extractor, patch(
        "src.soda_curation.main.SectionExtractorOpenAI"
    ) as mock_section_extractor, patch(
        "src.soda_curation.main.FigureCaptionExtractorOpenAI"
    ) as mock_caption_extractor, patch(
        "src.soda_curation.main.DataAvailabilityExtractorOpenAI"
    ) as mock_data_extractor, patch(
        "src.soda_curation.main.PanelSourceAssignerOpenAI"
    ) as mock_panel_assigner:
        # Configure mocks
        mock_instance = MagicMock()
        mock_instance.extract_structure.return_value = test_structure
        mock_instance.extract_docx_content.return_value = "test content"
        mock_extractor.return_value = mock_instance

        mock_section_instance = MagicMock()
        mock_section_instance.extract_sections.return_value = (
            "test legends",
            "test data",
            test_structure,
        )
        mock_section_extractor.return_value = mock_section_instance

        mock_caption_instance = MagicMock()
        mock_caption_instance.extract_individual_captions.return_value = test_structure
        mock_caption_extractor.return_value = mock_caption_instance

        mock_data_instance = MagicMock()
        mock_data_instance.extract_data_sources.return_value = test_structure
        mock_data_extractor.return_value = mock_data_instance

        # Update this mock to return a list of figures
        mock_panel_instance = MagicMock()
        mock_panel_instance.assign_panel_source.return_value = test_structure.figures
        mock_panel_assigner.return_value = mock_panel_instance

        # Run main
        result = main(mock_zip_content, mock_config)

        # Verify output
        assert isinstance(result, str)
        result_dict = json.loads(result)
        assert (
            "manuscript_id" in result_dict
        ), f"Expected 'manuscript_id' in {result_dict}"
        assert result_dict["manuscript_id"] == "EMBOJ-DUMMY-ZIP"


@patch("yaml.safe_load")
def test_main_no_output_path_returns_json(
    mock_yaml_load, mock_zip_content, mock_config, mock_structure
):
    """Test main returns JSON string when no output path provided."""
    # Mock the yaml config loading
    mock_yaml_load.return_value = {"default": MOCK_CONFIG}

    # Create a ZipStructure instance instead of a dictionary
    test_structure = ZipStructure(
        manuscript_id="EMBOJ-DUMMY-ZIP",
        xml="EMBOJ-DUMMY-ZIP.xml",
        docx="Doc/EMBOJ-DUMMY-ZIPManuscript_TextIG.docx",
        pdf="pdf/EMBOJ-DUMMY-ZIP.pdf",
        figures=[],
        errors=[],
        appendix=[],
        non_associated_sd_files=[],
        cost=ProcessingCost(),
        data_availability={"section_text": "", "data_sources": []},
        ai_response_locate_captions="",
        ai_response_extract_individual_captions="",
    )

    with patch("src.soda_curation.main.XMLStructureExtractor") as mock_extractor, patch(
        "src.soda_curation.main.SectionExtractorOpenAI"
    ) as mock_section_extractor, patch(
        "src.soda_curation.main.FigureCaptionExtractorOpenAI"
    ) as mock_caption_extractor, patch(
        "src.soda_curation.main.DataAvailabilityExtractorOpenAI"
    ) as mock_data_extractor:
        # Configure mocks to return the ZipStructure object
        mock_instance = MagicMock()
        mock_instance.extract_structure.return_value = test_structure
        mock_instance.extract_docx_content.return_value = "test content"
        mock_extractor.return_value = mock_instance

        mock_section_instance = MagicMock()
        mock_section_instance.extract_sections.return_value = (
            "test legends",
            "test data",
            test_structure,
        )
        mock_section_extractor.return_value = mock_section_instance

        mock_caption_instance = MagicMock()
        mock_caption_instance.extract_individual_captions.return_value = test_structure
        mock_caption_extractor.return_value = mock_caption_instance

        mock_data_instance = MagicMock()
        mock_data_instance.extract_data_sources.return_value = test_structure
        mock_data_extractor.return_value = mock_data_instance

        result = main(mock_zip_content, mock_config)
        assert isinstance(result, str)
        result_dict = json.loads(result)
        assert result_dict["manuscript_id"] == "EMBOJ-DUMMY-ZIP"
