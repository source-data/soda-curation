"""Tests for caption extraction functionality."""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.soda_curation.pipeline.extract_captions.extract_captions_openai import (
    FigureCaptionExtractorOpenAI,
)
from src.soda_curation.pipeline.manuscript_structure.manuscript_structure import (
    Figure,
    Panel,
    ProcessingCost,
    ZipStructure,
)
from src.soda_curation.pipeline.prompt_handler import PromptHandler

# Test configurations
VALID_CONFIG = {
    "pipeline": {
        "extract_individual_captions": {
            "openai": {
                "model": "gpt-4o",
                "temperature": 0.1,
                "top_p": 1.0,
                "prompts": {"system": "System prompt", "user": "User prompt"},
            }
        }
    }
}

INVALID_MODEL_CONFIG = {
    "pipeline": {
        "extract_individual_captions": {"openai": {"model": "invalid-model"}},
    }
}

INVALID_PARAMS_CONFIG = {
    "pipeline": {
        "extract_individual_captions": {
            "openai": {
                "model": "gpt-4o",
                "temperature": 3.0,
                "top_p": 2.0,
            }
        }
    }
}

# Mock response content
MOCK_EXTRACT_RESPONSE = {
    "figures": [
        {
            "figure_label": "Figure 1",
            "caption_title": "Test caption for figure 1",
            "figure_caption": "A) Panel A description. B) Panel B description.",
            "panels": [
                {"panel_label": "A", "panel_caption": "Panel A description"},
                {"panel_label": "B", "panel_caption": "Panel B description"},
            ],
        },
        {
            "figure_label": "Figure 2",
            "caption_title": "Test caption for figure 2",
            "figure_caption": "Multiple panels showing different aspects.",
            "panels": [],
        },
    ]
}


@pytest.fixture
def mock_prompt_handler():
    """Create a mock prompt handler."""
    mock_handler = MagicMock()
    mock_handler.get_prompt.return_value = {
        "system": "System prompt",
        "user": "User prompt",
    }
    return mock_handler


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client with properly structured responses."""
    with patch("openai.OpenAI") as mock_client:
        instance = mock_client.return_value
        instance.beta.chat.completions.parse.return_value = MagicMock(
            choices=[
                MagicMock(message=MagicMock(content=json.dumps(MOCK_EXTRACT_RESPONSE)))
            ],
            usage=MagicMock(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
            ),
        )
        yield mock_client


@pytest.fixture
def zip_structure():
    """Create a basic ZipStructure instance for testing."""
    return ZipStructure(
        manuscript_id="test_manuscript",
        xml="test.xml",
        docx="test.docx",
        pdf="test.pdf",
        figures=[
            Figure(figure_label="Figure 1", img_files=[], sd_files=[]),
            Figure(figure_label="Figure 2", img_files=[], sd_files=[]),
        ],
        errors=[],
        cost=ProcessingCost(),
    )


class TestConfigValidation:
    """Test configuration validation functionality."""

    def test_valid_config(self, mock_prompt_handler):
        """Test that valid configuration is accepted."""
        extractor = FigureCaptionExtractorOpenAI(VALID_CONFIG, mock_prompt_handler)
        assert extractor.config == VALID_CONFIG

    def test_invalid_model(self, mock_prompt_handler):
        """Test that invalid model raises ValueError."""
        with pytest.raises(ValueError, match="Invalid model"):
            FigureCaptionExtractorOpenAI(INVALID_MODEL_CONFIG, mock_prompt_handler)

    def test_invalid_parameters(self, mock_prompt_handler):
        """Test that invalid parameters raise ValueError."""
        with pytest.raises(ValueError):
            FigureCaptionExtractorOpenAI(INVALID_PARAMS_CONFIG, mock_prompt_handler)


class TestIndividualCaptionExtraction:
    """Test individual caption extraction functionality."""

    def test_successful_extraction(
        self, mock_openai_client, mock_prompt_handler, zip_structure
    ):
        """Test successful extraction of individual captions."""
        extractor = FigureCaptionExtractorOpenAI(VALID_CONFIG, mock_prompt_handler)
        result = extractor.extract_individual_captions("test content", zip_structure)

        assert isinstance(result, ZipStructure)
        assert result.ai_response_extract_individual_captions == json.dumps(
            MOCK_EXTRACT_RESPONSE
        )

        # Verify figures were updated correctly
        for figure_data in MOCK_EXTRACT_RESPONSE["figures"]:
            matching_figures = [
                f
                for f in result.figures
                if f.figure_label == figure_data["figure_label"]
            ]
            if matching_figures:
                assert matching_figures[0].caption_title == figure_data["caption_title"]
                assert (
                    matching_figures[0].figure_caption == figure_data["figure_caption"]
                )

    def test_successful_extraction_with_panels(
        self, mock_openai_client, mock_prompt_handler, zip_structure
    ):
        """Test successful extraction of captions with panel information."""
        extractor = FigureCaptionExtractorOpenAI(VALID_CONFIG, mock_prompt_handler)
        updated_zip = extractor.extract_individual_captions(
            "test content", zip_structure
        )

        assert len(updated_zip.figures) == 2

        # Check Figure 1
        figure1 = updated_zip.figures[0]
        assert figure1.figure_label == "Figure 1"
        assert figure1.caption_title == "Test caption for figure 1"
        assert (
            figure1.figure_caption == "A) Panel A description. B) Panel B description."
        )
        assert len(figure1.panels) == 2
        assert figure1.panels[0].panel_label == "A"
        assert figure1.panels[0].panel_caption == "Panel A description"
        assert figure1.panels[1].panel_label == "B"
        assert figure1.panels[1].panel_caption == "Panel B description"

        # Check Figure 2
        figure2 = updated_zip.figures[1]
        assert figure2.figure_label == "Figure 2"
        assert figure2.caption_title == "Test caption for figure 2"
        assert figure2.figure_caption == "Multiple panels showing different aspects."
        assert len(figure2.panels) == 0

    def test_api_error_handling(
        self, mock_openai_client, mock_prompt_handler, zip_structure
    ):
        """Test handling of API errors."""
        mock_openai_client.return_value.beta.chat.completions.parse.side_effect = (
            Exception("API Error")
        )

        extractor = FigureCaptionExtractorOpenAI(VALID_CONFIG, mock_prompt_handler)
        updated_zip = extractor.extract_individual_captions(
            "test content", zip_structure
        )

        assert updated_zip.ai_response_extract_individual_captions == ""

    def test_empty_content(
        self, mock_openai_client, mock_prompt_handler, zip_structure
    ):
        """Test handling of empty content."""
        # Make sure empty content causes an error
        mock_openai_client.return_value.beta.chat.completions.parse.side_effect = (
            Exception("Empty content")
        )

        extractor = FigureCaptionExtractorOpenAI(VALID_CONFIG, mock_prompt_handler)
        result = extractor.extract_individual_captions("", zip_structure)

        assert isinstance(result, ZipStructure)
        # Error cases should preserve empty string
        assert result.ai_response_extract_individual_captions == ""


class TestResponseParsing:
    """Test parsing of AI responses."""

    def test_parse_valid_response(self, mock_prompt_handler):
        """Test parsing of valid response format."""
        extractor = FigureCaptionExtractorOpenAI(VALID_CONFIG, mock_prompt_handler)
        result = extractor._parse_response(json.dumps(MOCK_EXTRACT_RESPONSE))

        assert len(result) > 0
        assert "figures" in result
        assert isinstance(result["figures"], list)
        assert len(result["figures"]) == 2
        assert all(
            key in result["figures"][0]
            for key in ["figure_label", "caption_title", "figure_caption"]
        )

    def test_parse_invalid_response(self, mock_prompt_handler):
        """Test parsing of invalid response."""
        extractor = FigureCaptionExtractorOpenAI(VALID_CONFIG, mock_prompt_handler)
        result = extractor._parse_response("Invalid JSON content")

        assert result == {}


class TestPanelDuplication:
    def test_remove_duplicate_panels(self):
        """Test that duplicate panels are properly removed."""
        # Create a mock config and prompt handler
        config = {
            "pipeline": {
                "extract_individual_captions": {
                    "openai": {
                        "model": "gpt-4o",
                        "temperature": 0.1,
                        "top_p": 1.0,
                    }
                }
            }
        }

        # Create a proper pipeline config for the prompt handler
        pipeline_config = {
            "extract_individual_captions": {
                "openai": {
                    "prompts": {
                        "system": "Mock system prompt",
                        "user": "Mock user prompt",
                    }
                }
            }
        }

        prompt_handler = PromptHandler(pipeline_config)

        # Create an instance of the extractor
        extractor = FigureCaptionExtractorOpenAI(config, prompt_handler)

        # Create test figures with duplicate panels
        figures = [
            Figure(
                figure_label="Figure 1",
                img_files=["fig1.jpg"],
                sd_files=[],
                panels=[
                    Panel(panel_label="A", panel_caption="Panel A caption"),
                    Panel(panel_label="B", panel_caption="Panel B caption"),
                    Panel(
                        panel_label="A", panel_caption="Duplicate panel A"
                    ),  # Duplicate
                    Panel(panel_label="C", panel_caption="Panel C caption"),
                ],
                duplicated_panels=[],  # Initialize empty list
            )
        ]

        # Create a ZipStructure with the test figures
        zip_structure = ZipStructure(figures=figures)

        # Call the method to remove duplicate panels
        cleaned_structure = extractor._remove_duplicate_panels(zip_structure)

        # Check that duplicates were removed
        figure = cleaned_structure.figures[0]
        assert len(figure.panels) == 3
        panel_labels = [p.panel_label for p in figure.panels]
        assert panel_labels == ["A", "B", "C"]

        # Check that duplicates were tracked
        assert len(figure.duplicated_panels) == 1
        assert figure.duplicated_panels[0].panel_label == "A"
        assert figure.duplicated_panels[0].panel_caption == "Duplicate panel A"
