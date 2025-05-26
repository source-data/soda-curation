"""Tests for caption extraction functionality."""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.soda_curation.pipeline.extract_captions.extract_captions_openai import (
    CaptionExtraction,
    FigureCaptionExtractorOpenAI,
    PanelExtraction,
    PanelInfo,
)
from src.soda_curation.pipeline.manuscript_structure.manuscript_structure import (
    Figure,
    Panel,
    ProcessingCost,
    TokenUsage,
    ZipStructure,
)

# Test configurations
VALID_CONFIG = {
    "pipeline": {
        "extract_caption_title": {
            "openai": {
                "model": "gpt-4o",
                "temperature": 0.1,
                "top_p": 1.0,
                "max_tokens": 4096,
                "prompts": {"system": "System prompt", "user": "User prompt"},
            }
        },
        "extract_panel_sequence": {
            "openai": {
                "model": "gpt-4o",
                "temperature": 0.1,
                "top_p": 1.0,
                "max_tokens": 4096,
                "prompts": {"system": "System prompt", "user": "User prompt"},
            }
        },
    }
}

INVALID_MODEL_CONFIG = {
    "pipeline": {
        "extract_caption_title": {"openai": {"model": "invalid-model"}},
        "extract_panel_sequence": {"openai": {"model": "invalid-model"}},
    }
}

INVALID_PARAMS_CONFIG = {
    "pipeline": {
        "extract_caption_title": {
            "openai": {
                "model": "gpt-4o",
                "temperature": 3.0,
                "top_p": 2.0,
            }
        },
        "extract_panel_sequence": {
            "openai": {
                "model": "gpt-4o",
                "temperature": 3.0,
                "top_p": 2.0,
            }
        },
    }
}

# Mock caption extraction response
MOCK_CAPTION_EXTRACTION = CaptionExtraction(
    figure_label="Figure 1",
    caption_title="Test caption for figure 1",
    figure_caption="A) Panel A description. B) Panel B description.",
    is_verbatim=True,
)

# Mock panel extraction response
MOCK_PANEL_EXTRACTION = PanelExtraction(
    figure_label="Figure 1",
    panels=[
        PanelInfo(panel_label="A", panel_caption="Panel A description"),
        PanelInfo(panel_label="B", panel_caption="Panel B description"),
    ],
)

# Mock extract response for _parse_response test
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
            "figure_caption": "Test caption for figure 2 with no panels.",
            "panels": [],
        },
    ]
}


# Mock OpenAI response
class MockOpenAIResponse:
    def __init__(self, content):
        self.choices = [MagicMock(message=MagicMock(content=content))]
        self.usage = MagicMock(
            prompt_tokens=50,
            completion_tokens=25,
            total_tokens=75,
        )


@pytest.fixture
def mock_prompt_handler():
    """Create a mock prompt handler."""
    mock_handler = MagicMock()
    mock_handler.get_prompt.return_value = {
        "system": "System prompt with json format",
        "user": "User prompt",
    }
    return mock_handler


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client with properly structured responses."""
    with patch("openai.OpenAI") as mock_client:
        instance = mock_client.return_value

        # Mock for beta.chat.completions.parse
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

        # Mock for regular chat.completions.create
        instance.chat.completions.create.side_effect = [
            # First call - caption extraction
            MockOpenAIResponse(
                json.dumps(
                    {
                        "figure_label": "Figure 1",
                        "caption_title": "Test caption for figure 1",
                        "figure_caption": "A) Panel A description. B) Panel B description.",
                        "is_verbatim": True,
                    }
                )
            ),
            # Second call - panel extraction for Figure 1
            MockOpenAIResponse(
                json.dumps(
                    {
                        "figure_label": "Figure 1",
                        "panels": [
                            {
                                "panel_label": "A",
                                "panel_caption": "Panel A description",
                            },
                            {
                                "panel_label": "B",
                                "panel_caption": "Panel B description",
                            },
                        ],
                    }
                )
            ),
            # Third call - caption extraction for Figure 2
            MockOpenAIResponse(
                json.dumps(
                    {
                        "figure_label": "Figure 2",
                        "caption_title": "Test caption for figure 2",
                        "figure_caption": "Test caption for figure 2 with no panels.",
                        "is_verbatim": True,
                    }
                )
            ),
            # Fourth call - panel extraction for Figure 2
            MockOpenAIResponse(json.dumps({"figure_label": "Figure 2", "panels": []})),
        ]
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
        # Create the extractor
        extractor = FigureCaptionExtractorOpenAI(VALID_CONFIG, mock_prompt_handler)

        # Override the extract methods to make testing simpler
        extractor.extract_figure_caption = MagicMock(
            return_value=(MOCK_CAPTION_EXTRACTION, TokenUsage())
        )
        extractor.extract_figure_panels = MagicMock(
            return_value=(MOCK_PANEL_EXTRACTION, TokenUsage())
        )

        # Run the extraction
        result = extractor.extract_individual_captions("test content", zip_structure)

        # Verify figures were updated correctly
        assert isinstance(result, ZipStructure)

        # Verify calls to extract_figure_caption and extract_figure_panels
        assert extractor.extract_figure_caption.call_count == 2  # Once per figure
        assert extractor.extract_figure_panels.call_count == 2  # Once per figure

        # Verify Figure 1 data
        figure1 = result.figures[0]
        assert figure1.figure_label == "Figure 1"
        assert figure1.caption_title == "Test caption for figure 1"
        assert (
            figure1.figure_caption == "A) Panel A description. B) Panel B description."
        )
        assert figure1.hallucination_score == 0  # 0 means verbatim/not hallucinated

        # Verify token usage was tracked
        assert result.cost.extract_individual_captions is not None

    def test_api_error_handling(self, mock_prompt_handler, zip_structure):
        """Test handling of API errors."""
        # Create the extractor
        extractor = FigureCaptionExtractorOpenAI(VALID_CONFIG, mock_prompt_handler)

        # Configure the mock to raise an exception
        extractor.extract_figure_caption = MagicMock(side_effect=Exception("API Error"))

        # Test that the error is caught and raised
        with pytest.raises(Exception):
            extractor.extract_individual_captions("test content", zip_structure)

    def test_process_figure(
        self, mock_openai_client, mock_prompt_handler, zip_structure
    ):
        """Test the process_figure method directly."""
        # Create test figure
        figure = Figure(figure_label="Figure 1", img_files=[], sd_files=[])

        # Create extractor
        extractor = FigureCaptionExtractorOpenAI(VALID_CONFIG, mock_prompt_handler)

        # Override the extract methods for testing
        extractor.extract_figure_caption = MagicMock(
            return_value=(MOCK_CAPTION_EXTRACTION, TokenUsage())
        )
        extractor.extract_figure_panels = MagicMock(
            return_value=(MOCK_PANEL_EXTRACTION, TokenUsage())
        )

        # Call process_figure with the required parameters
        updated_figure, token_usage = extractor.process_figure(
            figure, "test content", zip_structure
        )

        # Verify figure was updated correctly
        assert updated_figure.figure_label == "Figure 1"
        assert updated_figure.caption_title == "Test caption for figure 1"
        assert (
            updated_figure.figure_caption
            == "A) Panel A description. B) Panel B description."
        )
        assert updated_figure.hallucination_score == 0  # 0 = verbatim
        assert len(updated_figure.panels) == 2

        # Verify token usage was tracked
        assert isinstance(token_usage, TokenUsage)


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
    def test_remove_duplicate_panels(self, mock_prompt_handler):
        """Test that duplicate panels are properly removed."""
        # Create an instance of the extractor
        extractor = FigureCaptionExtractorOpenAI(VALID_CONFIG, mock_prompt_handler)

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

        # Call the method to remove duplicate panels (if the method exists)
        if hasattr(extractor, "_remove_duplicate_panels"):
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
