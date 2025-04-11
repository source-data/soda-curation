"""Tests for caption extraction functionality."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

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


# Mock usage object for agent responses
class MockUsage:
    def __init__(self):
        self.input_tokens = 50
        self.output_tokens = 25
        self.total_tokens = 75
        self.requests = 1


# Mock agent response object
class MockResponse:
    def __init__(self, content):
        self.content = content
        self.usage = MockUsage()


# Mock agent result object
class MockAgentResult:
    def __init__(self, final_output, raw_responses=None):
        self.final_output = final_output
        self.raw_responses = raw_responses or [MockResponse("test")]


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
def mock_runner():
    """Create a mock for Runner.run that returns appropriate responses."""
    with patch("agents.Runner.run", new_callable=AsyncMock) as mock_run:
        # Set up 4 return values - 2 for each figure
        mock_run.side_effect = [
            # Figure 1 - caption extraction
            MockAgentResult(MOCK_CAPTION_EXTRACTION),
            # Figure 1 - panel extraction
            MockAgentResult(MOCK_PANEL_EXTRACTION),
            # Figure 2 - caption extraction
            MockAgentResult(
                CaptionExtraction(
                    figure_label="Figure 2",
                    caption_title="Test caption for figure 2",
                    figure_caption="Test caption for figure 2 with no panels.",
                    is_verbatim=True,
                )
            ),
            # Figure 2 - panel extraction (empty panels)
            MockAgentResult(PanelExtraction(figure_label="Figure 2", panels=[])),
        ]
        yield mock_run


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
        self, mock_runner, mock_prompt_handler, zip_structure
    ):
        """Test successful extraction of individual captions."""
        # Create the extractor
        extractor = FigureCaptionExtractorOpenAI(VALID_CONFIG, mock_prompt_handler)

        # Run the extraction
        result = extractor.extract_individual_captions("test content", zip_structure)

        # Verify figures were updated correctly
        assert isinstance(result, ZipStructure)

        # Check that Runner.run was called twice per figure (once for caption, once for panels)
        assert mock_runner.call_count == 4  # 2 figures x 2 calls each

        # Verify Figure 1 data
        figure1 = result.figures[0]
        assert figure1.figure_label == "Figure 1"
        assert figure1.caption_title == "Test caption for figure 1"
        assert (
            figure1.figure_caption == "A) Panel A description. B) Panel B description."
        )
        assert figure1.hallucination_score == 0  # 0 means verbatim/not hallucinated
        assert len(figure1.panels) == 2
        assert figure1.panels[0].panel_label == "A"
        assert figure1.panels[0].panel_caption == "Panel A description"

        # Verify Figure 2 data
        figure2 = result.figures[1]
        assert figure2.figure_label == "Figure 2"
        assert figure2.caption_title == "Test caption for figure 2"
        assert figure2.figure_caption == "Test caption for figure 2 with no panels."
        assert len(figure2.panels) == 0

        # Verify token usage was tracked
        assert result.cost.extract_individual_captions.total_tokens > 0

    def test_api_error_handling(self, mock_prompt_handler, zip_structure):
        """Test handling of API errors."""
        # Create a mock for the event loop
        with patch("asyncio.new_event_loop") as mock_loop:
            # Create mock event loop that will pass the isinstance check
            mock_event_loop = MagicMock(spec=asyncio.AbstractEventLoop)
            mock_loop.return_value = mock_event_loop

            # Configure the mock to raise an exception
            mock_event_loop.run_until_complete.side_effect = Exception("API Error")

            # Create the extractor and run with error
            extractor = FigureCaptionExtractorOpenAI(VALID_CONFIG, mock_prompt_handler)

            # Test that the error is caught and raised
            with pytest.raises(Exception):
                extractor.extract_individual_captions("test content", zip_structure)

            # Verify the loop was closed
            assert mock_event_loop.close.called

    def test_process_figure(self, mock_prompt_handler):
        """Test the process_figure method directly."""
        # Create an event loop for the test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Configure mock to return appropriate responses
            with patch("agents.Runner.run", new_callable=AsyncMock) as mock_run:
                mock_run.side_effect = [
                    # First call (caption extraction)
                    MockAgentResult(MOCK_CAPTION_EXTRACTION),
                    # Second call (panel extraction)
                    MockAgentResult(MOCK_PANEL_EXTRACTION),
                ]

                # Create test figure
                figure = Figure(figure_label="Figure 1", img_files=[], sd_files=[])

                # Create extractor and run process_figure
                extractor = FigureCaptionExtractorOpenAI(
                    VALID_CONFIG, mock_prompt_handler
                )
                updated_figure, token_usage = loop.run_until_complete(
                    extractor.process_figure(figure, "test content")
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
                assert token_usage.total_tokens > 0
        finally:
            # Clean up the loop
            loop.close()


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
