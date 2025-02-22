"""
Tests for the match_caption_panel modules, focusing on data preservation and enrichment.
"""

import base64
import io
from typing import List
from unittest.mock import Mock, patch

import numpy as np
import pytest
from PIL import Image
from pydantic import BaseModel

from src.soda_curation.pipeline.manuscript_structure.manuscript_structure import (
    Figure,
    Panel,
    ZipStructure,
)
from src.soda_curation.pipeline.match_caption_panel.match_caption_panel_base import (
    MatchPanelCaption,
    PanelObject,
)
from src.soda_curation.pipeline.match_caption_panel.match_caption_panel_openai import (
    MatchPanelCaptionOpenAI,
)


# Add these classes at module level
class Message(BaseModel):
    content: PanelObject


class Choice(BaseModel):
    message: Message


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class MockChatCompletionResponse(BaseModel):
    choices: List[Choice]
    usage: Usage


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    return {
        "extraction_dir": "/tmp",
        "pipeline": {
            "match_caption_panel": {
                "openai": {
                    "model": "gpt-4o",
                    "temperature": 0.3,
                    "top_p": 1.0,
                    "max_tokens": 512,
                }
            }
        },
    }


@pytest.fixture
def mock_prompt_handler():
    """Create a mock prompt handler."""
    handler = Mock()
    handler.get_prompt.return_value = {
        "system": "You are a helpful assistant",
        "user": "Please analyze this image",
    }
    return handler


@pytest.fixture
def sample_zip_structure():
    """Create a sample ZipStructure with existing data."""
    panels = [
        Panel(
            panel_label="A",
            panel_caption="Original caption A",
            panel_bbox=[0.1, 0.1, 0.3, 0.3],
            confidence=0.9,
            sd_files=["sd1.txt"],
            ai_response={"key": "value"},
        ),
        Panel(
            panel_label="B",
            panel_caption="Original caption B",
            panel_bbox=[0.4, 0.1, 0.6, 0.3],
            confidence=0.85,
            sd_files=["sd2.txt"],
            ai_response={"key2": "value2"},
        ),
    ]

    figures = [
        Figure(
            figure_label="Figure 1",
            figure_caption="This is figure 1 with panels A and B",
            img_files=["figure1.png"],
            panels=panels,
            sd_files=["figure1_sd.txt"],  # Added missing required parameter
        )
    ]

    return ZipStructure(figures=figures)


@pytest.fixture
def mock_image():
    """Create a mock PIL Image."""
    img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
    return img


class TestMatchPanelCaptionBase:
    """Test the base MatchPanelCaption class functionality."""

    def test_panel_data_preservation(
        self, mock_config, mock_prompt_handler, sample_zip_structure, mock_image
    ):
        """Test that existing panel data is preserved when processing figures."""

        class TestMatchPanelCaption(MatchPanelCaption):
            def _validate_config(self):
                pass

            def _match_panel_caption(self, panel_image, figure_caption):
                return PanelObject(panel_label="A", panel_caption="New caption A")

        with patch(
            "src.soda_curation.pipeline.match_caption_panel.match_caption_panel_base.convert_to_pil_image"
        ) as mock_convert, patch(
            "src.soda_curation.pipeline.match_caption_panel.match_caption_panel_base.create_object_detection"
        ) as mock_create_detector:
            # Setup mocks
            mock_convert.return_value = (mock_image, "test.png")
            mock_detector = Mock()
            mock_detector.detect_panels.return_value = [
                {"bbox": [0.1, 0.1, 0.3, 0.3], "confidence": 0.9}
            ]
            mock_create_detector.return_value = mock_detector

            # Process figures
            matcher = TestMatchPanelCaption(mock_config, mock_prompt_handler)
            result = matcher.process_figures(sample_zip_structure)

            # Verify data preservation
            processed_panel = result.figures[0].panels[0]
            original_panel = sample_zip_structure.figures[0].panels[0]

            assert processed_panel.panel_label == "A"
            assert processed_panel.panel_caption == "New caption A"  # New data
            assert processed_panel.sd_files == original_panel.sd_files  # Preserved
            assert (
                processed_panel.ai_response == original_panel.ai_response
            )  # Preserved

    def test_unmatched_panels_preservation(
        self, mock_config, mock_prompt_handler, sample_zip_structure, mock_image
    ):
        """Test that unmatched panels from original data are preserved."""

        class TestMatchPanelCaption(MatchPanelCaption):
            def _validate_config(self):
                pass

            def _match_panel_caption(self, panel_image, figure_caption):
                return PanelObject(panel_label="A", panel_caption="New caption A")

        with patch(
            "src.soda_curation.pipeline.match_caption_panel.match_caption_panel_base.convert_to_pil_image"
        ) as mock_convert, patch(
            "src.soda_curation.pipeline.match_caption_panel.match_caption_panel_base.create_object_detection"
        ) as mock_create_detector:
            mock_convert.return_value = (mock_image, "test.png")
            mock_detector = Mock()
            # Only detect one panel
            mock_detector.detect_panels.return_value = [
                {"bbox": [0.1, 0.1, 0.3, 0.3], "confidence": 0.9}
            ]
            mock_create_detector.return_value = mock_detector

            matcher = TestMatchPanelCaption(mock_config, mock_prompt_handler)
            result = matcher.process_figures(sample_zip_structure)

            # Verify both panels are present
            assert len(result.figures[0].panels) == 2
            # Verify panel B is preserved exactly as original
            original_panel_b = sample_zip_structure.figures[0].panels[1]
            preserved_panel_b = [
                p for p in result.figures[0].panels if p.panel_label == "B"
            ][0]
            assert preserved_panel_b.panel_caption == original_panel_b.panel_caption
            assert preserved_panel_b.sd_files == original_panel_b.sd_files
            assert preserved_panel_b.ai_response == original_panel_b.ai_response


class TestMatchPanelCaptionOpenAI:
    """Test the OpenAI implementation of MatchPanelCaption."""

    def test_openai_config_validation(self, mock_config, mock_prompt_handler):
        """Test OpenAI configuration validation."""
        # Test valid config
        matcher = MatchPanelCaptionOpenAI(mock_config, mock_prompt_handler)
        assert matcher.openai_config["model"] == "gpt-4o"

        # Test invalid model
        invalid_config = mock_config.copy()
        invalid_config["pipeline"]["match_caption_panel"]["openai"][
            "model"
        ] = "invalid-model"
        with pytest.raises(ValueError, match="Invalid model"):
            MatchPanelCaptionOpenAI(invalid_config, mock_prompt_handler)

    def test_openai_panel_matching(
        self, mock_config, mock_prompt_handler, mock_image, sample_zip_structure
    ):
        """Test OpenAI panel caption matching."""
        with patch("openai.OpenAI") as mock_openai:
            # Create a PanelObject instance
            panel_obj = PanelObject(
                panel_label="A", panel_caption="AI generated caption"
            )

            # Create a proper ChatCompletion response structure using Pydantic models
            mock_response = MockChatCompletionResponse(
                choices=[Choice(message=Message(content=panel_obj))],
                usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
            )

            # Setup the OpenAI client mock
            mock_client = Mock()
            mock_client.beta.chat.completions.parse.return_value = mock_response
            mock_openai.return_value = mock_client

            matcher = MatchPanelCaptionOpenAI(mock_config, mock_prompt_handler)
            matcher.zip_structure = sample_zip_structure

            # Create base64 encoded image
            img_buffer = io.BytesIO()
            mock_image.save(img_buffer, format="PNG")
            encoded_image = base64.b64encode(img_buffer.getvalue()).decode("utf-8")

            result = matcher._match_panel_caption(encoded_image, "Test figure caption")

            # Verify the API was called with correct parameters
            mock_client.beta.chat.completions.parse.assert_called_once()

            # Verify the result
            assert isinstance(result, PanelObject)
            assert result.panel_label == "A"
            assert result.panel_caption == "AI generated caption"

    def test_error_handling(self, mock_config, mock_prompt_handler):
        """Test error handling in OpenAI implementation."""
        with patch("openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_client.beta.chat.completions.parse.side_effect = Exception("API Error")
            mock_openai.return_value = mock_client

            matcher = MatchPanelCaptionOpenAI(mock_config, mock_prompt_handler)
            result = matcher._match_panel_caption("invalid_image", "Test caption")

            assert isinstance(result, PanelObject)
            assert result.panel_label == ""
            assert result.panel_caption == ""
