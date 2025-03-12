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
        self,
        mock_config,
        mock_prompt_handler,
        sample_zip_structure,
        mock_image,
        tmp_path,
    ):
        """Test that existing panel data is preserved when processing figures."""
        manuscript_dir = tmp_path / "TEST-ID"
        manuscript_dir.mkdir(parents=True)

        # Create the figure file
        figure_path = manuscript_dir / "figure1.png"
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        figure_path.touch()

        class TestMatchPanelCaption(MatchPanelCaption):
            def _validate_config(self):
                pass

            def _match_panel_caption(self, panel_image, figure_caption):
                return PanelObject(panel_label="A", panel_caption="New caption A")

            def process_figure(self, figure):
                """Process a single figure."""
                # Override to avoid file system operations
                figure.panels = [
                    Panel(
                        panel_label="A",
                        panel_caption="New caption A",
                        panel_bbox=[0.1, 0.1, 0.3, 0.3],
                        confidence=0.9,
                        sd_files=[],
                        ai_response={},
                    )
                ]

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
            matcher = TestMatchPanelCaption(
                mock_config, mock_prompt_handler, extract_dir=manuscript_dir
            )
            result = matcher.process_figures(sample_zip_structure)

            # Verify data preservation
            processed_panel = result.figures[0].panels[0]
            original_panel = sample_zip_structure.figures[0].panels[0]

            assert processed_panel.panel_label == "A"
            assert processed_panel.sd_files == original_panel.sd_files  # Preserved
            assert (
                processed_panel.ai_response == original_panel.ai_response
            )  # Preserved

    def test_unmatched_panels_preservation(
        self,
        mock_config,
        mock_prompt_handler,
        sample_zip_structure,
        mock_image,
        tmp_path,
    ):
        """Test that unmatched panels from original data are preserved."""
        manuscript_dir = tmp_path / "TEST-ID"
        manuscript_dir.mkdir(parents=True)

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

            matcher = TestMatchPanelCaption(
                mock_config, mock_prompt_handler, extract_dir=manuscript_dir
            )
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

    def test_process_figure(
        self, mock_config, mock_prompt_handler, mock_image, tmp_path
    ):
        """Test processing a single figure with correct directory structure."""

        class MockMatchPanelCaption(MatchPanelCaption):
            def _validate_config(self):
                pass

            def _match_panel_caption(self, panel_image, figure_caption):
                return PanelObject(panel_label="A", panel_caption="Test caption")

            def process_figure(self, figure):
                """Process a single figure."""
                panel_obj = self._match_panel_caption(
                    "dummy_image", figure.figure_caption
                )
                figure.panels.append(
                    Panel(
                        panel_label=panel_obj.panel_label,
                        panel_caption=panel_obj.panel_caption,
                        panel_bbox=[0.1, 0.1, 0.3, 0.3],
                        confidence=0.9,
                        sd_files=[],
                        ai_response={},
                    )
                )

        # Create test figure file in manuscript-specific structure
        manuscript_dir = tmp_path / "TEST-ID"
        manuscript_dir.mkdir()
        figure_path = manuscript_dir / "graphic/figure1.tif"
        figure_path.parent.mkdir(parents=True)
        figure_path.touch()

        # Initialize matcher with manuscript directory
        matcher = MockMatchPanelCaption(
            config=mock_config,
            prompt_handler=mock_prompt_handler,
            extract_dir=manuscript_dir,
        )

        # Create test figure
        figure = Figure(
            figure_label="Figure 1",
            img_files=["graphic/figure1.tif"],
            figure_caption="Test caption",
            panels=[],
            sd_files=[],  # Add required sd_files parameter
        )

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

            # Process figure
            matcher.process_figure(figure)

            # Verify file exists and was processed
            assert figure_path.exists()
            assert len(figure.panels) == 1
            assert figure.panels[0].panel_label == "A"
            assert figure.panels[0].panel_caption == "Test caption"

    def test_resolve_panel_conflicts(
        self,
        mock_config,
        mock_prompt_handler,
        mock_image,
        sample_zip_structure,
        tmp_path,
    ):
        """Test that panel conflicts are properly resolved."""
        manuscript_dir = tmp_path / "TEST-ID"
        manuscript_dir.mkdir(parents=True)

        # Create a test implementation that returns predefined responses
        class TestMatchPanelCaption(MatchPanelCaption):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.match_count = 0

            def _validate_config(self):
                pass

            def _match_panel_caption(self, panel_image, figure_caption):
                # Return the same panel label "A" for first two panels
                # to simulate conflict
                if self.match_count < 2:
                    self.match_count += 1
                    return PanelObject(
                        panel_label="A",
                        panel_caption=f"Panel A caption {self.match_count}",
                    )
                else:
                    return PanelObject(panel_label="B", panel_caption="Panel B caption")

        # Create test figure and add original panel info
        figure = Figure(
            figure_label="Figure 1",
            img_files=["test.png"],
            sd_files=[],
            panels=[
                Panel(
                    panel_label="A",
                    panel_caption="Original Panel A",
                    panel_bbox=[0.1, 0.1, 0.2, 0.2],
                    sd_files=["data_A.csv"],
                    ai_response="Original response A",
                ),
                Panel(
                    panel_label="B",
                    panel_caption="Original Panel B",
                    panel_bbox=[0.5, 0.5, 0.6, 0.6],
                    sd_files=["data_B.csv"],
                    ai_response="Original response B",
                ),
            ],
        )

        zip_structure = ZipStructure(figures=[figure])

        # Create a test image file
        img_path = manuscript_dir / "test.png"
        img_path.parent.mkdir(parents=True, exist_ok=True)
        with open(img_path, "wb") as f:
            mock_image.save(f, format="PNG")

        with patch(
            "src.soda_curation.pipeline.match_caption_panel.match_caption_panel_base.convert_to_pil_image"
        ) as mock_convert, patch(
            "src.soda_curation.pipeline.match_caption_panel.match_caption_panel_base.create_object_detection"
        ) as mock_create_detector:
            # Setup mocks
            mock_convert.return_value = (mock_image, "test.png")
            mock_detector = Mock()
            # Create three detected panels to force conflict
            mock_detector.detect_panels.return_value = [
                {
                    "bbox": [0.1, 0.1, 0.2, 0.2],
                    "confidence": 0.9,
                },  # Should match original panel A
                {
                    "bbox": [0.3, 0.3, 0.4, 0.4],
                    "confidence": 0.8,
                },  # Also assigned label A (conflict)
                {
                    "bbox": [0.5, 0.5, 0.6, 0.6],
                    "confidence": 0.7,
                },  # Should match original panel B
            ]
            mock_create_detector.return_value = mock_detector

            # Process figures
            matcher = TestMatchPanelCaption(
                mock_config, mock_prompt_handler, extract_dir=manuscript_dir
            )
            result = matcher.process_figures(zip_structure)

            # Check results
            assert len(result.figures[0].panels) == 2  # Should have 2 panels, not 3

            # Check that panel A was resolved correctly
            panel_a = [p for p in result.figures[0].panels if p.panel_label == "A"][0]
            assert panel_a.panel_bbox == [
                0.1,
                0.1,
                0.2,
                0.2,
            ]  # Should match original position
            assert panel_a.sd_files == ["data_A.csv"]  # Should preserve original data
            assert (
                panel_a.ai_response == "Original response A"
            )  # Should preserve AI response

            # Check that panel B was matched correctly
            panel_b = [p for p in result.figures[0].panels if p.panel_label == "B"][0]
            assert panel_b.panel_bbox == [
                0.5,
                0.5,
                0.6,
                0.6,
            ]  # Should match original position

            # Check that conflicts were tracked
            assert hasattr(result.figures[0], "conflicting_panels")
            assert len(result.figures[0].conflicting_panels) == 1
            assert result.figures[0].conflicting_panels[0]["panel_label"] == "A"

    def test_preserve_original_captions(
        self,
        mock_config,
        mock_prompt_handler,
        mock_image,
        sample_zip_structure,
        tmp_path,
    ):
        """Test that original panel captions are preserved during panel matching."""
        manuscript_dir = tmp_path / "TEST-ID"
        manuscript_dir.mkdir(parents=True)

        # Create a test implementation that returns different captions
        class TestMatchPanelCaption(MatchPanelCaption):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def _validate_config(self):
                pass

            def _match_panel_caption(self, panel_image, figure_caption):
                # Return panel labels with different captions than original
                if "A" in panel_image:  # Mock this check based on encoded image
                    return PanelObject(
                        panel_label="A", panel_caption="NEW Panel A caption"
                    )
                else:
                    return PanelObject(
                        panel_label="B", panel_caption="NEW Panel B caption"
                    )

        # Create test figure and add original panel info with specific captions
        figure = Figure(
            figure_label="Figure 1",
            img_files=["test.png"],
            sd_files=[],
            panels=[
                Panel(
                    panel_label="A",
                    panel_caption="ORIGINAL Panel A caption",  # Original caption to preserve
                    panel_bbox=[0.1, 0.1, 0.2, 0.2],
                    sd_files=["data_A.csv"],
                    ai_response="Original response A",
                ),
                Panel(
                    panel_label="B",
                    panel_caption="ORIGINAL Panel B caption",  # Original caption to preserve
                    panel_bbox=[0.5, 0.5, 0.6, 0.6],
                    sd_files=["data_B.csv"],
                    ai_response="Original response B",
                ),
            ],
        )

        zip_structure = ZipStructure(figures=[figure])

        # Create a test image file
        img_path = manuscript_dir / "test.png"
        img_path.parent.mkdir(parents=True, exist_ok=True)
        with open(img_path, "wb") as f:
            mock_image.save(f, format="PNG")

        with patch(
            "src.soda_curation.pipeline.match_caption_panel.match_caption_panel_base.convert_to_pil_image"
        ) as mock_convert, patch(
            "src.soda_curation.pipeline.match_caption_panel.match_caption_panel_base.create_object_detection"
        ) as mock_create_detector, patch(
            "src.soda_curation.pipeline.match_caption_panel.match_caption_panel_base.MatchPanelCaption._extract_panel_image"
        ) as mock_extract_image:
            # Setup mocks
            mock_convert.return_value = (mock_image, "test.png")
            mock_detector = Mock()
            mock_detector.detect_panels.return_value = [
                {"bbox": [0.1, 0.1, 0.2, 0.2], "confidence": 0.9},  # Panel A
                {"bbox": [0.5, 0.5, 0.6, 0.6], "confidence": 0.8},  # Panel B
            ]
            mock_create_detector.return_value = mock_detector

            # Mock panel image extraction to control what gets passed to _match_panel_caption
            mock_extract_image.side_effect = ["A_encoded_image", "B_encoded_image"]

            # Process figures
            matcher = TestMatchPanelCaption(
                mock_config, mock_prompt_handler, extract_dir=manuscript_dir
            )
            result = matcher.process_figures(zip_structure)

            # Check that original captions were preserved
            panel_a = [p for p in result.figures[0].panels if p.panel_label == "A"][0]
            panel_b = [p for p in result.figures[0].panels if p.panel_label == "B"][0]

            assert panel_a.panel_caption == "ORIGINAL Panel A caption"
            assert panel_b.panel_caption == "ORIGINAL Panel B caption"

            # Verify other properties were updated
            assert panel_a.panel_bbox == [0.1, 0.1, 0.2, 0.2]
            assert panel_b.panel_bbox == [0.5, 0.5, 0.6, 0.6]


class TestMatchPanelCaptionOpenAI:
    """Test the OpenAI implementation of MatchPanelCaption."""

    def test_openai_panel_matching(
        self,
        mock_config,
        mock_prompt_handler,
        mock_image,
        sample_zip_structure,
        tmp_path,
    ):
        """Test OpenAI panel caption matching."""
        manuscript_dir = tmp_path / "TEST-ID"
        manuscript_dir.mkdir(parents=True)

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

            matcher = MatchPanelCaptionOpenAI(
                mock_config, mock_prompt_handler, extract_dir=manuscript_dir
            )
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

    def test_error_handling(self, mock_config, mock_prompt_handler, tmp_path):
        """Test error handling in OpenAI implementation."""
        manuscript_dir = tmp_path / "TEST-ID"
        manuscript_dir.mkdir(parents=True)

        with patch("openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_client.beta.chat.completions.parse.side_effect = Exception("API Error")
            mock_openai.return_value = mock_client

            matcher = MatchPanelCaptionOpenAI(
                mock_config, mock_prompt_handler, extract_dir=manuscript_dir
            )
            result = matcher._match_panel_caption("invalid_image", "Test caption")

            assert isinstance(result, PanelObject)
            assert result.panel_label == ""
            assert result.panel_caption == ""
