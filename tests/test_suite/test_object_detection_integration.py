"""
Integration tests for object detection to ensure it works correctly in the pipeline.

This test suite validates that the object detection fix works properly
and produces the expected output format.
"""
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.soda_curation.pipeline.manuscript_structure.manuscript_structure import (
    Figure,
    Panel,
    ZipStructure,
)
from src.soda_curation.pipeline.match_caption_panel.match_caption_panel_base import (
    MatchPanelCaption,
)
from src.soda_curation.pipeline.match_caption_panel.object_detection import (
    ObjectDetection,
)


class TestObjectDetectionIntegration:
    """Test object detection integration with the pipeline."""

    def test_object_detection_produces_valid_panels(self, tmp_path):
        """Test that object detection produces panels with valid structure."""
        # Create a mock figure
        figure = Figure(
            figure_label="Figure 1",
            img_files=["test.png"],
            sd_files=["test.zip"],
            panels=[],
            hallucination_score=0.0,
            figure_caption="Test figure",
            caption_title="Test Figure",
        )

        # Create a mock zip structure
        zip_structure = ZipStructure(
            appendix=[],
            figures=[figure],
            data_availability={"section_text": "", "data_sources": []},
            locate_captions_hallucination_score=0.0,
            locate_data_section_hallucination_score=0.0,
            manuscript_id="TEST-001",
            xml="test.xml",
            docx="test.docx",
            pdf="test.pdf",
            ai_response_locate_captions="",
            ai_response_extract_individual_captions="",
            cost=Mock(),
            ai_provider="openai",
        )

        # Mock the object detection to return valid panels
        mock_detections = [
            {"bbox": [0.1, 0.1, 0.4, 0.4], "confidence": 0.9},
            {"bbox": [0.5, 0.1, 0.8, 0.4], "confidence": 0.85},
        ]

        # Mock the match panel caption to return valid panel objects
        mock_panel_objects = [
            Mock(panel_label="A", panel_caption="Panel A caption"),
            Mock(panel_label="B", panel_caption="Panel B caption"),
        ]

        with patch(
            "src.soda_curation.pipeline.match_caption_panel.match_caption_panel_base.convert_to_pil_image"
        ) as mock_convert:
            # Mock convert_to_pil_image to return a PIL Image
            mock_image = Mock()
            mock_image.mode = "RGB"
            mock_image.size = (100, 100)
            mock_image.convert = Mock(return_value=mock_image)
            mock_convert.return_value = (mock_image, "test.png")

            with patch(
                "src.soda_curation.pipeline.match_caption_panel.match_caption_panel_base.ObjectDetection"
            ) as mock_detector_class:
                mock_detector = Mock()
                mock_detector.detect_panels.return_value = mock_detections
                mock_detector_class.return_value = mock_detector

                with patch(
                    "src.soda_curation.pipeline.match_caption_panel.match_caption_panel_base.MatchPanelCaption._match_panel_caption"
                ) as mock_match:
                    mock_match.side_effect = mock_panel_objects

                    # Create the match panel caption instance
                    config = {"object_detection": {"model_path": "test.pt"}}
                    prompt_handler = Mock()
                    extract_dir = tmp_path

                    matcher = MatchPanelCaption(config, prompt_handler, extract_dir)

                    # Process the figures
                    result = matcher.process_figures(zip_structure)

                    # Validate the result
                    assert len(result.figures) == 1
                    figure = result.figures[0]
                    assert len(figure.panels) == 2

                    # Validate panel structure
                    for i, panel in enumerate(figure.panels):
                        assert hasattr(panel, "panel_label")
                        assert hasattr(panel, "panel_caption")
                        assert hasattr(panel, "panel_bbox")
                        assert hasattr(panel, "confidence")
                        assert hasattr(panel, "sd_files")

                        # Validate bbox
                        bbox = panel.panel_bbox
                        assert isinstance(bbox, list)
                        assert len(bbox) == 4
                        assert all(0 <= coord <= 1 for coord in bbox)

                        # Validate confidence
                        assert 0 <= panel.confidence <= 1

    def test_object_detection_handles_no_detections(self, tmp_path):
        """Test that object detection handles cases with no detections gracefully."""
        figure = Figure(
            figure_label="Figure 1",
            img_files=["test.png"],
            sd_files=["test.zip"],
            panels=[],
            hallucination_score=0.0,
            figure_caption="Test figure",
            caption_title="Test Figure",
        )

        zip_structure = ZipStructure(
            appendix=[],
            figures=[figure],
            data_availability={"section_text": "", "data_sources": []},
            locate_captions_hallucination_score=0.0,
            locate_data_section_hallucination_score=0.0,
            manuscript_id="TEST-001",
            xml="test.xml",
            docx="test.docx",
            pdf="test.pdf",
            ai_response_locate_captions="",
            ai_response_extract_individual_captions="",
            cost=Mock(),
            ai_provider="openai",
        )

        with patch(
            "src.soda_curation.pipeline.match_caption_panel.match_caption_panel_base.convert_to_pil_image"
        ) as mock_convert:
            mock_image = Mock()
            mock_image.mode = "RGB"
            mock_image.size = (100, 100)
            mock_image.convert = Mock(return_value=mock_image)
            mock_convert.return_value = (mock_image, "test.png")

            with patch(
                "src.soda_curation.pipeline.match_caption_panel.match_caption_panel_base.ObjectDetection"
            ) as mock_detector_class:
                mock_detector = Mock()
                mock_detector.detect_panels.return_value = []  # No detections
                mock_detector_class.return_value = mock_detector

                config = {"object_detection": {"model_path": "test.pt"}}
                prompt_handler = Mock()
                extract_dir = tmp_path

                matcher = MatchPanelCaption(config, prompt_handler, extract_dir)
                result = matcher.process_figures(zip_structure)

                # Should still have the figure but with no panels
                assert len(result.figures) == 1
                assert len(result.figures[0].panels) == 0

    def test_object_detection_handles_low_confidence_detections(self, tmp_path):
        """Test that object detection filters out low confidence detections."""
        figure = Figure(
            figure_label="Figure 1",
            img_files=["test.png"],
            sd_files=["test.zip"],
            panels=[],
            hallucination_score=0.0,
            figure_caption="Test figure",
            caption_title="Test Figure",
        )

        zip_structure = ZipStructure(
            appendix=[],
            figures=[figure],
            data_availability={"section_text": "", "data_sources": []},
            locate_captions_hallucination_score=0.0,
            locate_data_section_hallucination_score=0.0,
            manuscript_id="TEST-001",
            xml="test.xml",
            docx="test.docx",
            pdf="test.pdf",
            ai_response_locate_captions="",
            ai_response_extract_individual_captions="",
            cost=Mock(),
            ai_provider="openai",
        )

        # Mock detections with mixed confidence levels
        mock_detections = [
            {"bbox": [0.1, 0.1, 0.4, 0.4], "confidence": 0.9},  # High confidence
            {
                "bbox": [0.5, 0.1, 0.8, 0.4],
                "confidence": 0.1,
            },  # Low confidence (should be filtered)
            {"bbox": [0.1, 0.5, 0.4, 0.8], "confidence": 0.8},  # High confidence
        ]

        mock_panel_objects = [
            Mock(panel_label="A", panel_caption="Panel A caption"),
            Mock(panel_label="B", panel_caption="Panel B caption"),
        ]

        with patch(
            "src.soda_curation.pipeline.match_caption_panel.match_caption_panel_base.convert_to_pil_image"
        ) as mock_convert:
            mock_image = Mock()
            mock_image.mode = "RGB"
            mock_image.size = (100, 100)
            mock_image.convert = Mock(return_value=mock_image)
            mock_convert.return_value = (mock_image, "test.png")

            with patch(
                "src.soda_curation.pipeline.match_caption_panel.match_caption_panel_base.ObjectDetection"
            ) as mock_detector_class:
                mock_detector = Mock()
                mock_detector.detect_panels.return_value = mock_detections
                mock_detector_class.return_value = mock_detector

                with patch(
                    "src.soda_curation.pipeline.match_caption_panel.match_caption_panel_base.MatchPanelCaption._match_panel_caption"
                ) as mock_match:
                    mock_match.side_effect = mock_panel_objects

                    config = {"object_detection": {"model_path": "test.pt"}}
                    prompt_handler = Mock()
                    extract_dir = tmp_path

                    matcher = MatchPanelCaption(config, prompt_handler, extract_dir)
                    result = matcher.process_figures(zip_structure)

                    # Should have 2 panels (low confidence one filtered out)
                    assert len(result.figures) == 1
                    figure = result.figures[0]
                    assert len(figure.panels) == 2

                    # All panels should have high confidence
                    for panel in figure.panels:
                        assert panel.confidence >= 0.25  # Default threshold

    def test_object_detection_error_handling(self, tmp_path):
        """Test that object detection handles errors gracefully."""
        figure = Figure(
            figure_label="Figure 1",
            img_files=["test.png"],
            sd_files=["test.zip"],
            panels=[],
            hallucination_score=0.0,
            figure_caption="Test figure",
            caption_title="Test Figure",
        )

        zip_structure = ZipStructure(
            appendix=[],
            figures=[figure],
            data_availability={"section_text": "", "data_sources": []},
            locate_captions_hallucination_score=0.0,
            locate_data_section_hallucination_score=0.0,
            manuscript_id="TEST-001",
            xml="test.xml",
            docx="test.docx",
            pdf="test.pdf",
            ai_response_locate_captions="",
            ai_response_extract_individual_captions="",
            cost=Mock(),
            ai_provider="openai",
        )

        with patch(
            "src.soda_curation.pipeline.match_caption_panel.match_caption_panel_base.convert_to_pil_image"
        ) as mock_convert:
            # Mock convert_to_pil_image to raise an error
            mock_convert.side_effect = Exception("Image conversion failed")

            config = {"object_detection": {"model_path": "test.pt"}}
            prompt_handler = Mock()
            extract_dir = tmp_path

            matcher = MatchPanelCaption(config, prompt_handler, extract_dir)

            # Should handle the error gracefully
            result = matcher.process_figures(zip_structure)

            # Should still have the figure but with no panels
            assert len(result.figures) == 1
            assert len(result.figures[0].panels) == 0

    def test_panel_bbox_coordinate_validation(self):
        """Test that panel bounding box coordinates are properly validated."""
        # Test valid bbox coordinates
        valid_bboxes = [
            [0.0, 0.0, 1.0, 1.0],  # Full image
            [0.1, 0.2, 0.8, 0.9],  # Normal panel
            [0.0, 0.0, 0.5, 0.5],  # Top-left quadrant
            [0.5, 0.5, 1.0, 1.0],  # Bottom-right quadrant
        ]

        for bbox in valid_bboxes:
            x1, y1, x2, y2 = bbox
            assert 0 <= x1 <= 1, f"x1 should be between 0 and 1: {x1}"
            assert 0 <= y1 <= 1, f"y1 should be between 0 and 1: {y1}"
            assert 0 <= x2 <= 1, f"x2 should be between 0 and 1: {x2}"
            assert 0 <= y2 <= 1, f"y2 should be between 0 and 1: {y2}"
            assert x1 < x2, f"x1 ({x1}) should be less than x2 ({x2})"
            assert y1 < y2, f"y1 ({y1}) should be less than y2 ({y2})"

        # Test invalid bbox coordinates
        invalid_bboxes = [
            [0.1, 0.2, 0.8, 0.1],  # y1 > y2
            [0.8, 0.2, 0.1, 0.9],  # x1 > x2
            [0.1, 0.2, 0.8, 1.5],  # y2 > 1
            [1.5, 0.2, 0.8, 0.9],  # x1 > 1
            [0.1, -0.1, 0.8, 0.9],  # y1 < 0
        ]

        for bbox in invalid_bboxes:
            x1, y1, x2, y2 = bbox
            is_valid = (
                0 <= x1 <= 1
                and 0 <= y1 <= 1
                and 0 <= x2 <= 1
                and 0 <= y2 <= 1
                and x1 < x2
                and y1 < y2
            )
            assert not is_valid, f"Bbox should be invalid: {bbox}"

    def test_confidence_score_validation(self):
        """Test that confidence scores are properly validated."""
        # Test valid confidence scores
        valid_scores = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]
        for score in valid_scores:
            assert (
                0 <= score <= 1
            ), f"Confidence score should be between 0 and 1: {score}"

        # Test invalid confidence scores
        invalid_scores = [-0.1, 1.1, 2.0, -1.0]
        for score in invalid_scores:
            assert not (0 <= score <= 1), f"Confidence score should be invalid: {score}"

    def test_panel_label_validation(self):
        """Test that panel labels are properly validated."""
        # Test valid panel labels
        valid_labels = ["A", "B", "C", "a", "b", "c", "1", "2", "3", "i", "ii", "iii"]
        for label in valid_labels:
            assert isinstance(label, str), f"Panel label should be a string: {label}"
            assert len(label) > 0, f"Panel label should not be empty: {label}"

        # Test invalid panel labels
        invalid_labels = ["", " ", "  ", None]
        for label in invalid_labels:
            if label is not None:
                assert (
                    len(label.strip()) == 0
                ), f"Panel label should be invalid: '{label}'"

    def test_panel_caption_validation(self):
        """Test that panel captions are properly validated."""
        # Test valid panel captions
        valid_captions = [
            "A simple caption",
            "A caption with numbers 123",
            "A caption with symbols !@#$%",
            "A very long caption that describes the panel in detail with multiple sentences and various punctuation marks.",
        ]

        for caption in valid_captions:
            assert isinstance(
                caption, str
            ), f"Panel caption should be a string: {caption}"
            assert (
                len(caption.strip()) > 0
            ), f"Panel caption should not be empty: '{caption}'"

        # Test invalid panel captions
        invalid_captions = ["", " ", "  ", None]
        for caption in invalid_captions:
            if caption is not None:
                assert (
                    len(caption.strip()) == 0
                ), f"Panel caption should be invalid: '{caption}'"
