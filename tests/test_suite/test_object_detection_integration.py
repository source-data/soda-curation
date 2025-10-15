"""
Test object detection integration with the pipeline's data structures.
"""
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.soda_curation.pipeline.manuscript_structure.manuscript_structure import (
    Figure,
    Panel,
    ZipStructure,
)


class TestObjectDetectionIntegration:
    """Test object detection integration with pipeline data structures."""

    def test_panel_structure_validation(self):
        """Test that Panel objects have the correct structure."""
        panel = Panel(
            panel_label="A",
            panel_caption="Test panel caption",
            panel_bbox=[0.1, 0.1, 0.4, 0.4],
            confidence=0.9,
            sd_files=["test.zip"],
            hallucination_score=0.0,
        )

        # Test required fields
        assert panel.panel_label == "A"
        assert panel.panel_caption == "Test panel caption"
        assert panel.panel_bbox == [0.1, 0.1, 0.4, 0.4]
        assert panel.confidence == 0.9
        assert panel.sd_files == ["test.zip"]
        assert panel.hallucination_score == 0.0

        # Test bbox coordinates are valid
        assert len(panel.panel_bbox) == 4
        for coord in panel.panel_bbox:
            assert 0 <= coord <= 1, f"Bbox coordinate {coord} should be between 0 and 1"

    def test_figure_structure_validation(self):
        """Test that Figure objects have the correct structure."""
        panel = Panel(
            panel_label="A",
            panel_caption="Test panel caption",
            panel_bbox=[0.1, 0.1, 0.4, 0.4],
            confidence=0.9,
            sd_files=["test.zip"],
            hallucination_score=0.0,
        )

        figure = Figure(
            figure_label="Figure 1",
            img_files=["test.png"],
            sd_files=["test.zip"],
            panels=[panel],
            hallucination_score=0.0,
            figure_caption="Test figure caption",
            caption_title="Test Figure",
        )

        # Test required fields
        assert figure.figure_label == "Figure 1"
        assert figure.img_files == ["test.png"]
        assert figure.sd_files == ["test.zip"]
        assert len(figure.panels) == 1
        assert figure.panels[0].panel_label == "A"
        assert figure.hallucination_score == 0.0
        assert figure.figure_caption == "Test figure caption"
        assert figure.caption_title == "Test Figure"

    def test_zip_structure_validation(self):
        """Test that ZipStructure objects have the correct structure."""
        panel = Panel(
            panel_label="A",
            panel_caption="Test panel caption",
            panel_bbox=[0.1, 0.1, 0.4, 0.4],
            confidence=0.9,
            sd_files=["test.zip"],
            hallucination_score=0.0,
        )

        figure = Figure(
            figure_label="Figure 1",
            img_files=["test.png"],
            sd_files=["test.zip"],
            panels=[panel],
            hallucination_score=0.0,
            figure_caption="Test figure caption",
            caption_title="Test Figure",
        )

        zip_structure = ZipStructure(
            appendix=[],
            figures=[figure],
            data_availability={"section_text": "Test data", "data_sources": []},
            locate_captions_hallucination_score=0.0,
            locate_data_section_hallucination_score=0.0,
            manuscript_id="TEST-001",
            xml="test.xml",
            docx="test.docx",
            pdf="test.pdf",
            ai_response_locate_captions="Test response",
            ai_response_extract_individual_captions="Test response",
            cost=Mock(),
            ai_provider="openai",
        )

        # Test required fields
        assert zip_structure.manuscript_id == "TEST-001"
        assert len(zip_structure.figures) == 1
        assert zip_structure.figures[0].figure_label == "Figure 1"
        assert zip_structure.ai_provider == "openai"
        assert zip_structure.xml == "test.xml"
        assert zip_structure.docx == "test.docx"
        assert zip_structure.pdf == "test.pdf"

    def test_panel_bbox_coordinates_validation(self):
        """Test that panel bbox coordinates are properly validated."""
        # Test valid coordinates
        valid_bbox = [0.1, 0.2, 0.8, 0.9]
        panel = Panel(
            panel_label="A",
            panel_caption="Test panel",
            panel_bbox=valid_bbox,
            confidence=0.9,
            sd_files=[],
            hallucination_score=0.0,
        )

        assert len(panel.panel_bbox) == 4
        for i, coord in enumerate(panel.panel_bbox):
            assert isinstance(coord, (int, float)), f"Coordinate {i} should be a number"
            assert (
                0 <= coord <= 1
            ), f"Coordinate {i} ({coord}) should be between 0 and 1"

    def test_confidence_score_validation(self):
        """Test that confidence scores are properly validated."""
        # Test valid confidence scores
        for confidence in [0.0, 0.5, 0.9, 1.0]:
            panel = Panel(
                panel_label="A",
                panel_caption="Test panel",
                panel_bbox=[0.1, 0.1, 0.4, 0.4],
                confidence=confidence,
                sd_files=[],
                hallucination_score=0.0,
            )
            assert (
                0 <= panel.confidence <= 1
            ), f"Confidence {confidence} should be between 0 and 1"

    def test_hallucination_score_validation(self):
        """Test that hallucination scores are properly validated."""
        # Test valid hallucination scores
        for score in [0.0, 0.5, 0.9, 1.0]:
            panel = Panel(
                panel_label="A",
                panel_caption="Test panel",
                panel_bbox=[0.1, 0.1, 0.4, 0.4],
                confidence=0.9,
                sd_files=[],
                hallucination_score=score,
            )
            assert (
                0 <= panel.hallucination_score <= 1
            ), f"Hallucination score {score} should be between 0 and 1"
