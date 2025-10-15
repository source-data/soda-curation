"""
Test suite to validate the output format of the SODA curation pipeline.

This test suite ensures that the pipeline produces the expected JSON structure
with all required fields and proper data types, helping catch issues early.
"""
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from src.soda_curation.pipeline.manuscript_structure.manuscript_structure import (
    Figure,
    Panel,
    ZipStructure,
)


class TestOutputFormatValidation:
    """Test that the pipeline output matches the expected format."""

    @pytest.fixture
    def expected_output_structure(self) -> Dict[str, Any]:
        """Load the expected output structure from the example file."""
        example_file = (
            Path(__file__).parent / "test_data" / "EMBOJ-2024-119382.zip.json"
        )
        if not example_file.exists():
            pytest.skip("Example output file not found")

        with open(example_file, "r") as f:
            return json.load(f)

    def test_top_level_structure(self, expected_output_structure):
        """Test that the top-level structure contains all required fields."""
        required_fields = [
            "appendix",
            "figures",
            "data_availability",
            "locate_captions_hallucination_score",
            "locate_data_section_hallucination_score",
            "manuscript_id",
            "xml",
            "docx",
            "pdf",
            "ai_response_locate_captions",
            "ai_response_extract_individual_captions",
            "cost",
            "ai_provider",
        ]

        for field in required_fields:
            assert (
                field in expected_output_structure
            ), f"Missing required field: {field}"

    def test_appendix_structure(self, expected_output_structure):
        """Test that appendix is a list of strings."""
        appendix = expected_output_structure["appendix"]
        assert isinstance(appendix, list), "Appendix should be a list"

        for item in appendix:
            assert isinstance(item, str), "Each appendix item should be a string"
            assert item.endswith(
                (".pdf", ".zip", ".docx")
            ), f"Appendix item should be a file: {item}"

    def test_figures_structure(self, expected_output_structure):
        """Test that figures have the correct structure."""
        figures = expected_output_structure["figures"]
        assert isinstance(figures, list), "Figures should be a list"

        for figure in figures:
            self._validate_figure_structure(figure)

    def _validate_figure_structure(self, figure: Dict[str, Any]):
        """Validate the structure of a single figure."""
        required_fields = [
            "figure_label",
            "img_files",
            "sd_files",
            "panels",
            "hallucination_score",
            "figure_caption",
            "caption_title",
        ]

        for field in required_fields:
            assert field in figure, f"Figure missing required field: {field}"

        # Validate figure_label
        assert isinstance(
            figure["figure_label"], str
        ), "figure_label should be a string"
        assert figure["figure_label"].startswith(
            "Figure "
        ), "figure_label should start with 'Figure '"

        # Validate img_files
        assert isinstance(figure["img_files"], list), "img_files should be a list"
        for img_file in figure["img_files"]:
            assert isinstance(img_file, str), "Each img_file should be a string"
            assert img_file.endswith(
                (".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff")
            ), f"img_file should be an image: {img_file}"

        # Validate sd_files
        assert isinstance(figure["sd_files"], list), "sd_files should be a list"
        for sd_file in figure["sd_files"]:
            assert isinstance(sd_file, str), "Each sd_file should be a string"
            assert sd_file.endswith(".zip"), f"sd_file should be a zip: {sd_file}"

        # Validate panels
        assert isinstance(figure["panels"], list), "panels should be a list"
        for panel in figure["panels"]:
            self._validate_panel_structure(panel)

        # Validate hallucination_score
        assert isinstance(
            figure["hallucination_score"], (int, float)
        ), "hallucination_score should be a number"
        assert (
            0 <= figure["hallucination_score"] <= 1
        ), "hallucination_score should be between 0 and 1"

        # Validate captions
        assert isinstance(
            figure["figure_caption"], str
        ), "figure_caption should be a string"
        assert isinstance(
            figure["caption_title"], str
        ), "caption_title should be a string"

    def _validate_panel_structure(self, panel: Dict[str, Any]):
        """Validate the structure of a single panel."""
        required_fields = [
            "panel_label",
            "panel_caption",
            "panel_bbox",
            "confidence",
            "sd_files",
            "hallucination_score",
        ]

        for field in required_fields:
            assert field in panel, f"Panel missing required field: {field}"

        # Validate panel_label
        assert isinstance(panel["panel_label"], str), "panel_label should be a string"
        assert len(panel["panel_label"]) > 0, "panel_label should not be empty"

        # Validate panel_caption
        assert isinstance(
            panel["panel_caption"], str
        ), "panel_caption should be a string"
        assert len(panel["panel_caption"]) > 0, "panel_caption should not be empty"

        # Validate panel_bbox
        bbox = panel["panel_bbox"]
        assert isinstance(bbox, list), "panel_bbox should be a list"
        assert len(bbox) == 4, "panel_bbox should have exactly 4 coordinates"
        for coord in bbox:
            assert isinstance(
                coord, (int, float)
            ), "Each bbox coordinate should be a number"
            assert (
                0 <= coord <= 1
            ), f"Bbox coordinate should be between 0 and 1: {coord}"

        # Validate confidence
        confidence = panel["confidence"]
        assert isinstance(confidence, (int, float)), "confidence should be a number"
        assert (
            0 <= confidence <= 1
        ), f"confidence should be between 0 and 1: {confidence}"

        # Validate sd_files
        assert isinstance(panel["sd_files"], list), "sd_files should be a list"
        for sd_file in panel["sd_files"]:
            assert isinstance(sd_file, str), "Each sd_file should be a string"

        # Validate hallucination_score
        hallucination_score = panel["hallucination_score"]
        assert isinstance(
            hallucination_score, (int, float)
        ), "hallucination_score should be a number"
        assert (
            0 <= hallucination_score <= 1
        ), f"hallucination_score should be between 0 and 1: {hallucination_score}"

    def test_data_availability_structure(self, expected_output_structure):
        """Test that data_availability has the correct structure."""
        data_availability = expected_output_structure["data_availability"]

        required_fields = ["section_text", "data_sources"]
        for field in required_fields:
            assert (
                field in data_availability
            ), f"data_availability missing field: {field}"

        # Validate section_text
        assert isinstance(
            data_availability["section_text"], str
        ), "section_text should be a string"

        # Validate data_sources
        assert isinstance(
            data_availability["data_sources"], list
        ), "data_sources should be a list"
        for source in data_availability["data_sources"]:
            assert isinstance(source, dict), "Each data source should be a dictionary"

    def test_cost_structure(self, expected_output_structure):
        """Test that cost tracking has the correct structure."""
        cost = expected_output_structure["cost"]

        required_sections = [
            "extract_sections",
            "extract_individual_captions",
            "assign_panel_source",
            "match_caption_panel",
            "extract_data_sources",
            "total",
        ]

        for section in required_sections:
            assert section in cost, f"Cost missing section: {section}"

            section_cost = cost[section]
            required_fields = [
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "cost",
            ]

            for field in required_fields:
                assert (
                    field in section_cost
                ), f"Cost section {section} missing field: {field}"
                assert isinstance(
                    section_cost[field], (int, float)
                ), f"Cost field {field} should be a number"
                assert (
                    section_cost[field] >= 0
                ), f"Cost field {field} should be non-negative"

    def test_manuscript_metadata(self, expected_output_structure):
        """Test that manuscript metadata fields are correct."""
        # Validate manuscript_id
        manuscript_id = expected_output_structure["manuscript_id"]
        assert isinstance(manuscript_id, str), "manuscript_id should be a string"
        assert len(manuscript_id) > 0, "manuscript_id should not be empty"

        # Validate file references
        file_fields = ["xml", "docx", "pdf"]
        for field in file_fields:
            file_path = expected_output_structure[field]
            assert isinstance(file_path, str), f"{field} should be a string"
            assert file_path.endswith(
                f'.{field.split("_")[0]}'
            ), f"{field} should have correct extension"

    def test_ai_responses(self, expected_output_structure):
        """Test that AI response fields are strings."""
        ai_response_fields = [
            "ai_response_locate_captions",
            "ai_response_extract_individual_captions",
        ]

        for field in ai_response_fields:
            response = expected_output_structure[field]
            assert isinstance(response, str), f"{field} should be a string"

    def test_hallucination_scores(self, expected_output_structure):
        """Test that hallucination scores are valid numbers."""
        hallucination_fields = [
            "locate_captions_hallucination_score",
            "locate_data_section_hallucination_score",
        ]

        for field in hallucination_fields:
            score = expected_output_structure[field]
            assert isinstance(score, (int, float)), f"{field} should be a number"
            assert 0 <= score <= 1, f"{field} should be between 0 and 1"

    def test_panel_bbox_coordinates(self, expected_output_structure):
        """Test that panel bounding box coordinates are valid."""
        for figure in expected_output_structure["figures"]:
            for panel in figure["panels"]:
                bbox = panel["panel_bbox"]

                # Check that coordinates are in correct order: [x1, y1, x2, y2]
                x1, y1, x2, y2 = bbox
                assert (
                    x1 < x2
                ), f"Panel {panel['panel_label']}: x1 ({x1}) should be less than x2 ({x2})"
                assert (
                    y1 < y2
                ), f"Panel {panel['panel_label']}: y1 ({y1}) should be less than y2 ({y2})"

    def test_panel_labels_are_unique_per_figure(self, expected_output_structure):
        """Test that panel labels are unique within each figure."""
        for figure in expected_output_structure["figures"]:
            panel_labels = [panel["panel_label"] for panel in figure["panels"]]
            assert len(panel_labels) == len(
                set(panel_labels)
            ), f"Figure {figure['figure_label']} has duplicate panel labels: {panel_labels}"

    def test_figure_labels_are_unique(self, expected_output_structure):
        """Test that figure labels are unique."""
        figure_labels = [
            figure["figure_label"] for figure in expected_output_structure["figures"]
        ]
        assert len(figure_labels) == len(
            set(figure_labels)
        ), f"Duplicate figure labels found: {figure_labels}"

    def test_confidence_scores_are_reasonable(self, expected_output_structure):
        """Test that confidence scores are within reasonable ranges."""
        for figure in expected_output_structure["figures"]:
            for panel in figure["panels"]:
                confidence = panel["confidence"]
                # Most panels should have confidence > 0.5
                assert (
                    confidence > 0.1
                ), f"Panel {panel['panel_label']} has very low confidence: {confidence}"

    def test_sd_files_references_are_valid(self, expected_output_structure):
        """Test that source data file references are valid."""
        for figure in expected_output_structure["figures"]:
            for panel in figure["panels"]:
                for sd_file in panel["sd_files"]:
                    # Check that sd_file references follow expected pattern
                    if ":" in sd_file:
                        # This is a zip:file reference
                        zip_part, file_part = sd_file.split(":", 1)
                        assert zip_part.endswith(
                            ".zip"
                        ), f"SD file zip part should end with .zip: {zip_part}"
                        assert (
                            len(file_part) > 0
                        ), f"SD file should have a file part: {sd_file}"
                    else:
                        # This should be a direct file reference
                        assert sd_file.endswith(
                            (".xlsx", ".tif", ".txt", ".csv")
                        ), f"SD file should have valid extension: {sd_file}"

    def test_output_is_json_serializable(self, expected_output_structure):
        """Test that the output can be serialized to JSON."""
        # This should not raise an exception
        json_str = json.dumps(expected_output_structure, indent=2)
        assert isinstance(json_str, str)
        assert len(json_str) > 0

        # Test that it can be deserialized back
        deserialized = json.loads(json_str)
        assert deserialized == expected_output_structure

    def test_zip_structure_compatibility(self):
        """Test that the expected structure is compatible with ZipStructure model."""
        # This test ensures our expected format matches what ZipStructure expects
        example_file = (
            Path(__file__).parent / "test_data" / "EMBOJ-2024-119382.zip.json"
        )
        if not example_file.exists():
            pytest.skip("Example output file not found")

        with open(example_file, "r") as f:
            data = json.load(f)

        # Test that we can create a ZipStructure from this data
        # (This would require implementing a from_dict method in ZipStructure)
        # For now, we just validate the structure matches what ZipStructure expects

        # Validate figures structure
        for figure_data in data["figures"]:
            assert "figure_label" in figure_data
            assert "img_files" in figure_data
            assert "panels" in figure_data

            # Validate panels structure
            for panel_data in figure_data["panels"]:
                assert "panel_label" in panel_data
                assert "panel_caption" in panel_data
                assert "panel_bbox" in panel_data
                assert "confidence" in panel_data


class TestOutputFormatRegression:
    """Regression tests to ensure output format doesn't break."""

    def test_no_empty_strings_in_required_fields(self):
        """Test that required string fields are not empty."""
        example_file = (
            Path(__file__).parent / "test_data" / "EMBOJ-2024-119382.zip.json"
        )
        if not example_file.exists():
            pytest.skip("Example output file not found")

        with open(example_file, "r") as f:
            data = json.load(f)

        # Check that string fields are not empty
        string_fields_to_check = ["manuscript_id", "xml", "docx", "pdf"]

        for field in string_fields_to_check:
            value = data[field]
            assert value.strip() != "", f"Field {field} should not be empty"

        # Check figure fields
        for figure in data["figures"]:
            assert (
                figure["figure_label"].strip() != ""
            ), "figure_label should not be empty"
            assert (
                figure["figure_caption"].strip() != ""
            ), "figure_caption should not be empty"
            assert (
                figure["caption_title"].strip() != ""
            ), "caption_title should not be empty"

            # Check panel fields
            for panel in figure["panels"]:
                assert (
                    panel["panel_label"].strip() != ""
                ), "panel_label should not be empty"
                assert (
                    panel["panel_caption"].strip() != ""
                ), "panel_caption should not be empty"

    def test_consistent_data_types(self):
        """Test that data types are consistent throughout the output."""
        example_file = (
            Path(__file__).parent / "test_data" / "EMBOJ-2024-119382.zip.json"
        )
        if not example_file.exists():
            pytest.skip("Example output file not found")

        with open(example_file, "r") as f:
            data = json.load(f)

        # All figures should have the same structure
        figure_keys = set()
        for figure in data["figures"]:
            figure_keys.add(tuple(sorted(figure.keys())))

        assert (
            len(figure_keys) == 1
        ), f"Figures have inconsistent structure: {figure_keys}"

        # All panels should have the same structure
        panel_keys = set()
        for figure in data["figures"]:
            for panel in figure["panels"]:
                panel_keys.add(tuple(sorted(panel.keys())))

        assert len(panel_keys) == 1, f"Panels have inconsistent structure: {panel_keys}"
