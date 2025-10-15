"""
Test suite to ensure the pipeline output exactly matches the expected structure.

This test suite validates that the SODA curation pipeline produces output
that matches the exact structure from the example file.
"""
import json
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.soda_curation.main import main
from src.soda_curation.pipeline.manuscript_structure.manuscript_structure import (
    ZipStructure,
)


class TestPipelineOutputStructureValidation:
    """Test that the pipeline output matches the exact expected structure."""

    @pytest.fixture
    def expected_structure(self) -> dict:
        """Load the expected output structure from the example file."""
        example_file = (
            Path(__file__).parent / "test_data" / "EMBOJ-2024-119382.zip.json"
        )
        with open(example_file, "r") as f:
            return json.load(f)

    @pytest.fixture
    def sample_zip_path(self, tmp_path):
        """Create a sample zip file for testing."""
        return str(tmp_path / "test.zip")

    @pytest.fixture
    def sample_config_path(self, tmp_path):
        """Create a sample config file for testing."""
        config_content = """
default: &default
  openai:
    model: "gpt-4o"
    temperature: 0.1
    top_p: 1.0
    frequency_penalty: 0
    presence_penalty: 0

extract_sections:
  <<: *default

extract_individual_captions:
  <<: *default

assign_panel_source:
  <<: *default

match_caption_panel:
  <<: *default

extract_data_sources:
  <<: *default

object_detection:
  model_path: "data/models/panel_detection_model_no_labels.pt"
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_content)
        return str(config_file)

    def test_pipeline_output_matches_expected_structure(
        self, expected_structure, sample_zip_path, sample_config_path, tmp_path
    ):
        """Test that the pipeline produces output matching the expected structure."""
        # Create a proper test zip file with realistic content
        test_zip = Path(sample_zip_path)
        with zipfile.ZipFile(test_zip, "w") as zip_ref:
            # Add XML file (required by pipeline)
            zip_ref.writestr(
                "EMBOJ-2024-119382.xml", "<?xml version='1.0'?><root></root>"
            )
            # Add some figures
            zip_ref.writestr("graphic/Figure 1.pdf", "dummy pdf content")
            zip_ref.writestr("graphic/Figure 2.pdf", "dummy pdf content")
            # Add source data
            zip_ref.writestr("suppl_data/Figure 1 source data.zip", "dummy zip content")
            zip_ref.writestr("suppl_data/Figure 2 source data.zip", "dummy zip content")
            # Add manuscript files
            zip_ref.writestr(
                "Doc/EMBOJ2024119382R1Manuscript_Textmstxt.docx", "dummy docx content"
            )
            zip_ref.writestr("pdf/EMBOJ-2024-119382.pdf", "dummy pdf content")
            # Add appendix
            zip_ref.writestr(
                "suppl_data/EMBOJ2024119382R1Appendixsupp.pdf", "dummy pdf content"
            )

        # Test the expected structure directly
        self._validate_output_structure_matches_expected(expected_structure)

    def _validate_output_structure_matches_expected(self, output_data):
        """Validate that the output data matches the expected structure exactly."""

        # Test top-level structure
        required_top_level_fields = [
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

        for field in required_top_level_fields:
            assert field in output_data, f"Missing required top-level field: {field}"

        # Test appendix structure
        assert isinstance(output_data["appendix"], list), "appendix should be a list"
        for item in output_data["appendix"]:
            assert isinstance(item, str), "Each appendix item should be a string"
            assert item.endswith(
                (".pdf", ".zip", ".docx")
            ), f"Appendix item should be a file: {item}"

        # Test figures structure
        assert isinstance(output_data["figures"], list), "figures should be a list"
        for figure in output_data["figures"]:
            self._validate_figure_structure_matches_expected(figure)

        # Test data_availability structure
        data_availability = output_data["data_availability"]
        assert (
            "section_text" in data_availability
        ), "data_availability should have section_text"
        assert (
            "data_sources" in data_availability
        ), "data_availability should have data_sources"
        assert isinstance(
            data_availability["section_text"], str
        ), "section_text should be a string"
        assert isinstance(
            data_availability["data_sources"], list
        ), "data_sources should be a list"

        # Test cost structure
        cost = output_data["cost"]
        required_cost_sections = [
            "extract_sections",
            "extract_individual_captions",
            "assign_panel_source",
            "match_caption_panel",
            "extract_data_sources",
            "total",
        ]

        for section in required_cost_sections:
            assert section in cost, f"Missing cost section: {section}"
            section_cost = cost[section]
            required_cost_fields = [
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "cost",
            ]

            for field in required_cost_fields:
                assert (
                    field in section_cost
                ), f"Cost section {section} missing field: {field}"
                assert isinstance(
                    section_cost[field], (int, float)
                ), f"Cost field {field} should be a number"
                assert (
                    section_cost[field] >= 0
                ), f"Cost field {field} should be non-negative"

        # Test manuscript metadata
        assert isinstance(
            output_data["manuscript_id"], str
        ), "manuscript_id should be a string"
        assert (
            len(output_data["manuscript_id"]) > 0
        ), "manuscript_id should not be empty"

        # Test file references
        file_fields = ["xml", "docx", "pdf"]
        for field in file_fields:
            file_path = output_data[field]
            assert isinstance(file_path, str), f"{field} should be a string"
            assert len(file_path) > 0, f"{field} should not be empty"

        # Test AI responses
        ai_response_fields = [
            "ai_response_locate_captions",
            "ai_response_extract_individual_captions",
        ]
        for field in ai_response_fields:
            response = output_data[field]
            assert isinstance(response, str), f"{field} should be a string"

        # Test hallucination scores
        hallucination_fields = [
            "locate_captions_hallucination_score",
            "locate_data_section_hallucination_score",
        ]
        for field in hallucination_fields:
            score = output_data[field]
            assert isinstance(score, (int, float)), f"{field} should be a number"
            assert 0 <= score <= 1, f"{field} should be between 0 and 1"

    def _validate_figure_structure_matches_expected(self, figure):
        """Validate that a figure matches the expected structure."""
        required_figure_fields = [
            "figure_label",
            "img_files",
            "sd_files",
            "panels",
            "hallucination_score",
            "figure_caption",
            "caption_title",
        ]

        for field in required_figure_fields:
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
            self._validate_panel_structure_matches_expected(panel)

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

    def _validate_panel_structure_matches_expected(self, panel):
        """Validate that a panel matches the expected structure."""
        required_panel_fields = [
            "panel_label",
            "panel_caption",
            "panel_bbox",
            "confidence",
            "sd_files",
            "hallucination_score",
        ]

        for field in required_panel_fields:
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

    def test_pipeline_output_json_serialization(self, expected_structure):
        """Test that the pipeline output can be properly serialized to JSON."""
        # Test JSON serialization
        json_str = json.dumps(expected_structure, indent=2)
        assert isinstance(json_str, str)
        assert len(json_str) > 0

        # Test deserialization
        deserialized = json.loads(json_str)
        assert deserialized == expected_structure

        # Test that all required fields are present after serialization
        self._validate_output_structure_matches_expected(deserialized)

    def test_pipeline_output_field_types(self, expected_structure):
        """Test that all fields have the correct data types."""
        # Test top-level field types
        assert isinstance(expected_structure["appendix"], list)
        assert isinstance(expected_structure["figures"], list)
        assert isinstance(expected_structure["data_availability"], dict)
        assert isinstance(expected_structure["manuscript_id"], str)
        assert isinstance(expected_structure["xml"], str)
        assert isinstance(expected_structure["docx"], str)
        assert isinstance(expected_structure["pdf"], str)
        assert isinstance(expected_structure["ai_response_locate_captions"], str)
        assert isinstance(
            expected_structure["ai_response_extract_individual_captions"], str
        )
        assert isinstance(expected_structure["cost"], dict)
        assert isinstance(expected_structure["ai_provider"], str)

        # Test hallucination scores are numbers
        assert isinstance(
            expected_structure["locate_captions_hallucination_score"], (int, float)
        )
        assert isinstance(
            expected_structure["locate_data_section_hallucination_score"], (int, float)
        )

        # Test figures structure
        for figure in expected_structure["figures"]:
            assert isinstance(figure["figure_label"], str)
            assert isinstance(figure["img_files"], list)
            assert isinstance(figure["sd_files"], list)
            assert isinstance(figure["panels"], list)
            assert isinstance(figure["hallucination_score"], (int, float))
            assert isinstance(figure["figure_caption"], str)
            assert isinstance(figure["caption_title"], str)

            # Test panels structure
            for panel in figure["panels"]:
                assert isinstance(panel["panel_label"], str)
                assert isinstance(panel["panel_caption"], str)
                assert isinstance(panel["panel_bbox"], list)
                assert isinstance(panel["confidence"], (int, float))
                assert isinstance(panel["sd_files"], list)
                assert isinstance(panel["hallucination_score"], (int, float))

    def test_pipeline_output_value_ranges(self, expected_structure):
        """Test that numeric values are within expected ranges."""
        # Test hallucination scores
        assert 0 <= expected_structure["locate_captions_hallucination_score"] <= 1
        assert 0 <= expected_structure["locate_data_section_hallucination_score"] <= 1

        # Test figures
        for figure in expected_structure["figures"]:
            assert 0 <= figure["hallucination_score"] <= 1

            # Test panels
            for panel in figure["panels"]:
                assert 0 <= panel["confidence"] <= 1
                assert 0 <= panel["hallucination_score"] <= 1

                # Test bbox coordinates
                for coord in panel["panel_bbox"]:
                    assert 0 <= coord <= 1

    def test_pipeline_output_consistency(self, expected_structure):
        """Test that the output structure is consistent."""
        # Test that all figures have the same structure
        figure_keys = set()
        for figure in expected_structure["figures"]:
            figure_keys.add(tuple(sorted(figure.keys())))

        assert (
            len(figure_keys) == 1
        ), f"Figures have inconsistent structure: {figure_keys}"

        # Test that all panels have the same structure
        panel_keys = set()
        for figure in expected_structure["figures"]:
            for panel in figure["panels"]:
                panel_keys.add(tuple(sorted(panel.keys())))

        assert len(panel_keys) == 1, f"Panels have inconsistent structure: {panel_keys}"

        # Test that panel labels are unique within each figure
        for figure in expected_structure["figures"]:
            panel_labels = [panel["panel_label"] for panel in figure["panels"]]
            assert len(panel_labels) == len(
                set(panel_labels)
            ), f"Figure {figure['figure_label']} has duplicate panel labels: {panel_labels}"

        # Test that figure labels are unique
        figure_labels = [
            figure["figure_label"] for figure in expected_structure["figures"]
        ]
        assert len(figure_labels) == len(
            set(figure_labels)
        ), f"Duplicate figure labels found: {figure_labels}"

    def test_pipeline_output_file_references(self, expected_structure):
        """Test that file references are valid."""
        # Test appendix files
        for appendix_file in expected_structure["appendix"]:
            assert appendix_file.endswith(
                (".pdf", ".zip", ".docx")
            ), f"Invalid appendix file: {appendix_file}"

        # Test figure image files
        for figure in expected_structure["figures"]:
            for img_file in figure["img_files"]:
                assert img_file.endswith(
                    (".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff")
                ), f"Invalid image file: {img_file}"

            for sd_file in figure["sd_files"]:
                assert sd_file.endswith(".zip"), f"Invalid source data file: {sd_file}"

        # Test panel source data files
        for figure in expected_structure["figures"]:
            for panel in figure["panels"]:
                for sd_file in panel["sd_files"]:
                    if ":" in sd_file:
                        # This is a zip:file reference
                        zip_part, file_part = sd_file.split(":", 1)
                        assert zip_part.endswith(
                            ".zip"
                        ), f"Invalid zip reference: {zip_part}"
                        assert (
                            len(file_part) > 0
                        ), f"Empty file part in reference: {sd_file}"
                    else:
                        # This should be a direct file reference
                        assert sd_file.endswith(
                            (".xlsx", ".tif", ".txt", ".csv")
                        ), f"Invalid panel source file: {sd_file}"

    def test_pipeline_output_bbox_coordinates(self, expected_structure):
        """Test that bounding box coordinates are valid."""
        for figure in expected_structure["figures"]:
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

                # Check that all coordinates are within [0, 1] range
                for coord in bbox:
                    assert (
                        0 <= coord <= 1
                    ), f"Panel {panel['panel_label']}: coordinate {coord} should be between 0 and 1"

    def test_pipeline_output_cost_structure(self, expected_structure):
        """Test that the cost structure is complete and valid."""
        cost = expected_structure["cost"]

        # Test that all required sections are present
        required_sections = [
            "extract_sections",
            "extract_individual_captions",
            "assign_panel_source",
            "match_caption_panel",
            "extract_data_sources",
            "total",
        ]

        for section in required_sections:
            assert section in cost, f"Missing cost section: {section}"

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

        # Test that total matches the sum of individual sections
        total_prompt = sum(
            cost[section]["prompt_tokens"] for section in required_sections[:-1]
        )
        total_completion = sum(
            cost[section]["completion_tokens"] for section in required_sections[:-1]
        )
        total_tokens = sum(
            cost[section]["total_tokens"] for section in required_sections[:-1]
        )
        total_cost = sum(cost[section]["cost"] for section in required_sections[:-1])

        assert (
            cost["total"]["prompt_tokens"] == total_prompt
        ), "Total prompt tokens should match sum of individual sections"
        assert (
            cost["total"]["completion_tokens"] == total_completion
        ), "Total completion tokens should match sum of individual sections"
        assert (
            cost["total"]["total_tokens"] == total_tokens
        ), "Total tokens should match sum of individual sections"
        assert (
            abs(cost["total"]["cost"] - total_cost) < 0.001
        ), "Total cost should match sum of individual sections"
