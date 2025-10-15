"""
Fast Pipeline Structure Validation Test

This test validates that the pipeline output structure matches the expected format
by testing against a known good output file. This is much faster than running the full pipeline.
"""
import json
from pathlib import Path

import pytest


class TestPipelineStructureValidationFast:
    """Test that the pipeline output structure matches the expected format."""

    @pytest.fixture
    def known_good_output_path(self):
        """Path to a known good pipeline output file."""
        return "data/output/testing-last-outputs.json"

    @pytest.fixture
    def ground_truth_path(self):
        """Path to the ground truth JSON file."""
        return "tests/test_suite/test_data/EMBOJ-2024-119382.zip.json"

    def test_pipeline_output_structure_validation(
        self, known_good_output_path, ground_truth_path
    ):
        """Test that the known good output has the correct structure."""
        # Verify files exist
        assert Path(
            known_good_output_path
        ).exists(), f"Known good output file not found: {known_good_output_path}"
        assert Path(
            ground_truth_path
        ).exists(), f"Ground truth file not found: {ground_truth_path}"

        # Load output data
        with open(known_good_output_path, "r") as f:
            output_data = json.load(f)

        print("=" * 60)
        print("VALIDATING PIPELINE OUTPUT STRUCTURE")
        print("=" * 60)

        # Test essential fields are present
        essential_fields = {
            "figures",
            "data_availability",
            "cost",
            "ai_provider",
            "manuscript_id",
            "locate_captions_hallucination_score",
            "locate_data_section_hallucination_score",
            "xml",
            "docx",
            "pdf",
            "ai_response_locate_captions",
            "ai_response_extract_individual_captions",
        }

        output_fields = set(output_data.keys())
        missing_essential_fields = essential_fields - output_fields

        if missing_essential_fields:
            pytest.fail(
                f"Output missing essential fields: {sorted(missing_essential_fields)}"
            )

        print(f"✅ All essential fields present: {sorted(essential_fields)}")

        # Test figures structure
        assert isinstance(output_data["figures"], list), "figures should be a list"
        assert len(output_data["figures"]) > 0, "should have at least one figure"

        print(f"✅ Figures: {len(output_data['figures'])} figures found")

        # Test each figure has correct structure
        for i, figure in enumerate(output_data["figures"]):
            required_figure_fields = ["figure_label", "img_files", "sd_files", "panels"]
            for field in required_figure_fields:
                assert field in figure, f"Figure {i} missing field: {field}"

            # Test panels structure
            assert isinstance(
                figure["panels"], list
            ), f"Figure {i} panels should be a list"
            assert (
                len(figure["panels"]) > 0
            ), f"Figure {i} should have at least one panel"

            # Test each panel has required fields
            for j, panel in enumerate(figure["panels"]):
                required_panel_fields = [
                    "panel_label",
                    "panel_caption",
                    "panel_bbox",
                    "confidence",
                    "sd_files",
                    "hallucination_score",
                ]
                for field in required_panel_fields:
                    assert (
                        field in panel
                    ), f"Figure {i} panel {j} missing field: {field}"

                # Test panel_bbox is valid
                bbox = panel["panel_bbox"]
                assert isinstance(
                    bbox, list
                ), f"Figure {i} panel {j} bbox should be a list"
                assert (
                    len(bbox) == 4
                ), f"Figure {i} panel {j} bbox should have 4 coordinates"
                for k, coord in enumerate(bbox):
                    assert isinstance(
                        coord, (int, float)
                    ), f"Figure {i} panel {j} bbox coordinate {k} should be a number"
                    assert (
                        0 <= coord <= 1
                    ), f"Figure {i} panel {j} bbox coordinate {k} should be between 0 and 1"

            print(
                f"✅ Figure {i+1}: {figure['figure_label']} - {len(figure['panels'])} panels"
            )

        # Test data_availability structure
        assert (
            "section_text" in output_data["data_availability"]
        ), "data_availability should have section_text"
        assert (
            "data_sources" in output_data["data_availability"]
        ), "data_availability should have data_sources"
        assert isinstance(
            output_data["data_availability"]["data_sources"], list
        ), "data_sources should be a list"

        data_sources = output_data["data_availability"]["data_sources"]
        print(f"✅ Data availability: {len(data_sources)} data sources found")

        # Test data_sources structure
        for i, source in enumerate(data_sources):
            assert isinstance(source, dict), f"Data source {i} should be a dictionary"
            required_source_fields = ["database", "accession_number", "url"]
            for field in required_source_fields:
                assert field in source, f"Data source {i} missing field: {field}"
                assert isinstance(
                    source[field], str
                ), f"Data source {i} field {field} should be a string"
                assert (
                    len(source[field]) > 0
                ), f"Data source {i} field {field} should not be empty"

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
            assert section in cost, f"Cost missing section: {section}"
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

        print("✅ Cost structure validation passed")

        # Test JSON serialization
        json_str = json.dumps(output_data, indent=2)
        assert isinstance(json_str, str)
        assert len(json_str) > 0

        # Test deserialization
        deserialized = json.loads(json_str)
        assert deserialized == output_data

        print("✅ JSON serialization validation passed")

        print("=" * 60)
        print("🎉 ALL STRUCTURE VALIDATIONS PASSED!")
        print("✅ Pipeline output structure is correct")
        print("✅ All essential fields are present")
        print("✅ Figures and panels have correct structure")
        print("✅ Data availability format is correct")
        print("✅ Cost tracking is complete")
        print("✅ JSON serialization works correctly")
        print("=" * 60)

    def test_ground_truth_file_structure(self, ground_truth_path):
        """Test that the ground truth file has the expected structure."""
        with open(ground_truth_path, "r") as f:
            ground_truth = json.load(f)

        # Test essential fields
        essential_fields = {
            "figures",
            "data_availability",
            "cost",
            "ai_provider",
            "manuscript_id",
            "locate_captions_hallucination_score",
            "locate_data_section_hallucination_score",
            "xml",
            "docx",
            "pdf",
            "ai_response_locate_captions",
            "ai_response_extract_individual_captions",
        }

        for field in essential_fields:
            assert field in ground_truth, f"Ground truth missing field: {field}"

        print("✅ Ground truth file validation passed")
        print(f"   - {len(ground_truth['figures'])} figures")
        print(
            f"   - {sum(len(fig['panels']) for fig in ground_truth['figures'])} total panels"
        )
        print(f"   - Manuscript ID: {ground_truth['manuscript_id']}")

    def test_output_consistency(self, known_good_output_path):
        """Test that the output is consistent and well-formed."""
        with open(known_good_output_path, "r") as f:
            output_data = json.load(f)

        # Test that all figures have unique labels
        figure_labels = [figure["figure_label"] for figure in output_data["figures"]]
        assert len(figure_labels) == len(
            set(figure_labels)
        ), f"Duplicate figure labels found: {figure_labels}"

        # Test that panel labels are unique within each figure
        for i, figure in enumerate(output_data["figures"]):
            panel_labels = [panel["panel_label"] for panel in figure["panels"]]
            assert len(panel_labels) == len(
                set(panel_labels)
            ), f"Figure {i} has duplicate panel labels: {panel_labels}"

        # Test that all numeric values are reasonable
        for figure in output_data["figures"]:
            assert (
                0 <= figure.get("hallucination_score", 0) <= 1
            ), "Figure hallucination score should be between 0 and 1"

            for panel in figure["panels"]:
                assert (
                    0 <= panel.get("confidence", 0) <= 1
                ), "Panel confidence should be between 0 and 1"
                assert (
                    0 <= panel.get("hallucination_score", 0) <= 1
                ), "Panel hallucination score should be between 0 and 1"

        print("✅ Output consistency validation passed")
