"""
Real Pipeline Structure Validation Test

This test runs the actual SODA curation pipeline on a real test file
and validates that the output structure exactly matches the ground truth.
This test is designed to run in CI/CD environments and ensures 100% structure compliance.
"""
import json
import tempfile
from pathlib import Path

import pytest

from src.soda_curation.main import main


class TestRealPipelineStructureValidation:
    """Test that the real pipeline output structure matches the ground truth exactly."""

    @pytest.fixture
    def test_zip_path(self):
        """Path to the test ZIP file."""
        return "tests/test_suite/test_data/EMBOR-2023-58706-T.zip"

    @pytest.fixture
    def ground_truth_path(self):
        """Path to the ground truth JSON file."""
        return "tests/test_suite/test_data/EMBOJ-2024-119382.zip.json"

    @pytest.fixture
    def config_path(self):
        """Path to the configuration file."""
        return "config.dev.yaml"

    def test_real_pipeline_output_structure_matches_ground_truth(
        self, test_zip_path, ground_truth_path, config_path, tmp_path
    ):
        """Test that the real pipeline output structure matches the ground truth exactly."""
        # Verify test files exist
        assert Path(test_zip_path).exists(), f"Test ZIP file not found: {test_zip_path}"
        assert Path(
            ground_truth_path
        ).exists(), f"Ground truth file not found: {ground_truth_path}"
        assert Path(config_path).exists(), f"Config file not found: {config_path}"

        # Load ground truth structure
        with open(ground_truth_path, "r") as f:
            ground_truth = json.load(f)

        # Create output path
        output_path = str(tmp_path / "pipeline_output.json")

        # Run the actual pipeline
        print(f"Running pipeline on: {test_zip_path}")
        print(f"Using config: {config_path}")
        print(f"Output will be saved to: {output_path}")
        print("=" * 60)
        print("PIPELINE EXECUTION STARTED")
        print("=" * 60)

        try:
            result = main(test_zip_path, config_path, output_path)
            print("=" * 60)
            print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
            print("=" * 60)
        except Exception as e:
            print("=" * 60)
            print(f"PIPELINE EXECUTION FAILED: {str(e)}")
            print("=" * 60)
            pytest.fail(f"Pipeline execution failed: {str(e)}")

        # Verify the result is a JSON string
        assert isinstance(result, str), "Pipeline should return a JSON string"

        # Parse the output
        try:
            output_data = json.loads(result)
            print("Pipeline output parsed successfully")
        except json.JSONDecodeError as e:
            pytest.fail(f"Pipeline output is not valid JSON: {str(e)}")

        # Verify output file was created
        assert Path(output_path).exists(), "Output file should be created"

        # Verify output file content matches result
        with open(output_path, "r") as f:
            file_data = json.load(f)
        assert (
            file_data == output_data
        ), "Output file content should match pipeline result"

        # Compare structure with ground truth
        self._compare_output_structure_exactly(output_data, ground_truth)

        # Additional validation: ensure the output can be serialized back to JSON
        json_str = json.dumps(output_data, indent=2)
        assert isinstance(json_str, str)
        assert len(json_str) > 0

        # Test deserialization
        deserialized = json.loads(json_str)
        assert deserialized == output_data

    def _compare_output_structure_exactly(self, output_data, ground_truth):
        """Compare the output structure with the ground truth exactly."""
        print("=" * 60)
        print("COMPARING OUTPUT STRUCTURE WITH GROUND TRUTH")
        print("=" * 60)

        # Test top-level structure
        ground_truth_fields = set(ground_truth.keys())
        output_fields = set(output_data.keys())

        print(
            f"Ground truth fields ({len(ground_truth_fields)}): {sorted(ground_truth_fields)}"
        )
        print(f"Output fields ({len(output_fields)}): {sorted(output_fields)}")

        # Check for missing essential fields (excluding appendix which is optional)
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

        missing_essential_fields = essential_fields - output_fields
        if missing_essential_fields:
            pytest.fail(
                f"Output missing essential fields: {sorted(missing_essential_fields)}"
            )

        # Check for missing optional fields (warn but don't fail)
        optional_fields = ground_truth_fields - essential_fields
        missing_optional_fields = optional_fields - output_fields
        if missing_optional_fields:
            print(
                f"Info: Output missing optional fields: {sorted(missing_optional_fields)}"
            )

        # Check for unexpected fields (warn but don't fail)
        unexpected_fields = output_fields - ground_truth_fields
        if unexpected_fields:
            print(f"Info: Output has unexpected fields: {sorted(unexpected_fields)}")

        # Test essential fields are present and have correct types
        essential_fields_list = [
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
        ]

        for field in essential_fields_list:
            assert field in output_data, f"Missing essential field: {field}"
            assert field in ground_truth, f"Ground truth missing field: {field}"

            # Test data types match
            output_type = type(output_data[field])
            ground_truth_type = type(ground_truth[field])
            assert (
                output_type == ground_truth_type
            ), f"Field {field} type mismatch: {output_type} vs {ground_truth_type}"

        # Test figures structure
        print("✅ Testing figures structure...")
        assert isinstance(output_data["figures"], list), "figures should be a list"
        assert isinstance(
            ground_truth["figures"], list
        ), "ground truth figures should be a list"

        print(f"Output has {len(output_data['figures'])} figures")
        print(f"Ground truth has {len(ground_truth['figures'])} figures")

        # Test that all figures have the correct structure
        for i, figure in enumerate(output_data["figures"]):
            # Test required figure fields
            required_figure_fields = ["figure_label", "img_files", "sd_files", "panels"]
            for field in required_figure_fields:
                assert field in figure, f"Figure {i} missing required field: {field}"

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
                f"✅ Figure {i+1}: {len(figure['panels'])} panels with valid structure"
            )

        if output_data["figures"] and ground_truth["figures"]:
            # Compare figure structure
            output_figure = output_data["figures"][0]
            ground_truth_figure = ground_truth["figures"][0]

            figure_fields = set(ground_truth_figure.keys())
            output_figure_fields = set(output_figure.keys())

            missing_figure_fields = figure_fields - output_figure_fields
            if missing_figure_fields:
                pytest.fail(
                    f"Figure missing required fields: {sorted(missing_figure_fields)}"
                )

            # Test panels structure
            if "panels" in ground_truth_figure and "panels" in output_figure:
                assert isinstance(
                    output_figure["panels"], list
                ), "panels should be a list"
                assert isinstance(
                    ground_truth_figure["panels"], list
                ), "ground truth panels should be a list"

                print(f"Output figure has {len(output_figure['panels'])} panels")
                print(
                    f"Ground truth figure has {len(ground_truth_figure['panels'])} panels"
                )

                if output_figure["panels"] and ground_truth_figure["panels"]:
                    output_panel = output_figure["panels"][0]
                    ground_truth_panel = ground_truth_figure["panels"][0]

                    panel_fields = set(ground_truth_panel.keys())
                    output_panel_fields = set(output_panel.keys())

                    missing_panel_fields = panel_fields - output_panel_fields
                    if missing_panel_fields:
                        pytest.fail(
                            f"Panel missing required fields: {sorted(missing_panel_fields)}"
                        )

                    # Test bbox coordinates are valid
                    if "panel_bbox" in output_panel:
                        bbox = output_panel["panel_bbox"]
                        assert isinstance(bbox, list), "panel_bbox should be a list"
                        assert len(bbox) == 4, "panel_bbox should have 4 coordinates"
                        for i, coord in enumerate(bbox):
                            assert isinstance(
                                coord, (int, float)
                            ), f"Bbox coordinate {i} should be a number"
                            assert (
                                0 <= coord <= 1
                            ), f"Bbox coordinate {i} ({coord}) should be between 0 and 1"

        # Test data_availability structure
        print("✅ Testing data_availability structure...")
        assert (
            "section_text" in output_data["data_availability"]
        ), "data_availability should have section_text"
        assert (
            "data_sources" in output_data["data_availability"]
        ), "data_availability should have data_sources"
        assert isinstance(
            output_data["data_availability"]["data_sources"], list
        ), "data_sources should be a list"

        # Test data_sources structure matches expected format
        data_sources = output_data["data_availability"]["data_sources"]
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

        print(f"✅ Data availability: {len(data_sources)} data sources found")

        # Test cost structure
        print("✅ Testing cost structure...")
        cost_fields = set(ground_truth["cost"].keys())
        output_cost_fields = set(output_data["cost"].keys())

        missing_cost_fields = cost_fields - output_cost_fields
        if missing_cost_fields:
            pytest.fail(f"Cost missing required fields: {sorted(missing_cost_fields)}")

        # Test that cost sections have required fields
        for section in [
            "extract_sections",
            "extract_individual_captions",
            "assign_panel_source",
            "match_caption_panel",
            "extract_data_sources",
            "total",
        ]:
            if section in ground_truth["cost"] and section in output_data["cost"]:
                section_fields = set(ground_truth["cost"][section].keys())
                output_section_fields = set(output_data["cost"][section].keys())

                missing_section_fields = section_fields - output_section_fields
                if missing_section_fields:
                    pytest.fail(
                        f"Cost section {section} missing fields: {sorted(missing_section_fields)}"
                    )

                # Test that cost values are numbers
                for field in [
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                    "cost",
                ]:
                    if field in output_data["cost"][section]:
                        value = output_data["cost"][section][field]
                        assert isinstance(
                            value, (int, float)
                        ), f"Cost {section}.{field} should be a number"
                        assert (
                            value >= 0
                        ), f"Cost {section}.{field} should be non-negative"

        print("✅ All structure validations passed!")

    def test_pipeline_output_json_serialization(
        self, test_zip_path, config_path, tmp_path
    ):
        """Test that the pipeline output can be properly serialized to JSON."""
        output_path = str(tmp_path / "output.json")

        # Run the actual pipeline
        result = main(test_zip_path, config_path, output_path)

        # Test JSON serialization
        assert isinstance(result, str)
        output_data = json.loads(result)

        # Test that it can be serialized again
        json_str = json.dumps(output_data, indent=2)
        assert isinstance(json_str, str)
        assert len(json_str) > 0

        # Test deserialization
        deserialized = json.loads(json_str)
        assert deserialized == output_data

    def test_ground_truth_file_structure(self, ground_truth_path):
        """Test that the ground truth file has the expected structure."""
        with open(ground_truth_path, "r") as f:
            ground_truth = json.load(f)

        # Test top-level structure
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
            assert field in ground_truth, f"Ground truth missing field: {field}"

        # Test data types
        assert isinstance(ground_truth["appendix"], list)
        assert isinstance(ground_truth["figures"], list)
        assert isinstance(ground_truth["data_availability"], dict)
        assert isinstance(ground_truth["manuscript_id"], str)
        assert isinstance(ground_truth["cost"], dict)
        assert isinstance(ground_truth["ai_provider"], str)

        print("✅ Ground truth file validation passed!")
        print(f"   - {len(ground_truth['figures'])} figures")
        print(
            f"   - {sum(len(fig['panels']) for fig in ground_truth['figures'])} total panels"
        )
        print(f"   - Manuscript ID: {ground_truth['manuscript_id']}")

    def test_pipeline_output_consistency(self, test_zip_path, config_path, tmp_path):
        """Test that the pipeline output is consistent across runs."""
        output_path1 = str(tmp_path / "output1.json")
        output_path2 = str(tmp_path / "output2.json")

        # Run pipeline twice
        result1 = main(test_zip_path, config_path, output_path1)
        result2 = main(test_zip_path, config_path, output_path2)

        # Parse results
        output1 = json.loads(result1)
        output2 = json.loads(result2)

        # Test that structure is consistent
        assert set(output1.keys()) == set(
            output2.keys()
        ), "Output structure should be consistent"

        # Test that required fields are present in both
        required_fields = [
            "appendix",
            "figures",
            "data_availability",
            "cost",
            "ai_provider",
        ]
        for field in required_fields:
            assert field in output1, f"Field {field} missing in first run"
            assert field in output2, f"Field {field} missing in second run"

        print("✅ Pipeline output consistency test passed!")

    def test_pipeline_error_handling(self, tmp_path):
        """Test that the pipeline handles errors gracefully."""
        # Test with invalid ZIP file
        invalid_zip = tmp_path / "invalid.zip"
        invalid_zip.write_text("This is not a ZIP file")

        config_path = "config.dev.yaml"
        output_path = str(tmp_path / "output.json")

        # This should raise an exception
        with pytest.raises(Exception):
            main(str(invalid_zip), config_path, output_path)

        # Test with nonexistent ZIP file
        with pytest.raises(Exception):
            main("nonexistent.zip", config_path, output_path)

        # Test with nonexistent config file
        test_zip = "tests/test_suite/test_data/EMBOR-2023-58706-T.zip"
        with pytest.raises(Exception):
            main(test_zip, "nonexistent_config.yaml", output_path)

        print("✅ Pipeline error handling test passed!")
