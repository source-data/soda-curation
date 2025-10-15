"""
Test suite specifically for ZIP file validation issues.

This test suite demonstrates and validates the ZIP file validation
that was causing the original issue.
"""
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.soda_curation._main_utils import validate_paths


class TestZipValidation:
    """Test ZIP file validation to catch the exact issue you experienced."""

    def test_validate_paths_with_valid_zip(self, tmp_path):
        """Test that validate_paths works with a valid ZIP file."""
        # Create a valid ZIP file
        zip_path = tmp_path / "valid.zip"
        config_path = tmp_path / "config.yaml"
        output_path = tmp_path / "output.json"

        # Create config file
        config_path.write_text("test: config")

        # Create a valid ZIP file
        with zipfile.ZipFile(zip_path, "w") as zip_ref:
            zip_ref.writestr("test.xml", "<?xml version='1.0'?><root></root>")
            zip_ref.writestr("graphic/figure1.pdf", "dummy content")

        # This should not raise any exception
        validate_paths(str(zip_path), str(config_path), str(output_path))

    def test_validate_paths_with_empty_file(self, tmp_path):
        """Test that validate_paths fails with an empty file (the original issue)."""
        # Create an empty file (not a ZIP)
        zip_path = tmp_path / "empty.zip"
        config_path = tmp_path / "config.yaml"
        output_path = tmp_path / "output.json"

        # Create config file
        config_path.write_text("test: config")

        # Create an empty file (this is what was causing the original issue)
        zip_path.touch()

        # This should raise a BadZipFile exception
        with pytest.raises(zipfile.BadZipFile, match="Invalid ZIP file"):
            validate_paths(str(zip_path), str(config_path), str(output_path))

    def test_validate_paths_with_corrupted_zip(self, tmp_path):
        """Test that validate_paths fails with a corrupted ZIP file."""
        # Create a file that looks like a ZIP but is corrupted
        zip_path = tmp_path / "corrupted.zip"
        config_path = tmp_path / "config.yaml"
        output_path = tmp_path / "output.json"

        # Create config file
        config_path.write_text("test: config")

        # Create a file with ZIP-like content but corrupted
        zip_path.write_text("PK\x03\x04corrupted content")

        # This should raise a BadZipFile exception
        with pytest.raises(zipfile.BadZipFile, match="Invalid ZIP file"):
            validate_paths(str(zip_path), str(config_path), str(output_path))

    def test_validate_paths_with_nonexistent_file(self, tmp_path):
        """Test that validate_paths fails with a nonexistent file."""
        config_path = tmp_path / "config.yaml"
        output_path = tmp_path / "output.json"

        # Create config file
        config_path.write_text("test: config")

        # This should raise a FileNotFoundError
        with pytest.raises(FileNotFoundError, match="ZIP file.*does not exist"):
            validate_paths("nonexistent.zip", str(config_path), str(output_path))

    def test_validate_paths_with_empty_zip(self, tmp_path):
        """Test that validate_paths fails with an empty ZIP file."""
        # Create an empty ZIP file
        zip_path = tmp_path / "empty.zip"
        config_path = tmp_path / "config.yaml"
        output_path = tmp_path / "output.json"

        # Create config file
        config_path.write_text("test: config")

        # Create an empty ZIP file
        with zipfile.ZipFile(zip_path, "w"):
            pass  # Empty ZIP

        # This should raise a BadZipFile exception
        with pytest.raises(zipfile.BadZipFile, match="ZIP file.*is empty"):
            validate_paths(str(zip_path), str(config_path), str(output_path))

    def test_validate_paths_with_text_file_named_zip(self, tmp_path):
        """Test that validate_paths fails with a text file named .zip."""
        # Create a text file with .zip extension
        zip_path = tmp_path / "text.zip"
        config_path = tmp_path / "config.yaml"
        output_path = tmp_path / "output.json"

        # Create config file
        config_path.write_text("test: config")

        # Create a text file with .zip extension
        zip_path.write_text("This is just a text file, not a ZIP file")

        # This should raise a BadZipFile exception
        with pytest.raises(zipfile.BadZipFile, match="Invalid ZIP file"):
            validate_paths(str(zip_path), str(config_path), str(output_path))

    def test_validate_paths_error_messages(self, tmp_path):
        """Test that validate_paths provides helpful error messages."""
        config_path = tmp_path / "config.yaml"
        output_path = tmp_path / "output.json"

        # Create config file
        config_path.write_text("test: config")

        # Test different error scenarios
        test_cases = [
            ("", "ZIP path must be provided"),
            (None, "ZIP path must be provided"),
            ("nonexistent.zip", "ZIP file.*does not exist"),
        ]

        for zip_path, expected_error in test_cases:
            with pytest.raises((ValueError, FileNotFoundError), match=expected_error):
                validate_paths(zip_path, str(config_path), str(output_path))

    def test_zip_validation_integration(self, tmp_path):
        """Test ZIP validation as part of the main pipeline flow."""
        # This test simulates what happens when the main function is called
        zip_path = tmp_path / "test.zip"
        config_path = tmp_path / "config.yaml"
        output_path = tmp_path / "output.json"

        # Create config file
        config_path.write_text("test: config")

        # Test with invalid ZIP (empty file)
        zip_path.touch()

        # This should fail at the validate_paths step
        with pytest.raises(zipfile.BadZipFile):
            validate_paths(str(zip_path), str(config_path), str(output_path))

        # Now create a valid ZIP
        with zipfile.ZipFile(zip_path, "w") as zip_ref:
            zip_ref.writestr("test.xml", "<?xml version='1.0'?><root></root>")

        # This should pass
        validate_paths(str(zip_path), str(config_path), str(output_path))

    def test_zip_validation_with_real_zip_structure(self, tmp_path):
        """Test ZIP validation with a realistic ZIP file structure."""
        zip_path = tmp_path / "realistic.zip"
        config_path = tmp_path / "config.yaml"
        output_path = tmp_path / "output.json"

        # Create config file
        config_path.write_text("test: config")

        # Create a realistic ZIP file structure
        with zipfile.ZipFile(zip_path, "w") as zip_ref:
            # Add XML file (required)
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

        # This should pass validation
        validate_paths(str(zip_path), str(config_path), str(output_path))

        # Verify the ZIP file has the expected structure
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            files = zip_ref.namelist()
            assert "EMBOJ-2024-119382.xml" in files
            assert any(f.startswith("graphic/") for f in files)
            assert any(f.startswith("suppl_data/") for f in files)
            assert any(f.startswith("Doc/") for f in files)
            assert any(f.startswith("pdf/") for f in files)
