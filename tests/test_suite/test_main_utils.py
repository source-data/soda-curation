"""Tests for main utility functions."""

import os
import shutil
import unittest
import zipfile
from pathlib import Path

import pytest

from src.soda_curation._main_utils import (
    calculate_hallucination_score,
    cleanup_extract_dir,
    exact_match_check,
    fuzzy_match_score,
    normalize_text,
    setup_extract_dir,
    strip_html_tags,
    validate_paths,
    write_output,
)
from src.soda_curation.pipeline.manuscript_structure.manuscript_xml_parser import (
    XMLStructureExtractor,
)


@pytest.fixture
def mock_paths(tmp_path):
    """Create mock paths for testing."""
    return {
        "zip_path": str(tmp_path / "test.zip"),
        "config_path": str(tmp_path / "config.yaml"),
        "output_path": str(tmp_path / "output.json"),
    }


@pytest.fixture
def mock_zip_with_docx(tmp_path):
    """Create a test ZIP file with a DOCX in manuscript ID directory structure."""
    zip_path = tmp_path / "test.zip"
    manuscript_id = "TEST-2023-12345"
    docx_content = b"test docx content"

    with zipfile.ZipFile(zip_path, "w") as zf:
        # Add files preserving manuscript ID prefix
        zf.writestr(f"{manuscript_id}/Doc/manuscript.docx", docx_content)
        zf.writestr(f"{manuscript_id}.xml", "<xml>test</xml>")

    return zip_path, manuscript_id, docx_content


def test_validate_paths_missing_zip():
    """Test validation fails when no ZIP path provided."""
    with pytest.raises(ValueError, match="ZIP path must be provided"):
        validate_paths(None, "config.yaml")


def test_validate_paths_missing_config():
    """Test validation fails when no config path provided."""
    with pytest.raises(ValueError, match="config path must be provided"):
        validate_paths("test.zip", None)


def test_validate_paths_nonexistent_zip(mock_paths):
    """Test validation fails when ZIP file doesn't exist."""
    with pytest.raises(FileNotFoundError, match="ZIP file .* does not exist"):
        validate_paths(mock_paths["zip_path"], mock_paths["config_path"])


def test_validate_paths_nonexistent_config(mock_paths):
    """Test validation fails when config file doesn't exist."""
    Path(mock_paths["zip_path"]).touch()
    with pytest.raises(FileNotFoundError, match="Config file .* does not exist"):
        validate_paths(mock_paths["zip_path"], mock_paths["config_path"])


def test_validate_paths_creates_output_dir(mock_paths, tmp_path):
    """Test validation creates output directory if needed."""
    Path(mock_paths["zip_path"]).touch()
    Path(mock_paths["config_path"]).touch()

    nested_output = tmp_path / "nested" / "path" / "output.json"
    validate_paths(
        mock_paths["zip_path"], mock_paths["config_path"], str(nested_output)
    )

    assert nested_output.parent.exists()


def test_setup_extract_dir():
    """Test temporary directory creation."""
    extract_dir = setup_extract_dir()
    try:
        assert extract_dir.exists()
        assert extract_dir.is_dir()
        assert "soda_curation_" in extract_dir.name

        # Verify we can write to it
        test_file = extract_dir / "test.txt"
        test_file.write_text("test")
        assert test_file.exists()

    finally:
        cleanup_extract_dir(extract_dir)
        assert not extract_dir.exists()


def test_write_output_success(tmp_path):
    """Test successful output writing."""
    output_path = tmp_path / "output.json"
    test_json = '{"test": "data"}'

    write_output(test_json, str(output_path))

    assert output_path.exists()
    assert output_path.read_text() == test_json


def test_write_output_failure(tmp_path):
    """Test output writing failure."""
    invalid_path = tmp_path / "nonexistent" / "output.json"

    with pytest.raises(Exception):
        write_output('{"test": "data"}', str(invalid_path))


def test_cleanup_extract_dir(tmp_path):
    """Test extraction directory cleanup."""
    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    (extract_dir / "test.txt").touch()

    cleanup_extract_dir(extract_dir)

    assert not extract_dir.exists()


def test_cleanup_nonexistent_dir():
    """Test cleanup of nonexistent directory doesn't raise."""
    cleanup_extract_dir(Path("/nonexistent/path"))


def test_extract_and_find_docx(mock_zip_with_docx):
    """Test extracting ZIP and finding DOCX with manuscript ID prefix."""
    zip_path, manuscript_id, docx_content = mock_zip_with_docx
    extract_dir = setup_extract_dir()

    try:
        # Extract ZIP contents
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

        # Verify DOCX exists at expected path with manuscript ID
        docx_path = extract_dir / manuscript_id / "Doc/manuscript.docx"
        assert docx_path.exists()
        assert docx_path.read_bytes() == docx_content

        # Verify direct path from manuscript ID prefix works
        docx_relative = f"{manuscript_id}/Doc/manuscript.docx"
        docx_full = extract_dir / docx_relative
        assert docx_full.exists()
        assert docx_full.read_bytes() == docx_content

    finally:
        cleanup_extract_dir(extract_dir)
        assert not extract_dir.exists()


def test_extract_dir_cleanup_handles_permission_error(tmp_path):
    """Test cleanup handles permission errors gracefully."""
    extract_dir = setup_extract_dir()
    try:
        # Create nested structure
        nested_dir = extract_dir / "nested"
        nested_dir.mkdir()
        test_file = nested_dir / "test.txt"
        test_file.write_text("test")

        # Make directory read-only on Unix-like systems
        if os.name != "nt":  # Skip on Windows
            nested_dir.chmod(0o555)

        cleanup_extract_dir(extract_dir)
        # Even with permission error, should not raise

    finally:
        # Ensure cleanup in case of test failure
        if extract_dir.exists():
            # Reset permissions to allow cleanup
            if os.name != "nt":
                nested_dir.chmod(0o755)
            shutil.rmtree(extract_dir)


def test_extraction_preserves_manuscript_structure(tmp_path):
    """Test that files are extracted to manuscript ID subdirectory."""
    # Create test ZIP with manuscript structure
    zip_path = tmp_path / "test.zip"
    manuscript_id = "TEST-2023-12345"

    with zipfile.ZipFile(zip_path, "w") as zf:
        # Add XML file
        zf.writestr(f"{manuscript_id}.xml", "<xml>test content</xml>")

        # Add files with and without manuscript ID prefix
        zf.writestr(f"{manuscript_id}/Doc/manuscript.docx", "test docx")
        zf.writestr(f"{manuscript_id}/graphic/Figure 1.tif", "test image")
        zf.writestr("suppl_data/source.zip", "test data")

    # Extract files
    extract_dir = tmp_path / "extract"
    extractor = XMLStructureExtractor(str(zip_path), str(extract_dir))

    # Verify structure
    manuscript_dir = extract_dir / manuscript_id
    assert manuscript_dir.exists()

    # Check paths relative to manuscript directory
    assert (manuscript_dir / "Doc/manuscript.docx").exists()
    assert (manuscript_dir / "graphic/Figure 1.tif").exists()
    assert (manuscript_dir / "suppl_data/source.zip").exists()

    # Verify file contents are preserved
    with open(manuscript_dir / "Doc/manuscript.docx") as f:
        assert f.read() == "test docx"

    # Check that XML file is accessible
    assert extractor.manuscript_id == manuscript_id

    # Test full path resolution
    test_path = "Doc/manuscript.docx"
    full_path = extractor.get_full_path(test_path)
    assert full_path == manuscript_dir / test_path
    assert full_path.exists()


class TestHallucinationDetection(unittest.TestCase):
    def setUp(self):
        # Example source document content (with HTML)
        self.source_text = """
        <p><strong>Fig. 2. Dyrk4 is required for the virus-mediated innate
        immune response.</strong></p>
        <p>(A-C) Effects of Dyrk4 deficiency on the virus-induced transcription
        of <em>Ifnb1</em>. <em>Dyrk4</em><sup>+/+</sup> and
        <em>Dyrk</em>4<sup>-/-</sup> BMDCs, BMDMs, and MLFs were infected with
        SeV, VSV or HSV-1 for 8 h before qPCR analysis. (<em>P</em> value; A;
        BMDCs: 0.0025, MLFs: 0.00103; B; BMDCs: 3.2 × 10<sup>-6</sup>, MLFs:
        0.00097, BMDMs: 0.0026; C; MLFs: 3.7 × 10<sup>-5</sup>) (<em>n</em> = 3
        biological replicates).</p>
        <p>(D) Effects of Dyrk4 deficiency on the virus-induced transcription of
        downstream antiviral genes. <em>Dyrk4</em><sup>+/+</sup> and
        <em>Dyrk</em>4<sup>-/-</sup> BMDMs were infected with SeV for 8 h before
        qPCR analysis. (<em>P</em> value; <em>Ifnb1</em>: 3.4 × 10<sup>-6</sup>,
        <em>Isg15</em>: 2.6 × 10<sup>-5</sup>, <em>Cxcl10</em>: 3.3 ×
        10<sup>-5</sup>, <em>Il6</em>: 1.6 × 10<sup>-6</sup>) (<em>n</em> = 3
        biological replicates).</p>
        """

        # Example exact match (with HTML)
        self.exact_match = """
        (D) Effects of Dyrk4 deficiency on the virus-induced transcription of
        downstream antiviral genes. <em>Dyrk4</em><sup>+/+</sup> and
        <em>Dyrk</em>4<sup>-/-</sup> BMDMs were infected with SeV for 8 h before
        qPCR analysis.
        """

        # Plain text version (without HTML tags)
        self.plain_text_match = """
        (D) Effects of Dyrk4 deficiency on the virus-induced transcription of
        downstream antiviral genes. Dyrk4+/+ and
        Dyrk4-/- BMDMs were infected with SeV for 8 h before
        qPCR analysis.
        """

        # Example close match (minor differences)
        self.close_match = """
        (D) Effects of Dyrk4 deficiency on virus-induced transcription of
        downstream antiviral genes. Dyrk4+/+ and
        Dyrk4-/- BMDMs were infected with SeV for 8h before
        qPCR analysis.
        """

        # Example hallucination (not in source)
        self.hallucination = """
        (E) Dyrk4 knockout mice exhibited decreased virus clearance and
        increased susceptibility to viral infection compared to wild type.
        """

    def test_strip_html_tags(self):
        """Test HTML tag stripping function"""
        html = "<p>This is <strong>bold</strong> and <em>italic</em> text.</p>"
        plain = strip_html_tags(html)
        self.assertEqual(plain, "This is bold and italic text.")

        # Test with sup tags
        html_sup = "Test <em>Dyrk4</em><sup>+/+</sup> and <em>Dyrk</em>4<sup>-/-</sup>"
        plain_sup = strip_html_tags(html_sup)
        self.assertIn("+/+", plain_sup)
        self.assertIn("-/-", plain_sup)

    def test_normalize_text(self):
        """Test text normalization function"""
        # Test without HTML stripping
        normalized_with_html = normalize_text(
            "This is a <strong>TEST</strong>\nwith\r\nmultiple   spaces.",
            strip_html=False,
        )
        self.assertEqual(
            normalized_with_html,
            "this is a <strong>test</strong> with multiple spaces.",
        )

        # Test with HTML stripping
        normalized_without_html = normalize_text(
            "This is a <strong>TEST</strong>\nwith\r\nmultiple   spaces."
        )
        self.assertEqual(
            normalized_without_html, "this is a test with multiple spaces."
        )

        # Test with sup/sub tags
        html_sup = "Test <em>Dyrk4</em><sup>+/+</sup> and <em>Dyrk</em>4<sup>-/-</sup>"
        normalized_with_sup = normalize_text(html_sup)
        self.assertIn("+/+", normalized_with_sup)
        self.assertIn("-/-", normalized_with_sup)

    def test_exact_match_check(self):
        """Test exact matching functionality"""
        # Should find exact HTML match
        self.assertTrue(exact_match_check(self.exact_match, self.source_text))

        # Plain text against HTML should NOT be an exact match
        # but should be detected via the hallucination score
        self.assertFalse(exact_match_check(self.plain_text_match, self.source_text))

        # Verify that despite not being an exact match, the hallucination
        # score correctly identifies that this is not a hallucination
        self.assertEqual(
            calculate_hallucination_score(self.plain_text_match, self.source_text), 0.0
        )

        # Shouldn't find close matches or hallucinations
        self.assertFalse(exact_match_check(self.close_match, self.source_text))
        self.assertFalse(exact_match_check(self.hallucination, self.source_text))

    def test_fuzzy_match_score(self):
        """Test fuzzy matching functionality"""
        # Exact match should have very high score (near 100)
        exact_score = fuzzy_match_score(self.exact_match, self.source_text)
        self.assertGreaterEqual(exact_score, 95)

        # Plain text match should also have very high score
        plain_text_score = fuzzy_match_score(self.plain_text_match, self.source_text)
        self.assertGreaterEqual(plain_text_score, 95)
        print(f"\nDEBUG plain text fuzzy match score: {plain_text_score}")

        # Close match should have decent score
        close_score = fuzzy_match_score(self.close_match, self.source_text)
        self.assertGreaterEqual(close_score, 70)

        # Hallucination should have lower score
        hallucination_score = fuzzy_match_score(self.hallucination, self.source_text)
        self.assertLess(hallucination_score, 60)

    def test_hallucination_score(self):
        """Test hallucination score calculation"""
        # Exact match should have 0 hallucination score
        exact_score = calculate_hallucination_score(self.exact_match, self.source_text)
        self.assertEqual(exact_score, 0.0)

        # Plain text match should also have 0 hallucination score
        plain_text_score = calculate_hallucination_score(
            self.plain_text_match, self.source_text
        )
        self.assertEqual(plain_text_score, 0.0)

        # Close match should have low hallucination score
        close_score = calculate_hallucination_score(self.close_match, self.source_text)
        self.assertLess(close_score, 0.3)

        # Hallucination should have high hallucination score
        hallucination_score = calculate_hallucination_score(
            self.hallucination, self.source_text
        )
        self.assertGreater(hallucination_score, 0.4)

    def test_edge_cases(self):
        """Test hallucination detection with edge cases"""
        # Empty strings
        self.assertEqual(calculate_hallucination_score("", self.source_text), 1.0)
        self.assertEqual(calculate_hallucination_score(self.exact_match, ""), 1.0)
        self.assertEqual(calculate_hallucination_score("", ""), 1.0)
