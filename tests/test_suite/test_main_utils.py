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
    normalize,
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
        validate_paths("", "config.yaml")


def test_validate_paths_missing_config():
    """Test validation fails when no config path provided."""
    with pytest.raises(ValueError, match="config path must be provided"):
        validate_paths("test.zip", "")


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
        <p>(D) Effects of Dyrk4 deficiency on the virus-induced transcription of
        downstream antiviral genes. <em>Dyrk4</em><sup>+/+</sup> and
        <em>Dyrk</em>4<sup>-/-</sup> BMDMs were infected with SeV for 8 h before
        qPCR analysis. (<em>P</em> value; <em>Ifnb1</em>: 3.4 × 10<sup>-6</sup>,
        <em>Isg15</em>: 2.6 × 10<sup>-5</sup>, <em>Cxcl10</em>: 3.3 ×
        10<sup>-5</sup>, <em>Il6</em>: 1.6 × 10<sup>-6</sup>) (<em>n</em> = 3
        biological replicates).</p>
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
            calculate_hallucination_score(self.plain_text_match, self.source_text),
            0.0,
            f"EXACT_MATCH: {self.exact_match}, SOURCE_TEXT: {self.source_text}",
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
        self.assertEqual(
            exact_score,
            0.0,
            f"EXACT_MATCH: {self.exact_match}, SOURCE_TEXT: {self.source_text}",
        )

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


class TestNormalizeFunction(unittest.TestCase):
    """Test the core normalize function with various options."""

    def test_default_settings(self):
        """Test normalize with default settings."""
        original = (
            "This is a TEST<em>with</em> punctuation: $100, 50% off!\nMultiple lines."
        )
        normalized = normalize(original)
        self.assertIn("this is a test with punctuation", normalized)
        self.assertNotIn("$", normalized)  # Punctuation removed
        self.assertNotIn("\n", normalized)  # Line breaks converted to spaces

    def test_selective_operations(self):
        """Test normalize with specific operations."""
        original = (
            "This is a TEST<em>with</em> punctuation: $100, 50% off!\nMultiple lines."
        )
        selective_ops = normalize(original, do=["strip", "lower", "line_breaks"])
        self.assertEqual(
            "this is a test<em>with</em> punctuation: $100, 50% off! multiple lines.",
            selective_ops,
        )

    def test_keep_punctuation(self):
        """Test keeping certain punctuation."""
        original = (
            "This is a TEST<em>with</em> punctuation: $100, 50% off!\nMultiple lines."
        )
        keep_punct = normalize(
            original,
            do_not_remove="$%",
            do=["strip", "lower", "punctuation", "line_breaks"],
        )
        self.assertIn("$", keep_punct)
        self.assertIn("%", keep_punct)
        self.assertNotIn(",", keep_punct)  # Other punctuation removed

    def test_special_character_preservation(self):
        """Test preservation of special character sequences."""
        special_chars = normalize(
            "Test with +/+ and -/- special sequences", do=["special_chars", "lower"]
        )
        self.assertIn("+/+", special_chars)
        self.assertIn("-/-", special_chars)

    def test_unicode_normalization(self):
        """Test Unicode normalization."""
        unicode_text = normalize("Café with açaí", do=["unicode", "lower"])
        self.assertIn("cafe", unicode_text)
        self.assertIn("acai", unicode_text)

    def test_empty_string(self):
        """Test handling of empty string."""
        self.assertEqual("", normalize(""))


class TestEnhancedNormalizeText(unittest.TestCase):
    """Test the enhanced normalize_text function."""

    def setUp(self):
        self.html_content = "<p>This is a <strong>TEST</strong> with $100 value.</p>"

    def test_default_parameters(self):
        """Test with default parameters (should strip HTML)."""
        default_norm = normalize_text(self.html_content)
        self.assertNotIn("<p>", default_norm)
        self.assertNotIn("<strong>", default_norm)
        self.assertEqual("this is a test with $100 value.", default_norm)

    def test_without_html_stripping(self):
        """Test without HTML stripping."""
        with_html = normalize_text(self.html_content, strip_html=False)
        self.assertIn("<p>", with_html)
        self.assertIn("<strong>", with_html)

    def test_keep_chars_parameter(self):
        """Test with keep_chars parameter."""
        keep_dollars = normalize_text(
            self.html_content, keep_chars="$", config={"remove_punctuation": True}
        )
        self.assertIn("$", keep_dollars)

    def test_config_parameters(self):
        """Test with config parameters."""
        config_norm = normalize_text(
            "Café with açaí and $100",
            config={"normalize_unicode": True, "remove_punctuation": True},
        )
        self.assertIn("cafe", config_norm)
        self.assertIn("acai", config_norm)
        self.assertNotIn("$", config_norm)  # Removed as punctuation

    def test_backward_compatibility(self):
        """Test backward compatibility with original code."""
        source_html = """
        <p>(D) Effects of Dyrk4 deficiency on the virus-induced transcription of
        downstream antiviral genes. <em>Dyrk4</em><sup>+/+</sup> and
        <em>Dyrk</em>4<sup>-/-</sup> BMDMs were infected with SeV for 8 h before
        qPCR analysis.</p>
        """

        normalized = normalize_text(source_html)
        self.assertIn("+/+", normalized)  # Should preserve special sequences
        self.assertIn("-/-", normalized)
        self.assertNotIn("<p>", normalized)  # Should strip HTML
        self.assertNotIn("<em>", normalized)

    def test_special_case_handling(self):
        """Test special case handling for data with superscripts."""
        sup_text = (
            "Wild-type (Dyrk4<sup>+/+</sup>) and knockout (Dyrk4<sup>-/-</sup>) mice"
        )
        normalized_sup = normalize_text(sup_text)
        self.assertIn("+/+", normalized_sup)
        self.assertIn("-/-", normalized_sup)


class TestNormalizeWithComplexText(unittest.TestCase):
    """Test normalize functions with more complex text examples."""

    def setUp(self):
        self.complex_html = """
        <p class="section">Results from <em>in vivo</em> experiments</p>
        <p>The wild-type (WT, <em>Dyrk4</em><sup>+/+</sup>) and knockout (KO, <em>Dyrk4</em><sup>-/-</sup>) 
        mice were infected with 2×10<sup>4</sup> PFU of virus. Measurements were taken at 
        24, 48, and 72 hours post-infection. The p-value was p &lt; 0.001.</p>
        <ol>
            <li>First measurement: 12.5 μg/mL</li>
            <li>Second measurement: 24.7 μg/mL</li>
        </ol>
        """

    def test_normalize_with_complex_html(self):
        """Test normalize with complex HTML content."""
        normalized = normalize(self.complex_html, do=["html_tags", "strip", "lower"])

        # Check content preserved
        self.assertIn("results from in vivo experiments", normalized)
        self.assertIn("wild-type", normalized)
        self.assertIn("dyrk4 +/+", normalized)
        self.assertIn("dyrk4 -/-", normalized)
        self.assertIn("2×10 4 pfu", normalized.lower())

        # Check HTML removed
        self.assertNotIn("<p>", normalized)
        self.assertNotIn("<em>", normalized)
        self.assertNotIn("<ol>", normalized)

        # Check entities decoded
        self.assertIn("p < 0.001", normalized)

    def test_normalize_text_with_complex_html(self):
        """Test normalize_text with complex HTML and various configs."""
        # Basic normalization
        basic = normalize_text(self.complex_html)
        self.assertIn("results from in vivo experiments", basic)
        self.assertNotIn("<p>", basic)
        self.assertIn("dyrk4 +/+", basic)
        self.assertIn("dyrk4 -/-", basic)

        # Keep units in punctuation
        with_units = normalize_text(
            self.complex_html, keep_chars="/<>", config={"remove_punctuation": True}
        )
        self.assertIn("μg/ml", with_units.lower())
        self.assertIn("<", with_units)  # preserved from p < 0.001

        # Normalize unicode but keep special chars
        unicode_norm = normalize_text(
            self.complex_html, config={"normalize_unicode": True}
        )
        self.assertNotIn("μg", unicode_norm)  # Greek mu normalized
        self.assertIn("dyrk4 +/+", unicode_norm)  # Special sequence preserved

        # Test with no HTML stripping
        with_html = normalize_text(self.complex_html, strip_html=False)
        self.assertIn("<p", with_html)
        self.assertIn("<em>", with_html)


class TestNormalizeCompatibility(unittest.TestCase):
    """Test compatibility with hallucination detection and other existing functionality."""

    def test_hallucination_detection_compatibility(self):
        """Ensure normalize functions work with hallucination detection."""
        source = """
        <p>The experiment showed <em>Dyrk4</em><sup>+/+</sup> mice had significantly higher 
        expression of IFN-β compared to <em>Dyrk4</em><sup>-/-</sup> mice.</p>
        """

        extracted = "The experiment showed Dyrk4 +/+ mice had significantly higher expression of IFN-β"

        # Test that this is properly detected as not hallucinated
        norm_source = normalize_text(source)
        norm_extracted = normalize_text(extracted)

        self.assertIn(norm_extracted, norm_source)

        # Verify hallucination score
        self.assertEqual(calculate_hallucination_score(extracted, source), 0.0)

        # Test with a hallucinated extract
        hallucinated = (
            "The experiment showed Dyrk4 +/+ mice had reduced viral titers in the lung."
        )
        self.assertGreater(calculate_hallucination_score(hallucinated, source), 0.2)

    def test_fuzzy_matching_integration(self):
        """Test integration with fuzzy matching."""
        source = """
        <p>Measurements were taken at 24, 48, and 72 hours post-infection.</p>
        """

        # Should still match with slight variations
        close_extract = "Measurements were done at 24, 48, and 72h after infection."

        # Calculate similarity score using fuzzy matching
        similarity = fuzzy_match_score(close_extract, source)
        self.assertGreater(similarity, 75)  # Should have good similarity
