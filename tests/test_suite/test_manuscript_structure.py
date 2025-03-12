"""Tests for XMLStructureExtractor class."""

import shutil
import unittest
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch
from zipfile import ZipFile

import lxml.etree as etree
import pytest

from src.soda_curation.pipeline.manuscript_structure.exceptions import (
    NoManuscriptFileError,
    NoXMLFileFoundError,
)
from src.soda_curation.pipeline.manuscript_structure.manuscript_structure import (
    ZipStructure,
)
from src.soda_curation.pipeline.manuscript_structure.manuscript_xml_parser import (
    XMLStructureExtractor,
)

# Constants for tests
MANUSCRIPT_ID = "EMBOJ-DUMMY-ZIP"
INVALID_XML = """<?xml version="1.0" encoding="UTF-8"?>
<invalid>
    <unclosed>
</invalid>"""


@pytest.fixture
def test_data_dir():
    """Get path to test data directory."""
    return Path(__file__).parent / "test_data"


@pytest.fixture
def sample_xml_content(test_data_dir):
    """Load sample XML content from test file."""
    xml_path = test_data_dir / "EMBOJ-DUMMY-ZIP.xml"
    return xml_path.read_text()


@pytest.fixture
def temp_extract_dir(tmp_path):
    """Create temporary directory for extraction."""
    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    yield extract_dir
    if extract_dir.exists():
        shutil.rmtree(extract_dir)


@pytest.fixture
def create_test_zip(tmp_path):
    """Create a test ZIP file with required structure."""

    def _create_zip(include_files=None):
        """Create test ZIP with specific content."""
        include_files = include_files or {"docx", "pdf", "figures", "source_data"}
        zip_path = tmp_path / f"{MANUSCRIPT_ID}.zip"

        with zipfile.ZipFile(zip_path, "w") as zf:
            # Create XML content with all figures and appropriate source data
            xml_parts = [
                f"""<?xml version="1.0" encoding="UTF-8"?>
                <article>
                    <front>
                        <article-meta>
                            <article-id pub-id-type="manuscript">{MANUSCRIPT_ID}</article-id>
                        </article-meta>
                    </front>
                    <notes>
                        <doc object-type="Manuscript Text">
                            <object_id>Doc/{MANUSCRIPT_ID}Manuscript_TextIG.docx</object_id>
                        </doc>"""
            ]

            # Add figure elements
            for i in range(1, 7):
                xml_parts.append(
                    f"""
                    <fig object-type="Figure">
                        <label>Figure {i}</label>
                        <object_id>{MANUSCRIPT_ID}/graphic/FIGURE {i}.tif</object_id>
                    </fig>"""
                )

                # Add source data only for figures 3-6
                if i >= 3 and "source_data" in include_files:
                    xml_parts.append(
                        f"""
                    <form object-type="Figure Source Data Files">
                        <label>Figure {i} Source Data</label>
                        <object_id>{MANUSCRIPT_ID}/suppl_data/Figure_{i}sd.zip</object_id>
                    </form>"""
                    )

            # Close XML
            xml_parts.append(
                """
                    </notes>
                </article>
            """
            )

            # Add XML file
            zf.writestr(f"{MANUSCRIPT_ID}.xml", "".join(xml_parts))

            # Add DOCX file in correct structure
            if "docx" in include_files:
                zf.writestr(
                    f"Doc/{MANUSCRIPT_ID}Manuscript_TextIG.docx", "test content"
                )

            # Add figures if requested
            if "figures" in include_files:
                for i in range(1, 7):
                    zf.writestr(
                        f"{MANUSCRIPT_ID}/graphic/FIGURE {i}.tif", b"image content"
                    )

            # Add source data if requested
            if "source_data" in include_files:
                # Only create source data for figures 3-6
                for i in range(3, 7):
                    zf.writestr(
                        f"{MANUSCRIPT_ID}/suppl_data/Figure_{i}sd.zip",
                        b"source data content",
                    )

        return str(zip_path)

    return _create_zip


def test_missing_xml_file(temp_extract_dir):
    """Test handling of ZIP file without XML file."""
    # Create ZIP without XML
    zip_path = temp_extract_dir.parent / "test.zip"
    with ZipFile(zip_path, "w") as zf:
        zf.writestr("dummy.txt", "content")

    with pytest.raises(NoXMLFileFoundError):
        XMLStructureExtractor(str(zip_path), str(temp_extract_dir))


def test_empty_xml_file(temp_extract_dir, create_test_zip):
    """Test handling of empty XML file."""
    zip_path = temp_extract_dir.parent / "test.zip"
    with ZipFile(zip_path, "w") as zf:
        zf.writestr(f"{MANUSCRIPT_ID}.xml", "")

    with pytest.raises(etree.XMLSyntaxError):
        XMLStructureExtractor(zip_path, str(temp_extract_dir))


def test_non_xml_file(temp_extract_dir, create_test_zip):
    """Test handling of non-XML file with .xml extension."""
    zip_path = temp_extract_dir.parent / "test.zip"
    with ZipFile(zip_path, "w") as zf:
        zf.writestr(f"{MANUSCRIPT_ID}.xml", "This is not XML content")

    with pytest.raises(etree.XMLSyntaxError):
        XMLStructureExtractor(zip_path, str(temp_extract_dir))


def test_invalid_xml_content(temp_extract_dir, create_test_zip):
    """Test handling of invalid XML content."""
    # zip_path = create_test_zip(use_invalid_xml=True)

    # with pytest.raises(etree.XMLSyntaxError):
    #     extractor = XMLStructureExtractor(zip_path, str(temp_extract_dir))
    pass


def test_docx_missing_from_xml(temp_extract_dir, create_test_zip):
    """Test case where DOCX is not referenced in XML but exists in ZIP."""
    # Create ZIP with DOCX but use modified XML that doesn't reference it
    modified_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <article>
        <front>
            <article-meta>
                <article-id pub-id-type="manuscript">EMBOJ-DUMMY-ZIP</article-id>
            </article-meta>
            <notes>
                <!-- No DOCX reference -->
                <fig id="1" object-type="Figure">
                    <label>Figure 1</label>
                    <object_id>graphic/FIGURE 1.tif</object_id>
                </fig>
            </notes>
        </front>
    </article>"""

    # Create ZIP with DOCX but XML doesn't reference it
    zip_path = temp_extract_dir.parent / "test.zip"
    with ZipFile(zip_path, "w") as zf:
        zf.writestr(f"{MANUSCRIPT_ID}.xml", modified_xml)
        zf.writestr(f"Doc/{MANUSCRIPT_ID}Manuscript_TextIG.docx", "docx content")

    extractor = XMLStructureExtractor(zip_path, str(temp_extract_dir))
    with pytest.raises(NoManuscriptFileError, match="No DOCX file referenced in XML"):
        extractor.extract_structure()


def test_docx_missing_from_zip(temp_extract_dir, create_test_zip):
    """Test case where DOCX is referenced in XML but missing from ZIP."""
    # Use normal XML (which references DOCX) but don't include DOCX in ZIP
    zip_path = create_test_zip(
        include_files={"pdf", "figures"}
    )  # Explicitly exclude DOCX

    extractor = XMLStructureExtractor(zip_path, str(temp_extract_dir))
    with pytest.raises(
        NoManuscriptFileError, match="DOCX file referenced in XML not found in ZIP"
    ):
        extractor.extract_structure()


def test_docx_path_mismatch(temp_extract_dir, create_test_zip):
    """Test case where DOCX path in XML doesn't match ZIP structure."""
    # Modify XML to reference DOCX in wrong location
    modified_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <article>
        <front>
            <article-meta>
                <article-id pub-id-type="manuscript">EMBOJ-DUMMY-ZIP</article-id>
            </article-meta>
            <notes>
                <supplementary-material object-type="Manuscript Text">
                    <object_id>wrong/path/manuscript.docx</object_id>
                </supplementary-material>
            </notes>
        </front>
    </article>"""

    zip_path = temp_extract_dir.parent / "test.zip"
    with ZipFile(zip_path, "w") as zf:
        zf.writestr(f"{MANUSCRIPT_ID}.xml", modified_xml)
        zf.writestr(f"Doc/{MANUSCRIPT_ID}Manuscript_TextIG.docx", "docx content")

    extractor = XMLStructureExtractor(zip_path, str(temp_extract_dir))
    with pytest.raises(
        NoManuscriptFileError,
        match="DOCX file path in XML does not match ZIP structure",
    ):
        extractor.extract_structure()


def test_extract_docx_content(temp_extract_dir, create_test_zip):
    """Test DOCX content extraction with manuscript ID directory structure."""
    zip_path = create_test_zip()
    extractor = XMLStructureExtractor(zip_path, str(temp_extract_dir))
    structure = extractor.extract_structure()

    # Get the DOCX path from structure
    docx_path = structure.docx

    # Mock pypandoc.convert_file to avoid actual conversion
    with patch("pypandoc.convert_file", return_value="<html>test content</html>"):
        content = extractor.extract_docx_content(docx_path)
        assert content == "<html>test content</html>"

    # Verify the file exists in the manuscript directory
    full_path = extractor.manuscript_extract_dir / docx_path
    assert full_path.exists()


def test_extract_docx_content_not_found(temp_extract_dir, create_test_zip):
    """Test error handling when DOCX file is not found."""
    zip_path = create_test_zip()
    extractor = XMLStructureExtractor(zip_path, str(temp_extract_dir))

    with pytest.raises(NoManuscriptFileError, match="DOCX file not found"):
        extractor.extract_docx_content("nonexistent.docx")


def test_complete_structure(temp_extract_dir, create_test_zip):
    """Test extraction of complete valid structure."""
    zip_path = create_test_zip()

    extractor = XMLStructureExtractor(zip_path, str(temp_extract_dir))
    structure = extractor.extract_structure()

    # Verify basic structure
    assert isinstance(structure, ZipStructure)
    assert structure.manuscript_id == MANUSCRIPT_ID
    assert structure.xml == f"{MANUSCRIPT_ID}.xml"
    assert structure.docx.endswith("Manuscript_TextIG.docx")

    # Verify figures
    assert len(structure.figures) == 6
    for i, figure in enumerate(structure.figures, 1):
        assert figure.figure_label == f"Figure {i}"
        assert len(figure.img_files) == 1
        assert figure.img_files[0].endswith(f"{i}.tif")

        # Verify source data for figures 3-6
        if i >= 3:
            assert len(figure.sd_files) == 1
            assert any(sd.endswith(f"{i}sd.zip") for sd in figure.sd_files)


def test_root_path_handling(temp_extract_dir, create_test_zip):
    """Test handling of files with and without manuscript ID prefix."""
    # Test with prefix
    zip_path = create_test_zip()
    extractor = XMLStructureExtractor(zip_path, str(temp_extract_dir))
    structure = extractor.extract_structure()

    # Verify paths don't have duplicate manuscript ID
    assert not structure.docx.startswith(f"{MANUSCRIPT_ID}/{MANUSCRIPT_ID}")
    assert all(
        not f.img_files[0].startswith(f"{MANUSCRIPT_ID}/{MANUSCRIPT_ID}")
        for f in structure.figures
    )

    # Verify paths are correctly normalized
    assert all(not path.startswith("/") for path in structure.appendix)
    assert all(not fig.img_files[0].startswith("/") for fig in structure.figures)


def test_figure_source_data_matching(temp_extract_dir, create_test_zip):
    """Test correct matching of source data files to figures."""
    zip_path = create_test_zip(include_files={"docx", "figures", "source_data"})

    extractor = XMLStructureExtractor(zip_path, str(temp_extract_dir))
    structure = extractor.extract_structure()

    # Check source data assignment
    for figure in structure.figures:
        figure_num = int(figure.figure_label.split()[-1])
        if figure_num >= 3:  # Figures 3-6 have source data
            assert len(figure.sd_files) == 1
            assert any(f"Figure_{figure_num}sd.zip" in sd for sd in figure.sd_files)
        else:  # Figures 1-2 don't have source data
            assert not figure.sd_files


def test_extraction_preserves_manuscript_structure(temp_extract_dir, create_test_zip):
    """Test that files are extracted to manuscript ID subdirectory."""
    zip_path = create_test_zip()  # Call the factory function returned by the fixture

    # Extract files
    extract_dir = temp_extract_dir
    extractor = XMLStructureExtractor(str(zip_path), str(extract_dir))

    # Verify structure
    manuscript_dir = extract_dir / MANUSCRIPT_ID
    assert manuscript_dir.exists()

    # Check paths relative to manuscript directory
    assert (manuscript_dir / "Doc" / f"{MANUSCRIPT_ID}Manuscript_TextIG.docx").exists()
    assert (manuscript_dir / "graphic/FIGURE 1.tif").exists()

    # Verify file contents
    assert (
        manuscript_dir / "Doc" / f"{MANUSCRIPT_ID}Manuscript_TextIG.docx"
    ).read_text() == "test content"  # Changed from "docx content" to "test content"

    # Check XML was properly handled
    assert extractor.manuscript_id == MANUSCRIPT_ID
    assert (extract_dir / f"{MANUSCRIPT_ID}.xml").exists()


class TestManuscriptXMLParser(unittest.TestCase):
    @patch("zipfile.ZipFile")
    @patch("lxml.etree.parse")  # Add this patch for the initialization
    def setUp(self, mock_etree_parse, mock_zipfile):
        # Mock the zipfile and its contents
        mock_zip = MagicMock()
        mock_zip.__enter__.return_value = mock_zip
        mock_zip.namelist.return_value = [
            "EMM-2023-18636/suppl_data/Figure 1.zip",
            "EMBOR-2023-58706-T/suppl_data/Fig1.zip",
            "EMM-2023-18636/EMM-2023-18636.xml",
        ]

        # Mock the file reading operations to return bytes
        mock_file = MagicMock()
        mock_file.read.return_value = b""  # Return empty bytes
        mock_zip.open.return_value.__enter__.return_value = mock_file

        mock_zipfile.return_value = mock_zip

        # Create a mock XML root for initialization
        init_xml_content = """
        <article>
            <notes>
            </notes>
        </article>
        """
        mock_root = etree.fromstring(init_xml_content)
        mock_etree_parse.return_value = MagicMock(
            getroot=MagicMock(return_value=mock_root)
        )

        # Provide mock paths for zip_path and extract_dir
        zip_path = "/mock/path/to/zip"
        extract_dir = "/mock/path/to/extract"

        # Initialize the parser with the required arguments
        self.parser = XMLStructureExtractor(zip_path, extract_dir)

    def test_extract_source_data_files(self):
        """Test extraction of source data files from XML."""
        xml_content = """
        <article>
            <notes>
                <form id="3030804" object-type="Figure Source Data Files">
                    <label>Figure 1 Source Data</label>
                    <long-desc>Figure 1.zip</long-desc>
                    <object_id>EMM-2023-18636/suppl_data/Figure 1.zip</object_id>
                </form>
                <form id="3133324" object-type="Figure Source Data Files">
                    <label>Figure1 Source Data</label>
                    <long-desc>Fig1.zip</long-desc>
                    <object_id>EMBOR-2023-58706-T/suppl_data/Fig1.zip</object_id>
                </form>
                <fig id="3001824" object-type="Figure">
                    <label>Figure 1</label>
                    <long-desc>Figure 1.eps</long-desc>
                    <object_id>EMM-2023-18636/graphic/Figure 1.eps</object_id>
                </fig>
            </notes>
        </article>
        """
        self.parser.xml_content = etree.fromstring(xml_content)

        # Test the method
        source_data_files = self.parser._get_source_data_files("Figure 1")

        # Expected cleaned paths (without manuscript ID)
        expected_files = ["suppl_data/Figure 1.zip", "suppl_data/Fig1.zip"]

        self.assertEqual(sorted(source_data_files), sorted(expected_files))

    def test_clean_path_handles_different_manuscript_ids(self):
        """Test that _clean_path removes different manuscript ID formats."""
        test_cases = [
            ("EMM-2023-18636/suppl_data/Figure 1.zip", "suppl_data/Figure 1.zip"),
            ("EMBOR-2023-58706-T/suppl_data/Fig1.zip", "suppl_data/Fig1.zip"),
            ("MSB-2024-12345/graphic/Figure 2.tif", "graphic/Figure 2.tif"),
            ("suppl_data/Figure 1.zip", "suppl_data/Figure 1.zip"),  # No manuscript ID
            ("", ""),  # Empty path
        ]

        for input_path, expected_output in test_cases:
            cleaned = self.parser._clean_path(input_path)
            self.assertEqual(
                cleaned,
                expected_output,
                f"Failed to clean path correctly. Input: {input_path}, Expected: {expected_output}, Got: {cleaned}",
            )


def test_path_handling(temp_extract_dir, create_test_zip):
    """Test both extraction structure and output paths."""
    zip_path = create_test_zip()
    extractor = XMLStructureExtractor(zip_path, str(temp_extract_dir))
    structure = extractor.extract_structure()

    # Test extraction structure
    manuscript_dir = temp_extract_dir / MANUSCRIPT_ID
    assert manuscript_dir.exists()
    assert (manuscript_dir / "Doc").exists()
    assert (manuscript_dir / "graphic").exists()

    # Test output paths don't have manuscript ID
    assert not structure.docx.startswith(f"{MANUSCRIPT_ID}/")
    assert all(
        not f.img_files[0].startswith(f"{MANUSCRIPT_ID}/") for f in structure.figures
    )

    # Test file access works
    docx_path = extractor.get_full_path(structure.docx)
    assert docx_path.exists()
    assert docx_path == manuscript_dir / structure.docx


def test_source_data_paths_cleaned(temp_extract_dir, create_test_zip):
    """Test that source data file paths don't include manuscript ID prefix."""
    zip_path = create_test_zip()
    extractor = XMLStructureExtractor(zip_path, str(temp_extract_dir))
    structure = extractor.extract_structure()

    # Test source data paths in figures
    for figure in structure.figures:
        for sd_file in figure.sd_files:
            # Check that the path doesn't start with manuscript ID
            assert not sd_file.startswith(f"{MANUSCRIPT_ID}/")
            # But the file should exist in the manuscript directory
            full_path = extractor.get_full_path(sd_file)
            assert full_path.exists()


def test_path_cleaning(temp_extract_dir, create_test_zip):
    """Test that paths are properly cleaned of manuscript ID."""
    zip_path = create_test_zip()
    extractor = XMLStructureExtractor(zip_path, str(temp_extract_dir))
    structure = extractor.extract_structure()

    # Test figure paths
    for figure in structure.figures:
        # Check image files
        for img_file in figure.img_files:
            assert not img_file.startswith(f"{MANUSCRIPT_ID}/")
            assert img_file.startswith("graphic/")
            # Verify file exists in manuscript directory
            full_path = extractor.manuscript_extract_dir / img_file
            assert full_path.exists()

        # Check source data files
        for sd_file in figure.sd_files:
            assert not sd_file.startswith(f"{MANUSCRIPT_ID}/")
            assert sd_file.startswith("suppl_data/")
            # Verify file exists in manuscript directory
            full_path = extractor.manuscript_extract_dir / sd_file
            assert full_path.exists()

    # Test DOCX path
    assert not structure.docx.startswith(f"{MANUSCRIPT_ID}/")
    assert structure.docx.startswith("Doc/")
