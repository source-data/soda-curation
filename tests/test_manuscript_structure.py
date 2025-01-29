"""Tests for XMLStructureExtractor class."""

import os
import pytest
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET
from zipfile import ZipFile

from src.soda_curation.pipeline.manuscript_structure.manuscript_structure import ZipStructure
from src.soda_curation.pipeline.manuscript_structure.manuscript_xml_parser import XMLStructureExtractor
from src.soda_curation.pipeline.manuscript_structure.exceptions import NoXMLFileFoundError, NoManuscriptFileError

# Constants for tests
MANUSCRIPT_ID = "EMBOJ-DUMMY-ZIP"

@pytest.fixture
def test_data_dir():
    """Get path to test data directory."""
    return Path(__file__).parent / "data"

@pytest.fixture
def sample_xml_content(test_data_dir):
    """Load sample XML content from test file."""
    xml_path = test_data_dir / "EMBOJ-DUMMY-ZIP.xml"
    return xml_path.read_text()

@pytest.fixture
def invalid_xml_content():
    """Create invalid XML content."""
    return """<?xml version="1.0" encoding="UTF-8"?>
    <invalid>
        <unclosed>
    </invalid>"""

@pytest.fixture
def temp_extract_dir(tmp_path):
    """Create temporary directory for extraction."""
    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    yield extract_dir
    if extract_dir.exists():
        shutil.rmtree(extract_dir)

@pytest.fixture
def create_test_zip(tmp_path, sample_xml_content):
    """Create a test ZIP file with specified structure."""
    def _create_zip(include_files=None, use_invalid_xml=False):
        """
        Create ZIP with specified files.
        
        Args:
            include_files (set): Set of file types to include ('docx', 'pdf', 'figures', 'source_data')
            use_invalid_xml (bool): Whether to use invalid XML content
        """
        include_files = include_files or {'docx', 'pdf', 'figures', 'source_data'}
        zip_path = tmp_path / f"{MANUSCRIPT_ID}.zip"
        
        with ZipFile(zip_path, 'w') as zf:
            # Add XML file
            xml_content = invalid_xml_content() if use_invalid_xml else sample_xml_content
            zf.writestr(f"{MANUSCRIPT_ID}.xml", xml_content)
            
            # Add DOCX if requested
            if 'docx' in include_files:
                zf.writestr(f"Doc/{MANUSCRIPT_ID}Manuscript_TextIG.docx", "docx content")
            
            # Add PDF if requested
            if 'pdf' in include_files:
                zf.writestr(f"pdf/{MANUSCRIPT_ID}.pdf", "pdf content")
            
            # Add figures if requested
            if 'figures' in include_files:
                for i in range(1, 7):
                    zf.writestr(f"graphic/FIGURE {i}.tif", b"image content")
                    
            # Add source data if requested
            if 'source_data' in include_files:
                for i in [3, 4, 5, 6]:
                    zf.writestr(f"suppl_data/{MANUSCRIPT_ID}Figure_{i}sd.zip", b"source data")
                    
            # Add appendix and other supplementary files
            zf.writestr("suppl_data/Appendix.pdf", "appendix content")
            zf.writestr(f"suppl_data/{MANUSCRIPT_ID}Peer Review File.pdf", "review content")
            zf.writestr(f"suppl_data/{MANUSCRIPT_ID}SourceData_Checklist.pdf", "checklist content")
            
            # Add production forms
            zf.writestr(f"prod_forms/{MANUSCRIPT_ID}Synopsis.docx", "synopsis content")
            zf.writestr(f"prod_forms/{MANUSCRIPT_ID}_eTOC_Blurb.docx", "blurb content")
            zf.writestr(f"prod_forms/{MANUSCRIPT_ID}Synopsis_Image.png", b"synopsis image")
            zf.writestr(f"prod_forms/{MANUSCRIPT_ID}eToC.png", b"etoc image")
        
        return str(zip_path)
    
    return _create_zip

def test_missing_xml_file(temp_extract_dir, create_test_zip):
    """Test handling of ZIP file without XML file."""
    # Create ZIP without any files
    zip_path = create_test_zip(include_files=set())
    
    with pytest.raises(NoXMLFileFoundError):
        XMLStructureExtractor(zip_path, str(temp_extract_dir))

def test_invalid_xml_content(temp_extract_dir, create_test_zip):
    """Test handling of invalid XML content."""
    zip_path = create_test_zip(use_invalid_xml=True)
    
    with pytest.raises(ET.ParseError):
        extractor = XMLStructureExtractor(zip_path, str(temp_extract_dir))
        extractor.extract_structure()

def test_missing_docx_file(temp_extract_dir, create_test_zip):
    """Test handling of ZIP file without DOCX manuscript."""
    zip_path = create_test_zip(include_files={'pdf', 'figures'})
    
    with pytest.raises(NoManuscriptFileError):
        extractor = XMLStructureExtractor(zip_path, str(temp_extract_dir))
        extractor.extract_structure()

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
    assert structure.pdf.endswith(".pdf")
    
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
            
    # Verify appendix
    assert len(structure.appendix) >= 1
    assert any("Appendix.pdf" in f for f in structure.appendix)

def test_minimal_structure(temp_extract_dir, create_test_zip):
    """Test extraction of minimal structure with only required files."""
    zip_path = create_test_zip(include_files={'docx'})
    
    extractor = XMLStructureExtractor(zip_path, str(temp_extract_dir))
    structure = extractor.extract_structure()
    
    assert isinstance(structure, ZipStructure)
    assert structure.docx.endswith(".docx")
    assert not structure.figures
    assert not structure.appendix
    assert not structure.pdf

def test_root_path_handling(temp_extract_dir, create_test_zip):
    """Test handling of files with and without manuscript ID prefix."""
    # Test with prefix
    zip_path = create_test_zip()
    extractor = XMLStructureExtractor(zip_path, str(temp_extract_dir))
    structure = extractor.extract_structure()
    
    # Verify paths don't have duplicate manuscript ID
    assert not structure.docx.startswith(f"{MANUSCRIPT_ID}/{MANUSCRIPT_ID}")
    assert all(not f.img_files[0].startswith(f"{MANUSCRIPT_ID}/{MANUSCRIPT_ID}") 
              for f in structure.figures)
    
    # Verify paths are correctly normalized
    assert all(not path.startswith("/") for path in structure.appendix)
    assert all(not fig.img_files[0].startswith("/") for fig in structure.figures)

def test_figure_source_data_matching(temp_extract_dir, create_test_zip):
    """Test correct matching of source data files to figures."""
    zip_path = create_test_zip(include_files={'docx', 'figures', 'source_data'})
    
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