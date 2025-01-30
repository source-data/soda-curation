"""Tests for XMLStructureExtractor class."""

import os
import pytest
import shutil
from pathlib import Path
from zipfile import ZipFile
import lxml.etree as etree 

from src.soda_curation.pipeline.manuscript_structure.manuscript_structure import ZipStructure
from src.soda_curation.pipeline.manuscript_structure.manuscript_xml_parser import XMLStructureExtractor
from src.soda_curation.pipeline.manuscript_structure.exceptions import NoXMLFileFoundError, NoManuscriptFileError

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
def create_test_zip(tmp_path, sample_xml_content):
    """Create a test ZIP file with specified structure."""
    def _create_zip(include_files=None, use_invalid_xml=False):
        include_files = include_files or {'docx', 'pdf', 'figures', 'source_data'}
        zip_path = tmp_path / f"{MANUSCRIPT_ID}.zip"
        
        with ZipFile(zip_path, 'w') as zf:
            # Add XML file
            xml_content = INVALID_XML if use_invalid_xml else sample_xml_content
            zf.writestr(f"{MANUSCRIPT_ID}.xml", xml_content)
            
            # Add DOCX if requested
            if 'docx' in include_files:
                zf.writestr(
                    f"Doc/{MANUSCRIPT_ID}R1Manuscript_TextIG.docx", 
                    "docx content"
                )
            
            # Add PDF if requested
            if 'pdf' in include_files:
                zf.writestr(f"pdf/{MANUSCRIPT_ID}.pdf", "pdf content")
            
            # Add figures if requested
            if 'figures' in include_files:
                for i in range(1, 7):
                    zf.writestr(
                        f"graphic/FIGURE {i}.tif",
                        b"image content"
                    )
                    
            # Add source data if requested
            if 'source_data' in include_files:
                for i in [3, 4, 5, 6]:
                    zf.writestr(
                        f"suppl_data/{MANUSCRIPT_ID}Figure_{i}sd.zip",
                        b"source data"
                    )
                    
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

def test_missing_xml_file(temp_extract_dir):
    """Test handling of ZIP file without XML file."""
    # Create ZIP without XML
    zip_path = temp_extract_dir.parent / "test.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr("dummy.txt", "content")
    
    with pytest.raises(NoXMLFileFoundError):
        XMLStructureExtractor(str(zip_path), str(temp_extract_dir))

def test_empty_xml_file(temp_extract_dir, create_test_zip):
    """Test handling of empty XML file."""
    zip_path = temp_extract_dir.parent / "test.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr(f"{MANUSCRIPT_ID}.xml", "")
    
    with pytest.raises(etree.XMLSyntaxError):
        XMLStructureExtractor(zip_path, str(temp_extract_dir))

def test_non_xml_file(temp_extract_dir, create_test_zip):
    """Test handling of non-XML file with .xml extension."""
    zip_path = temp_extract_dir.parent / "test.zip"
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr(f"{MANUSCRIPT_ID}.xml", "This is not XML content")
    
    with pytest.raises(etree.XMLSyntaxError):
        XMLStructureExtractor(zip_path, str(temp_extract_dir))

def test_invalid_xml_content(temp_extract_dir, create_test_zip):
    """Test handling of invalid XML content."""
    zip_path = create_test_zip(use_invalid_xml=True)
    
    with pytest.raises(etree.XMLSyntaxError):
        extractor = XMLStructureExtractor(zip_path, str(temp_extract_dir))

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
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr(f"{MANUSCRIPT_ID}.xml", modified_xml)
        zf.writestr(f"Doc/{MANUSCRIPT_ID}Manuscript_TextIG.docx", "docx content")
    
    extractor = XMLStructureExtractor(zip_path, str(temp_extract_dir))
    with pytest.raises(NoManuscriptFileError, match="No DOCX file referenced in XML"):
        extractor.extract_structure()

def test_docx_missing_from_zip(temp_extract_dir, create_test_zip):
    """Test case where DOCX is referenced in XML but missing from ZIP."""
    # Use normal XML (which references DOCX) but don't include DOCX in ZIP
    zip_path = create_test_zip(include_files={'pdf', 'figures'})  # Explicitly exclude DOCX
    
    extractor = XMLStructureExtractor(zip_path, str(temp_extract_dir))
    with pytest.raises(NoManuscriptFileError, match="DOCX file referenced in XML not found in ZIP"):
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
    with ZipFile(zip_path, 'w') as zf:
        zf.writestr(f"{MANUSCRIPT_ID}.xml", modified_xml)
        zf.writestr(f"Doc/{MANUSCRIPT_ID}Manuscript_TextIG.docx", "docx content")
    
    extractor = XMLStructureExtractor(zip_path, str(temp_extract_dir))
    with pytest.raises(NoManuscriptFileError, match="DOCX file path in XML does not match ZIP structure"):
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