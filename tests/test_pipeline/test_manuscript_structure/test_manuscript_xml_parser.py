import pytest
from unittest.mock import patch, Mock
from lxml import etree
from soda_curation.pipeline.manuscript_structure.manuscript_xml_parser import XMLStructureExtractor
import os

@pytest.fixture
def mock_xml_content():
    xml_content = """
    <article>
        <article-id pub-id-type="manuscript">TEST123</article-id>
        <supplementary-material object-type="Manuscript Text">
            <object_id>manuscript.docx</object_id>
        </supplementary-material>
        <merged_pdf object-type="Merged PDF">
            <object_id>manuscript.pdf</object_id>
        </merged_pdf>
        <fig object-type="Figure">
            <label>Figure 1</label>
            <object_id>figure1.png</object_id>
        </fig>
        <form object-type="Figure Source Data Files">
            <label>Figure 1 Source Data</label>
            <object_id>figure1_data.xlsx</object_id>
        </form>
        <form object-type="Expanded View Content (was Supplementary Information)">
            <label>Appendix</label>
            <object_id>appendix.pdf</object_id>
        </form>
    </article>
    """
    return etree.fromstring(xml_content)

@pytest.fixture
def mock_file_operations(mock_xml_content):
    with patch('os.listdir', return_value=['test.xml']), \
         patch('os.path.exists', return_value=True), \
         patch('lxml.etree.parse') as mock_parse:
        mock_parse.return_value.getroot.return_value = mock_xml_content
        yield mock_parse

@pytest.fixture
def xml_extractor(mock_file_operations, mock_xml_content):
    with patch('lxml.etree.parse') as mock_parse:
        mock_parse.return_value.getroot.return_value = mock_xml_content
        return XMLStructureExtractor('test.zip', '/tmp/extract')

def test_extract_xml_content(xml_extractor):
    assert xml_extractor.xml_content is not None

def test_get_manuscript_id(xml_extractor):
    assert xml_extractor.manuscript_id == "TEST123"

def test_clean_path(xml_extractor):
    assert xml_extractor._clean_path("TEST123/path/to/file.txt") == "path/to/file.txt"
    assert xml_extractor._clean_path("path/to/file.txt") == "path/to/file.txt"

def test_get_xml_file(xml_extractor):
    with patch('os.path.basename', return_value='test.xml'):
        assert xml_extractor._get_xml_file() == 'test.xml'

def test_get_docx_file(xml_extractor):
    assert xml_extractor._get_docx_file() == 'manuscript.docx'

def test_get_pdf_file(xml_extractor):
    assert xml_extractor._get_pdf_file() == 'manuscript.pdf'

def test_get_figures(xml_extractor):
    figures = xml_extractor._get_figures()
    assert len(figures) == 1
    assert figures[0].figure_label == "Figure 1"
    assert figures[0].img_files == ["figure1.png"]
    assert figures[0].sd_files == ["figure1_data.xlsx"]

def test_get_appendix(xml_extractor):
    assert xml_extractor._get_appendix() == ["appendix.pdf"]

def test_extract_structure(xml_extractor, mock_xml_content):
    with patch.object(xml_extractor, '_get_xml_file', return_value='test.xml'):
        structure = xml_extractor.extract_structure()
        assert structure.manuscript_id == "TEST123"
        assert structure.xml == "test.xml"
        assert structure.docx == "manuscript.docx"
        assert structure.pdf == "manuscript.pdf"
        assert len(structure.figures) == 1
        assert structure.appendix == ["appendix.pdf"]
