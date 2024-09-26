import pytest
from soda_curation.pipeline.manuscript_structure.manuscript_structure import Panel, Figure, ZipStructure, CustomJSONEncoder
import json

def test_panel_creation():
    panel = Panel("A", "This is panel A", [0.1, 0.1, 0.9, 0.9])
    assert panel.panel_label == "A"
    assert panel.panel_caption == "This is panel A"
    assert panel.panel_bbox == [0.1, 0.1, 0.9, 0.9]

def test_figure_creation():
    figure = Figure("Figure 1", ["image1.png"], ["data1.xlsx"], "This is Figure 1")
    assert figure.figure_label == "Figure 1"
    assert figure.img_files == ["image1.png"]
    assert figure.sd_files == ["data1.xlsx"]
    assert figure.figure_caption == "This is Figure 1"
    assert figure.panels == []
    assert figure.duplicated_panels == "false"

def test_zip_structure_creation():
    zip_structure = ZipStructure(
        manuscript_id="test_manuscript",
        xml="test.xml",
        docx="test.docx",
        pdf="test.pdf",
        appendix=["appendix1.pdf"],
        figures=[Figure("Figure 1", ["image1.png"], ["data1.xlsx"], "This is Figure 1")]
    )
    assert zip_structure.manuscript_id == "test_manuscript"
    assert zip_structure.xml == "test.xml"
    assert zip_structure.docx == "test.docx"
    assert zip_structure.pdf == "test.pdf"
    assert zip_structure.appendix == ["appendix1.pdf"]
    assert len(zip_structure.figures) == 1
    assert zip_structure.figures[0].figure_label == "Figure 1"

def test_custom_json_encoder():
    panel = Panel("A", "This is panel A", [0.1, 0.1, 0.9, 0.9])
    figure = Figure("Figure 1", ["image1.png"], ["data1.xlsx"], "This is Figure 1", [panel])
    zip_structure = ZipStructure(
        manuscript_id="test_manuscript",
        xml="test.xml",
        docx="test.docx",
        pdf="test.pdf",
        appendix=["appendix1.pdf"],
        figures=[figure]
    )
    
    encoded = json.dumps(zip_structure, cls=CustomJSONEncoder)
    decoded = json.loads(encoded)
    
    assert decoded["manuscript_id"] == "test_manuscript"
    assert decoded["figures"][0]["figure_label"] == "Figure 1"
    assert decoded["figures"][0]["panels"][0]["panel_label"] == "A"

def test_custom_json_encoder_unescape():
    encoder = CustomJSONEncoder()
    unescaped = encoder.unescape_string("This is a test\\nwith new line")
    assert unescaped == "This is a test\nwith new line"
