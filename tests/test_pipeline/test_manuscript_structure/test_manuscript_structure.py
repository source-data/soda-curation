"""
This module contains unit tests for the manuscript structure classes and functions.

It tests the creation and behavior of Panel, Figure, and ZipStructure objects,
as well as the custom JSON encoder used for serializing these objects.
"""

import json

import pytest

from soda_curation.pipeline.manuscript_structure.manuscript_structure import (
    CustomJSONEncoder,
    Figure,
    Panel,
    ZipStructure,
)


def test_panel_creation():
    """
    Test the creation of a Panel object.

    This test verifies that a Panel object is correctly instantiated with all its attributes,
    including the newly added confidence parameter.
    """
    panel = Panel("A", "This is panel A", [0.1, 0.1, 0.9, 0.9], confidence=0.95)
    assert panel.panel_label == "A"
    assert panel.panel_caption == "This is panel A"
    assert panel.panel_bbox == [0.1, 0.1, 0.9, 0.9]
    assert panel.confidence == 0.95


def test_figure_creation():
    """
    Test the creation of a Figure object.

    This test ensures that a Figure object is properly instantiated with all its attributes,
    including a list of Panel objects.
    """
    figure = Figure("Figure 1", ["image1.png"], ["data1.xlsx"], "This is Figure 1", 
                    panels=[Panel("A", "Panel A caption", [0.1, 0.1, 0.9, 0.9], confidence=0.95)])
    assert figure.figure_label == "Figure 1"
    assert figure.img_files == ["image1.png"]
    assert figure.sd_files == ["data1.xlsx"]
    assert figure.figure_caption == "This is Figure 1"
    assert len(figure.panels) == 1
    assert figure.panels[0].confidence == 0.95
    assert figure.duplicated_panels == "false"

def test_zip_structure_creation():
    """
    Test the creation of a ZipStructure object.

    This test verifies that a ZipStructure object is correctly instantiated with all its attributes,
    including a list of Figure objects and the new ai_response field.
    """
    zip_structure = ZipStructure(
        manuscript_id="test_manuscript",
        xml="test.xml",
        docx="test.docx",
        pdf="test.pdf",
        appendix=["appendix1.pdf"],
        figures=[Figure("Figure 1", ["image1.png"], ["data1.xlsx"], "This is Figure 1")],
        ai_response="Test AI response"
    )
    assert zip_structure.manuscript_id == "test_manuscript"
    assert zip_structure.xml == "test.xml"
    assert zip_structure.docx == "test.docx"
    assert zip_structure.pdf == "test.pdf"
    assert zip_structure.appendix == ["appendix1.pdf"]
    assert len(zip_structure.figures) == 1
    assert zip_structure.figures[0].figure_label == "Figure 1"
    assert zip_structure.ai_response == "Test AI response"

def test_custom_json_encoder():
    """
    Test the CustomJSONEncoder for serializing manuscript structure objects.

    This test ensures that Panel, Figure, and ZipStructure objects are correctly serialized to JSON
    and can be deserialized back into a dictionary with all the expected attributes.
    """
    panel = Panel("A", "This is panel A", [0.1, 0.1, 0.9, 0.9], confidence=0.95)
    figure = Figure("Figure 1", ["image1.png"], ["data1.xlsx"], "This is Figure 1", [panel])
    zip_structure = ZipStructure(
        manuscript_id="test_manuscript",
        xml="test.xml",
        docx="test.docx",
        pdf="test.pdf",
        appendix=["appendix1.pdf"],
        figures=[figure],
        ai_response="Test AI response"
    )
    
    encoded = json.dumps(zip_structure, cls=CustomJSONEncoder)
    decoded = json.loads(encoded)
    
    assert decoded["manuscript_id"] == "test_manuscript"
    assert decoded["figures"][0]["figure_label"] == "Figure 1"
    assert decoded["figures"][0]["panels"][0]["panel_label"] == "A"
    assert decoded["figures"][0]["panels"][0]["confidence"] == 0.95
    assert decoded["ai_response"] == "Test AI response"
    assert "ai_response" not in decoded["figures"][0]

def test_custom_json_encoder_unescape():
    """
    Test the unescape_string method of the CustomJSONEncoder.

    This test verifies that the unescape_string method correctly handles escaped characters
    in strings, particularly newline characters.
    """
    encoder = CustomJSONEncoder()
    unescaped = encoder.unescape_string("This is a test\\nwith new line")
    assert unescaped == "This is a test\nwith new line"
