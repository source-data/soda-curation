"""
This module contains unit tests for the base figure caption extraction functionality
in the soda_curation package.

It tests the core components of the caption extraction process that are common
across different AI implementations, focusing on the base FigureCaptionExtractor class.
"""

import pytest
from soda_curation.pipeline.extract_captions.extract_captions_base import FigureCaptionExtractor
from soda_curation.pipeline.manuscript_structure.manuscript_structure import ZipStructure, Figure

class TestFigureCaptionExtractor(FigureCaptionExtractor):
    """
    A test implementation of the FigureCaptionExtractor abstract base class.

    This class provides a concrete implementation of the abstract method
    for testing purposes. It simply returns the input ZipStructure without modification.
    """
    def extract_captions(self, file_path: str, zip_structure: ZipStructure) -> ZipStructure:
        """
        A dummy implementation of extract_captions for testing.

        Args:
            file_path (str): Path to the file (not used in this implementation).
            zip_structure (ZipStructure): The input ZipStructure.

        Returns:
            ZipStructure: The input ZipStructure without modification.
        """
        return zip_structure

def test_update_zip_structure():
    """
    Test the _update_zip_structure method of FigureCaptionExtractor.

    This test verifies that the _update_zip_structure method correctly updates
    the figure captions in a ZipStructure object when given a dictionary of captions.
    It checks that:
    1. Captions are correctly assigned to matching figures.
    2. The method handles cases where all figures have corresponding captions.
    """
    extractor = TestFigureCaptionExtractor()
    zip_structure = ZipStructure(
        manuscript_id="test",
        xml="test.xml",
        docx="test.docx",
        pdf="test.pdf",
        appendix=[],
        figures=[
            Figure("Figure 1", ["image1.png"], [], "TO BE ADDED IN LATER STEP", []),
            Figure("Figure 2", ["image2.png"], [], "TO BE ADDED IN LATER STEP", [])
        ]
    )
    captions = {
        "Figure 1": "This is caption for Figure 1",
        "Figure 2": "This is caption for Figure 2"
    }
    
    updated_structure = extractor._update_zip_structure(zip_structure, captions)
    
    assert updated_structure.figures[0].figure_caption == "This is caption for Figure 1"
    assert updated_structure.figures[1].figure_caption == "This is caption for Figure 2"

def test_update_zip_structure_missing_caption():
    """
    Test the _update_zip_structure method when a caption is missing.

    This test verifies that the _update_zip_structure method correctly handles cases
    where captions are not provided for all figures in the ZipStructure. It checks that:
    1. Captions are correctly assigned to figures with matching labels.
    2. Figures without corresponding captions retain their original caption.
    """
    extractor = TestFigureCaptionExtractor()
    zip_structure = ZipStructure(
        manuscript_id="test",
        xml="test.xml",
        docx="test.docx",
        pdf="test.pdf",
        appendix=[],
        figures=[
            Figure("Figure 1", ["image1.png"], [], "TO BE ADDED IN LATER STEP", []),
            Figure("Figure 2", ["image2.png"], [], "TO BE ADDED IN LATER STEP", [])
        ]
    )
    captions = {
        "Figure 1": "This is caption for Figure 1"
    }
    
    updated_structure = extractor._update_zip_structure(zip_structure, captions)
    
    assert updated_structure.figures[0].figure_caption == "This is caption for Figure 1"
    assert updated_structure.figures[1].figure_caption == "TO BE ADDED IN LATER STEP"

def test_update_zip_structure_empty_captions():
    """
    Test the _update_zip_structure method with an empty captions dictionary.

    This test verifies that the _update_zip_structure method correctly handles cases
    where an empty dictionary of captions is provided. It checks that:
    1. All figures in the ZipStructure retain their original captions.
    2. The method doesn't modify the ZipStructure when no new captions are provided.
    """
    extractor = TestFigureCaptionExtractor()
    zip_structure = ZipStructure(
        manuscript_id="test",
        xml="test.xml",
        docx="test.docx",
        pdf="test.pdf",
        appendix=[],
        figures=[
            Figure("Figure 1", ["image1.png"], [], "TO BE ADDED IN LATER STEP", []),
            Figure("Figure 2", ["image2.png"], [], "TO BE ADDED IN LATER STEP", [])
        ]
    )
    captions = {}
    
    updated_structure = extractor._update_zip_structure(zip_structure, captions)
    
    assert updated_structure.figures[0].figure_caption == "TO BE ADDED IN LATER STEP"
    assert updated_structure.figures[1].figure_caption == "TO BE ADDED IN LATER STEP"
