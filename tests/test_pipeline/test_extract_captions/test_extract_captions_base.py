import pytest
from soda_curation.pipeline.extract_captions.extract_captions_base import FigureCaptionExtractor
from soda_curation.pipeline.zip_structure.zip_structure_base import ZipStructure, Figure

class TestFigureCaptionExtractor(FigureCaptionExtractor):
    def extract_captions(self, file_path: str, zip_structure: ZipStructure) -> ZipStructure:
        return zip_structure

def test_update_zip_structure():
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
