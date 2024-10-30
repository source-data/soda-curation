"""
This module provides functionality for parsing XML files containing manuscript structure information.

It includes a class that extracts various components of a manuscript from an XML file,
such as figures, appendices, and associated files. This parsed information is then
used to create a structured representation of the manuscript.
"""

import logging
import os
from typing import List, Optional

from lxml import etree

from .exceptions import NoManuscriptFileError, NoXMLFileFoundError
from .manuscript_structure import Figure, ZipStructure

logger = logging.getLogger(__name__)

class XMLStructureExtractor:
    """
    A class for extracting manuscript structure information from XML files.
    
    Handles multiple XML formats and tag variations for manuscript components.
    """

    # Define constants for XML attributes and tags
    MANUSCRIPT_TEXT_TYPES = ["Manuscript Text"]
    FIGURE_TYPES = ["Figure"]
    EXPANDED_VIEW_TYPES = ["Expanded View Content (was Supplementary Information)"]
    SOURCE_DATA_TYPES = ["Figure Source Data Files"]
    DATA_SET_TYPES = ["Data Set"]

    def __init__(self, zip_path: str, extract_dir: str):
        """Initialize the extractor with paths."""
        self.zip_path = zip_path
        self.extract_dir = extract_dir
        self.xml_content = self._extract_xml_content()
        self.manuscript_id = self._get_manuscript_id()
        logger.info(f"Initialized XMLStructureExtractor with manuscript_id: {self.manuscript_id}")

    def _extract_xml_content(self) -> etree._Element:
        """Extract and parse XML content."""
        xml_files = [f for f in os.listdir(self.extract_dir) if f.endswith(".xml")]
        if not xml_files:
            raise NoXMLFileFoundError("No XML file found in the root directory")
        xml_file = xml_files[0]
        xml_path = os.path.join(self.extract_dir, xml_file)
        logger.info(f"Parsing XML file: {xml_path}")
        return etree.parse(xml_path).getroot()

    def _get_manuscript_id(self) -> str:
        """Extract manuscript ID from XML."""
        manuscript_id = self.xml_content.xpath("//article-id[@pub-id-type='manuscript']")
        return manuscript_id[0].text if manuscript_id else ""

    def _get_docx_file(self) -> Optional[str]:
        """
        Get the DOCX manuscript file path.
        
        Checks multiple possible XML structures for manuscript text documents.
        """
        # Build XPath to find manuscript text elements of various types
        xpath_query = " | ".join([
            f"//doc[@object-type='{type}']" for type in self.MANUSCRIPT_TEXT_TYPES
        ] + [
            f"//supplementary-material[@object-type='{type}']" for type in self.MANUSCRIPT_TEXT_TYPES
        ])
        
        manuscript_elements = self.xml_content.xpath(xpath_query)
        
        for element in manuscript_elements:
            # Get object_id element content
            object_id = element.xpath(".//object_id")
            if object_id and object_id[0].text.lower().endswith('.docx'):
                return self._clean_path(object_id[0].text)
        
        logger.warning("No DOCX manuscript file found")
        return None

    def _get_figures(self) -> List[Figure]:
        """Get list of figures from XML."""
        figures = []
        # Build XPath to find figure elements
        xpath_query = " | ".join([
            f"//fig[@object-type='{type}']" for type in self.FIGURE_TYPES
        ])
        
        for fig in self.xml_content.xpath(xpath_query):
            label = fig.xpath(".//label")[0].text
            # Skip EV figures as they go to appendix
            if "EV" in label:
                continue
                
            img_files = [self._clean_path(fig.xpath(".//object_id")[0].text)]
            sd_files = self._get_source_data_files(label)
            
            logger.info(f"Processing figure: {label}")
            figures.append(
                Figure(
                    figure_label=label,
                    img_files=img_files,
                    sd_files=sd_files,
                    figure_caption="",
                    panels=[],
                )
            )
        return figures

    def _get_source_data_files(self, figure_label: str) -> List[str]:
        """Get source data files for a specific figure."""
        # Build XPath for source data elements
        xpath_query = " | ".join([
            f"//form[@object-type='{type}'][label='{figure_label} Source Data']" 
            for type in self.SOURCE_DATA_TYPES
        ])
        
        sd_files = self.xml_content.xpath(xpath_query)
        
        # Also check for associated datasets
        dataset_query = " | ".join([
            f"//form[@object-type='{type}']" for type in self.DATA_SET_TYPES
        ])
        dataset_files = self.xml_content.xpath(dataset_query)
        
        files = []
        if sd_files:
            files.extend([self._clean_path(sd.xpath(".//object_id")[0].text) for sd in sd_files])
        if dataset_files:
            files.extend([self._clean_path(ds.xpath(".//object_id")[0].text) for ds in dataset_files])
        
        return files

    def _get_appendix(self) -> List[str]:
        """Get list of appendix files."""
        # Build XPath for expanded view content
        xpath_query = " | ".join([
            f"//form[@object-type='{type}'][label='Appendix']" 
            for type in self.EXPANDED_VIEW_TYPES
        ])
        
        appendix = self.xml_content.xpath(xpath_query)
        return [self._clean_path(app.xpath(".//object_id")[0].text) for app in appendix]

    def _clean_path(self, path: str) -> str:
        """Clean file path by removing manuscript ID prefix if present."""
        if self.manuscript_id and path.startswith(f"{self.manuscript_id}/"):
            path = path[len(self.manuscript_id) + 1:]
        return path

    def extract_structure(self) -> ZipStructure:
        """
        Extract complete manuscript structure from XML.
        
        Returns:
            ZipStructure: Structured representation of the manuscript.
            
        Raises:
            NoManuscriptFileError: If no manuscript file (DOCX/PDF) is found.
        """
        logger.info("Extracting structure from XML")
        
        xml_file = self._get_xml_file()
        docx_file = self._get_docx_file()
        pdf_file = self._get_pdf_file()

        if not docx_file and not pdf_file:
            raise NoManuscriptFileError("No PDF or DOCX manuscript file found")

        figures = self._get_figures()
        appendix = self._get_appendix()

        return ZipStructure(
            manuscript_id=self.manuscript_id,
            xml=xml_file,
            docx=docx_file or "",
            pdf=pdf_file or "",
            appendix=appendix,
            figures=figures,
        )

    def _get_xml_file(self) -> str:
        """Get name of the XML file."""
        return os.path.basename(self.xml_content.base)

    def _get_pdf_file(self) -> str:
        """Get PDF file path from XML."""
        pdf = self.xml_content.xpath("//merged_pdf[@object-type='Merged PDF']")
        if pdf:
            return self._clean_path(pdf[0].xpath(".//object_id")[0].text)
        logger.warning("No PDF file found")
        return ""
