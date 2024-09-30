"""
This module provides functionality for parsing XML files containing manuscript structure information.

It includes a class that extracts various components of a manuscript from an XML file,
such as figures, appendices, and associated files. This parsed information is then
used to create a structured representation of the manuscript.
"""

import logging
import os
from typing import List

from lxml import etree

from ..manuscript_structure.manuscript_structure import Figure, ZipStructure

logger = logging.getLogger(__name__)


class XMLStructureExtractor:
    """
    A class for extracting manuscript structure information from XML files.

    This class parses XML files typically found in scientific manuscript submissions,
    extracting information about figures, appendices, and associated files to create
    a structured representation of the manuscript.

    Attributes:
        zip_path (str): Path to the ZIP file containing the manuscript files.
        extract_dir (str): Directory where the ZIP file contents are extracted.
        xml_content (etree._Element): Parsed XML content of the manuscript.
        manuscript_id (str): Unique identifier for the manuscript.
    """

    def __init__(self, zip_path: str, extract_dir: str):
        """
        Initialize the XMLStructureExtractor.

        Args:
            zip_path (str): Path to the ZIP file containing the manuscript files.
            extract_dir (str): Directory where the ZIP file contents are extracted.
        """
        self.zip_path = zip_path
        self.extract_dir = extract_dir
        self.xml_content = self._extract_xml_content()
        self.manuscript_id = self._get_manuscript_id()

    def _extract_xml_content(self) -> etree._Element:
        """
        Extract and parse the XML content from the manuscript files.

        Returns:
            etree._Element: Parsed XML content.

        Raises:
            ValueError: If no XML file is found in the extracted directory.
        """
        xml_files = [f for f in os.listdir(self.extract_dir) if f.endswith(".xml")]
        if not xml_files:
            raise ValueError(
                "No XML file found in the root directory of the extracted ZIP"
            )
        xml_file = xml_files[0]
        xml_path = os.path.join(self.extract_dir, xml_file)
        return etree.parse(xml_path).getroot()

    def _get_manuscript_id(self) -> str:
        """
        Extract the manuscript ID from the XML content.

        Returns:
            str: The manuscript ID, or an empty string if not found.
        """
        manuscript_id = self.xml_content.xpath(
            "//article-id[@pub-id-type='manuscript']"
        )
        return manuscript_id[0].text if manuscript_id else ""

    def _clean_path(self, path: str) -> str:
        """
        Clean the file path by removing the manuscript ID prefix if present.

        Args:
            path (str): The original file path.

        Returns:
            str: The cleaned file path.
        """
        # Remove the manuscript ID from the beginning of the path if it's there
        if self.manuscript_id and path.startswith(f"{self.manuscript_id}/"):
            path = path[len(self.manuscript_id) + 1 :]
        return path

    def extract_structure(self) -> ZipStructure:
        """
        Extract the complete structure of the manuscript from the XML content.

        This method processes the XML content to extract information about the manuscript,
        including its ID, associated files (XML, DOCX, PDF), figures, and appendices.

        Returns:
            ZipStructure: A structured representation of the manuscript.
        """
        xml_file = self._get_xml_file()
        docx_file = self._get_docx_file()
        pdf_file = self._get_pdf_file()
        figures = self._get_figures()
        appendix = self._get_appendix()

        return ZipStructure(
            manuscript_id=self.manuscript_id,
            xml=xml_file,
            docx=docx_file,
            pdf=pdf_file,
            appendix=appendix,
            figures=figures,
        )

    def _get_xml_file(self) -> str:
        """
        Get the name of the XML file.

        Returns:
            str: The name of the XML file.
        """
        return os.path.basename(self.xml_content.base)

    def _get_docx_file(self) -> str:
        """
        Extract the path to the DOCX file from the XML content.

        Returns:
            str: The path to the DOCX file, or an empty string if not found.
        """
        docx = self.xml_content.xpath(
            "//supplementary-material[@object-type='Manuscript Text']"
        )
        if docx:
            return self._clean_path(docx[0].xpath(".//object_id")[0].text)
        return ""

    def _get_pdf_file(self) -> str:
        """
        Extract the path to the PDF file from the XML content.

        Returns:
            str: The path to the PDF file, or an empty string if not found.
        """
        pdf = self.xml_content.xpath("//merged_pdf[@object-type='Merged PDF']")
        if pdf:
            return self._clean_path(pdf[0].xpath(".//object_id")[0].text)
        return ""

    def _get_figures(self) -> List[Figure]:
        """
        Extract information about figures from the XML content.

        This method processes the XML to identify figures, their labels, associated image files,
        and source data files.

        Returns:
            List[Figure]: A list of Figure objects representing the figures in the manuscript.
        """
        figures = []
        for fig in self.xml_content.xpath("//fig[@object-type='Figure']"):
            label = fig.xpath(".//label")[0].text
            img_files = [self._clean_path(fig.xpath(".//object_id")[0].text)]
            sd_files = self._get_source_data_files(label)
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
        """
        Extract paths to source data files associated with a specific figure.

        Args:
            figure_label (str): The label of the figure to find source data for.

        Returns:
            List[str]: A list of file paths to source data files for the given figure.
        """
        sd_files = self.xml_content.xpath(
            f"//form[@object-type='Figure Source Data Files'][label='{figure_label} Source Data']"
        )
        dataset_files = self.xml_content.xpath(
            "//form[@object-type='Data Set'][starts-with(label, 'Dataset')]"
        )

        files = []
        if sd_files:
            files.extend(
                [self._clean_path(sd.xpath(".//object_id")[0].text) for sd in sd_files]
            )
        if dataset_files:
            files.extend(
                [
                    self._clean_path(ds.xpath(".//object_id")[0].text)
                    for ds in dataset_files
                ]
            )
        return files

    def _get_appendix(self) -> List[str]:
        """
        Extract paths to appendix files from the XML content.

        Returns:
            List[str]: A list of file paths to appendix files.
        """
        appendix = self.xml_content.xpath(
            "//form[@object-type='Expanded View Content (was Supplementary Information)'][label='Appendix']"
        )
        return [self._clean_path(app.xpath(".//object_id")[0].text) for app in appendix]
