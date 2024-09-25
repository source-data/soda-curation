import os
from lxml import etree
from typing import List, Dict, Any
from ..zip_structure.zip_structure_base import ZipStructure, Figure
import logging

logger = logging.getLogger(__name__)

class XMLStructureExtractor:
    def __init__(self, zip_path: str, extract_dir: str):
        self.zip_path = zip_path
        self.extract_dir = extract_dir
        self.xml_content = self._extract_xml_content()
        self.manuscript_id = self._get_manuscript_id()

    def _extract_xml_content(self) -> etree._Element:
        xml_files = [f for f in os.listdir(self.extract_dir) if f.endswith('.xml')]
        if not xml_files:
            raise ValueError("No XML file found in the root directory of the extracted ZIP")
        xml_file = xml_files[0]
        xml_path = os.path.join(self.extract_dir, xml_file)
        return etree.parse(xml_path).getroot()

    def _get_manuscript_id(self) -> str:
        manuscript_id = self.xml_content.xpath("//article-id[@pub-id-type='manuscript']")
        return manuscript_id[0].text if manuscript_id else ""

    def _clean_path(self, path: str) -> str:
        # Remove the manuscript ID from the beginning of the path if it's there
        if self.manuscript_id and path.startswith(f"{self.manuscript_id}/"):
            path = path[len(self.manuscript_id)+1:]
        return path

    def extract_structure(self) -> ZipStructure:
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
            figures=figures
        )

    def _get_xml_file(self) -> str:
        return os.path.basename(self.xml_content.base)

    def _get_docx_file(self) -> str:
        docx = self.xml_content.xpath("//supplementary-material[@object-type='Manuscript Text']")
        if docx:
            return self._clean_path(docx[0].xpath(".//object_id")[0].text)
        return ""

    def _get_pdf_file(self) -> str:
        pdf = self.xml_content.xpath("//merged_pdf[@object-type='Merged PDF']")
        if pdf:
            return self._clean_path(pdf[0].xpath(".//object_id")[0].text)
        return ""

    def _get_figures(self) -> List[Figure]:
        figures = []
        for fig in self.xml_content.xpath("//fig[@object-type='Figure']"):
            label = fig.xpath(".//label")[0].text
            img_files = [self._clean_path(fig.xpath(".//object_id")[0].text)]
            sd_files = self._get_source_data_files(label)
            figures.append(Figure(figure_label=label, img_files=img_files, sd_files=sd_files, figure_caption="", panels=[]))
        return figures

    def _get_source_data_files(self, figure_label: str) -> List[str]:
        sd_files = self.xml_content.xpath(f"//form[@object-type='Figure Source Data Files'][label='{figure_label} Source Data']")
        dataset_files = self.xml_content.xpath(f"//form[@object-type='Data Set'][starts-with(label, 'Dataset')]")
        
        files = []
        if sd_files:
            files.extend([self._clean_path(sd.xpath(".//object_id")[0].text) for sd in sd_files])
        if dataset_files:
            files.extend([self._clean_path(ds.xpath(".//object_id")[0].text) for ds in dataset_files])
        return files

    def _get_appendix(self) -> List[str]:
        appendix = self.xml_content.xpath("//form[@object-type='Expanded View Content (was Supplementary Information)'][label='Appendix']")
        return [self._clean_path(app.xpath(".//object_id")[0].text) for app in appendix]
