from abc import ABC, abstractmethod
import logging
from typing import Dict, Any
from ..zip_structure.zip_structure_base import ZipStructure

logger = logging.getLogger(__name__)

class FigureCaptionExtractor(ABC):
    """
    Abstract base class for extracting figure captions from a document.
    """

    @abstractmethod
    def extract_captions(self, docx_path: str, zip_structure: ZipStructure) -> ZipStructure:
        """
        Extract figure captions from the given DOCX file and update the ZipStructure.

        Args:
            docx_path (str): Path to the DOCX file.
            zip_structure (ZipStructure): The current ZIP structure.

        Returns:
            ZipStructure: Updated ZIP structure with extracted captions.
        """
        pass

    def _update_zip_structure(self, zip_structure: ZipStructure, captions: Dict[str, str]) -> ZipStructure:
        """
        Update the ZipStructure with extracted captions.

        Args:
            zip_structure (ZipStructure): The current ZIP structure.
            captions (Dict[str, str]): Dictionary of figure labels and their captions.

        Returns:
            ZipStructure: Updated ZIP structure.
        """
        for figure in zip_structure.figures:
            if figure.figure_label in captions:
                figure.figure_caption = captions[figure.figure_label]
                logger.debug(f"Updated caption for {figure.figure_label}")
            else:
                logger.warning(f"No caption found for {figure.figure_label}")
        return zip_structure