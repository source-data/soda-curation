from abc import ABC, abstractmethod
from typing import Dict, Any, List
from ..zip_structure.zip_structure_base import ZipStructure, Panel
import logging

logger = logging.getLogger(__name__)

class MatchPanelCaption(ABC):
    """
    Abstract base class for matching panel captions with their corresponding images.
    """

    @abstractmethod
    def match_captions(self, zip_structure: ZipStructure) -> ZipStructure:
        """
        Match panel captions with their corresponding images.

        Args:
            zip_structure (ZipStructure): The current ZIP structure with figures and panels.

        Returns:
            ZipStructure: Updated ZIP structure with matched panel captions.
        """
        pass

    def _update_zip_structure(self, zip_structure: ZipStructure, matched_captions: Dict[str, List[Panel]]) -> ZipStructure:
        """
        Update the ZipStructure with matched panel captions.

        Args:
            zip_structure (ZipStructure): The current ZIP structure.
            matched_captions (Dict[str, List[Panel]]): Dictionary of figure labels and their updated panels.

        Returns:
            ZipStructure: Updated ZIP structure.
        """
        for figure in zip_structure.figures:
            if figure.figure_label in matched_captions:
                figure.panels = matched_captions[figure.figure_label]
            else:
                logger.warning(f"No matched captions found for {figure.figure_label}")
        return zip_structure

    def _extract_panel_image(self, figure_path: str, bbox: List[float]) -> str:
        try:
            with Image.open(figure_path) as img:
                width, height = img.size
                left, top, right, bottom = [
                    int(coord * width if i % 2 == 0 else coord * height)
                    for i, coord in enumerate(bbox)
                ]
                panel = img.crop((left, top, right, bottom))
                
                buffered = io.BytesIO()
                panel.save(buffered, format="PNG")
                return base64.b64encode(buffered.getvalue()).decode("utf-8")
        except Exception as e:
            logger.error(f"Error extracting panel image: {str(e)}")
            return ""
