from abc import ABC, abstractmethod
from typing import Dict, Any, List
from ..zip_structure.zip_structure_base import ZipStructure, Panel, Figure
from PIL import Image
import io
import base64
import logging

logger = logging.getLogger(__name__)

class MatchPanelCaption(ABC):
    """
    Abstract base class for matching panel captions with their corresponding images.
    """

    @abstractmethod
    def match_captions(self, zip_structure: ZipStructure) -> ZipStructure:
        """
        Abstract method to match captions to panels in a ZipStructure.

        Args:
            zip_structure (ZipStructure): The ZipStructure containing figures and panels.

        Returns:
            ZipStructure: The updated ZipStructure with matched captions.
        """
        pass

    def _extract_panel_image(self, figure_path: str, bbox: List[float]) -> str:
        """
        Extract a panel image from a figure based on bounding box coordinates.

        Args:
            figure_path (str): Path to the figure image file.
            bbox (List[float]): Bounding box coordinates [x1, y1, x2, y2] in relative format.

        Returns:
            str: Base64 encoded string of the panel image.
        """
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
