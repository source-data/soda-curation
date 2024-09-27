"""
This module provides the base class for matching panel captions with their corresponding images.

It defines an abstract base class that all specific panel caption matching implementations
should inherit from, ensuring a consistent interface across different matching methods.
"""
import base64
import io
import logging
from abc import ABC, abstractmethod
from typing import List

from PIL import Image

from ..manuscript_structure.manuscript_structure import ZipStructure

logger = logging.getLogger(__name__)


class MatchPanelCaption(ABC):
    """
    Abstract base class for matching panel captions with their corresponding images.

    This class defines the interface that all panel caption matching implementations should follow.
    It provides a common structure for matching captions to panels and updating the ZipStructure.
    """

    @abstractmethod
    def match_captions(self, zip_structure: ZipStructure) -> ZipStructure:
        """
        Match captions to panels in a ZipStructure.

        This method should be implemented by subclasses to define the specific
        panel caption matching logic for different AI models or approaches.

        Args:
            zip_structure (ZipStructure): The ZipStructure containing figures and panels.

        Returns:
            ZipStructure: The updated ZipStructure with matched captions.
        """
        pass

    def _extract_panel_image(self, figure_path: str, bbox: List[float]) -> str:
        """
        Extract a panel image from a figure based on bounding box coordinates.

        This method opens the figure image, crops it according to the bounding box,
        and returns the panel image as a base64 encoded string.

        Args:
            figure_path (str): Path to the figure image file.
            bbox (List[float]): Bounding box coordinates [x1, y1, x2, y2] in relative format.

        Returns:
            str: Base64 encoded string of the panel image.

        Raises:
            Exception: If there's an error during image extraction or encoding.
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
