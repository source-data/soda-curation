"""Base class for figure caption extraction."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from pydantic import BaseModel

from ..manuscript_structure.manuscript_structure import ZipStructure
from ..prompt_handler import PromptHandler

logger = logging.getLogger(__name__)


class PanelList(BaseModel):
    """Model for a list of panels."""

    panel_label: str
    panel_caption: str


class IndividualCaption(BaseModel):
    """Model for individual figure caption."""

    figure_label: str
    caption_title: str
    figure_caption: str
    panels: List[PanelList]


class ExtractedCaptions(BaseModel):
    """Model for extracted captions response."""

    figures: List[IndividualCaption]


class FigureCaptionExtractor(ABC):
    """
    Abstract base class for extracting figure captions from a document.

    This class provides common functionality and defines the interface that all
    figure caption extractors should implement.
    """

    def __init__(self, config: Dict[str, Any], prompt_handler: PromptHandler):
        """
        Initialize with configuration and prompt handler.

        Args:
            config: Configuration dictionary for the extractor
            prompt_handler: Handler for managing prompts

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self.prompt_handler = prompt_handler
        self._validate_config()

    @abstractmethod
    def _validate_config(self) -> None:
        """
        Validate the configuration specific to each implementation.

        Raises:
            ValueError: If configuration is invalid
        """
        pass

    @abstractmethod
    def extract_individual_captions(
        self,
        doc_content: str,
        zip_structure: ZipStructure,
    ) -> ZipStructure:
        """
        Extract captions from the figure legends section and update ZipStructure.

        Args:
            doc_content: Figure legends section content to analyze
            zip_structure: Current ZIP structure to update

        Returns:
            Updated ZipStructure with extracted captions
        """
        pass

    def _parse_response(self, response: str) -> Dict:
        """Parse AI response containing caption data."""
        try:
            import json
            import re

            # Try to extract JSON from code block if present
            json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
            if json_match:
                response = json_match.group(1)
            else:
                # Try to extract any JSON object
                json_match = re.search(r"(\{.*\})", response, re.DOTALL)
                if json_match:
                    response = json_match.group(1)

            # Clean and normalize JSON string
            response = re.sub(r"[\n\r\t]", " ", response)
            response = re.sub(r"\s+", " ", response)

            return json.loads(response)

        except Exception as e:
            logger.error(f"Error parsing captions: {str(e)}")
            return {}
