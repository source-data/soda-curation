"""Base class for extracting manuscript sections."""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Tuple

from pydantic import BaseModel

from ..manuscript_structure.manuscript_structure import ZipStructure

logger = logging.getLogger(__name__)


class ExtractedSections(BaseModel):
    """Model for extracted sections from manuscript."""

    figure_legends: str
    data_availability: str


class ExtractedSectionsResponse(BaseModel):
    """Model for API response containing extracted sections."""

    sections: ExtractedSections


class SectionExtractor(ABC):
    """
    Abstract base class for extracting sections from manuscripts.

    This class defines the interface that all section extractors should implement.
    It provides common functionality for extracting figure legends and data availability
    sections from manuscript content.
    """

    def __init__(self, config: Dict, prompt_handler):
        """
        Initialize with configuration and prompt handler.

        Args:
            config: Configuration dictionary for the extractor
            prompt_handler: Handler for managing prompts
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
    def extract_sections(
        self,
        doc_content: str,
        zip_structure: ZipStructure,
    ) -> Tuple[str, str, ZipStructure]:
        """
        Extract figure legends and data availability sections from document content.

        Args:
            doc_content: Document content to analyze
            zip_structure: Current ZIP structure to update

        Returns:
            Tuple containing:
            - str: The figure legends section
            - str: The data availability section
            - ZipStructure: Updated structure with costs and responses
        """
        pass

    def _parse_response(self, response: str) -> Dict:
        """Parse AI response containing extracted sections."""
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
            logger.error(f"Error parsing extracted sections: {str(e)}")
            return {
                "figure_legends": "",
                "data_availability": "",
            }
