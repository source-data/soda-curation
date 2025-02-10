"""Base class for extracting data availability information from scientific manuscripts."""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List

from ..manuscript_structure.manuscript import ZipStructure

logger = logging.getLogger(__name__)


class DataAvailabilityExtractor(ABC):
    """
    Abstract base class for extracting data availability information.

    This class defines the interface that all data availability extractors must implement.
    It handles validation of configuration and provides a consistent structure for
    extracting both the data availability section and specific data sources.
    """

    def __init__(self, config: Dict, prompt_handler=None):
        """
        Initialize the extractor with configuration.

        Args:
            config: Configuration dictionary for the extractor
            prompt_handler: Optional handler for managing prompts
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
    def extract_data_availability(
        self, doc_content: str, zip_structure: ZipStructure
    ) -> ZipStructure:
        """
        Extract data availability information from document content.

        Args:
            doc_content: Document content to analyze
            zip_structure: Current ZIP structure to update

        Returns:
            Updated ZipStructure with data availability information
        """
        pass

    @abstractmethod
    def _locate_data_availability_section(self, doc_content: str) -> str:
        """
        Locate and extract the data availability section text.

        Args:
            doc_content: Document content to analyze

        Returns:
            Complete text of data availability section
        """
        pass

    @abstractmethod
    def _extract_data_sources(self, section_text: str) -> List[Dict]:
        """
        Extract structured data source information from section text.

        Args:
            section_text: Text of data availability section

        Returns:
            List of dictionaries containing structured data source information:
            [
                {
                    "database": str,
                    "accession_number": str,
                    "url": Optional[str]
                },
                ...
            ]
        """
        pass

    def _parse_response(self, response: str) -> List[Dict]:
        """Parse AI response containing data source information."""
        try:
            import json
            import re

            # Try to extract JSON from code block if present
            json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
            if json_match:
                response = json_match.group(1)
            else:
                # Try to extract any JSON object
                json_match = re.search(r"(\[.*\])", response, re.DOTALL)
                if json_match:
                    response = json_match.group(1)

            # Clean and normalize JSON string
            response = re.sub(r"[\n\r\t]", " ", response)
            response = re.sub(r"\s+", " ", response)

            return json.loads(response)

        except Exception as e:
            logger.error(f"Error parsing data sources: {str(e)}")
            return []
