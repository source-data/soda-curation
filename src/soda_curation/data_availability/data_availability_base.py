"""Base class for data availability extraction."""

from abc import ABC, abstractmethod
from typing import Dict


class DataAvailabilityExtractor(ABC):
    """Abstract base class for data availability extractors."""

    def __init__(self, config: Dict):
        """Initialize with configuration."""
        self.config = config

    @abstractmethod
    def extract_data_availability(self, doc_content: str) -> Dict:
        """
        Extract data availability information from document.

        Args:
            doc_content (str): Document content to analyze

        Returns:
            Dict: Dictionary containing:
                section_text (str): Complete text of data availability section
                data_sources (List[Dict]): List of structured data source information
        """
        pass

    @abstractmethod
    def _locate_data_availability_section(self, doc_content: str) -> str:
        """
        Locate and extract the data availability section text.

        Args:
            doc_content (str): Document content to analyze

        Returns:
            str: Complete text of data availability section or empty string if not found
        """
        pass

    @abstractmethod
    def _extract_data_records(self, section_text: str) -> list:
        """
        Extract structured data sources from data availability section.

        Args:
            section_text (str): Text of data availability section

        Returns:
            list: List of dictionaries containing structured data source information
        """
        pass
