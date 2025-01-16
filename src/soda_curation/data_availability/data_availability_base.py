"""
This module provides the base class for extracting data availability information from manuscripts.

It defines an abstract base class that all specific data availability extractor implementations
should inherit from, ensuring a consistent interface across different extraction methods.
"""

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

class DataRecord:
    """Class to represent a data record with database, accession number and URL."""
    
    def __init__(self, database: str, accession_number: str, url: str):
        self.database = database
        self.accession_number = accession_number 
        self.url = url

    def to_dict(self) -> Dict:
        return {
            "database": self.database,
            "accession_number": self.accession_number,
            "url": self.url
        }

class DataAvailabilityExtractor(ABC):
    """Abstract base class for extracting data availability information."""

    def __init__(self, config: Dict):
        """Initialize with configuration."""
        self.config = config
        logger.info(f"Initialized {self.__class__.__name__}")

    @abstractmethod
    def _extract_data_records(self, doc_content: str) -> List[DataRecord]:
        """
        Extract data records from the data availability section.

        Args:
            data_section: Text of the data availability section

        Returns:
            List of DataRecord objects containing the extracted information
        """
        pass

    def extract_data_availability(self, doc_content: str) -> List[Dict]:
        """
        Extract data availability information from document content.
        
        Args:
            doc_content (str): The document content in HTML format
            
        Returns:
            List[Dict]: List of data availability records with database, accession number and URL
        """
        try:
            # Extract records from the section
            records = self._extract_data_records(doc_content)
            
            # Convert to dictionary format
            return [record.to_dict() for record in records]

        except Exception as e:
            logger.error(f"Error extracting data availability: {str(e)}")
            return []