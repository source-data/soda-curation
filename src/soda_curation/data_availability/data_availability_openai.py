"""
This module provides OpenAI-specific implementation for extracting data availability information.

It implements the DataAvailabilityExtractor interface using OpenAI's GPT models to locate
and extract data availability information from manuscripts.
"""

import json
import logging
import re
from typing import Dict, List, Optional

import openai
from bs4 import BeautifulSoup

from .data_availability_base import DataAvailabilityExtractor, DataRecord
from .data_availability_prompts import (
    DATA_AVAILABILITY_PROMPT,
    get_data_availability_prompt,
)

logger = logging.getLogger(__name__)

class DataAvailabilityExtractorGPT(DataAvailabilityExtractor):
    """Implementation of data availability extraction using OpenAI's GPT models."""

    def __init__(self, config: Dict):
        """Initialize with OpenAI configuration."""
        super().__init__(config)
        self.client = openai.OpenAI(api_key=self.config["openai"]["api_key"])
        self.model = self.config["openai"].get("model", "gpt-4-1106-preview")
        logger.info(f"Initialized with model: {self.model}")

    def _extract_data_records(self, doc_content: str) -> List[DataRecord]:
        """
        Extract data records from the data availability section.
        
        Args:
            doc_content (str): The identified data availability section text
            
        Returns:
            List[DataRecord]: List of data records with structured information
        """
        try:            
            # Prepare prompt with the cleaned text
            prompt = get_data_availability_prompt(doc_content)

            # Call GPT API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": DATA_AVAILABILITY_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
            )

            if not response.choices:
                return []

            # Parse response
            try:
                content = response.choices[0].message.content.strip()
                
                # Clean up code blocks if present
                if content.startswith('```') and content.endswith('```'):
                    content = '\n'.join(content.split('\n')[1:-1])
                content = re.sub(r'^json\n', '', content)
                
                # Parse JSON
                records_data = json.loads(content)
                records = []
                
                for record_data in records_data:
                    database = record_data.get("database", "").strip()
                    accession = record_data.get("accession_number", "").strip()
                    url = record_data.get("url", "").strip()
                    
                    if database and accession:
                        records.append(DataRecord(
                            database=database,
                            accession_number=accession,
                            url=url or ""  # Use empty string if no URL
                        ))

                return records

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse GPT response as JSON: {str(e)}")
                return []

        except Exception as e:
            logger.error(f"Error extracting data records: {str(e)}")
            return []