from typing import Dict, List
import logging
import json
import openai
import re
from .data_availability_base import DataAvailabilityExtractor
from .data_availability_prompts import (
    SYSTEM_PROMPT_LOCATE,
    SYSTEM_PROMPT_EXTRACT,
    get_locate_data_availability_prompt,
    get_extract_data_sources_prompt
)

logger = logging.getLogger(__name__)

class DataAvailabilityExtractorGPT(DataAvailabilityExtractor):
    def __init__(self, config: Dict):
        self.config = config
        self.openai_config = config.get("openai", {})
        if not self.openai_config:
            raise ValueError("OpenAI configuration is missing")
        
        self.client = openai.OpenAI(api_key=self.openai_config["api_key"])
        self.model = self.openai_config.get("model", "gpt-4-1106-preview")
        
        # Define constant for no data response
        self.NO_DATA_RESPONSE = "<p>This study includes no data deposited in external repositories.</p>"

    def extract_data_availability(self, doc_content: str) -> Dict:
        """Extract data availability information from document."""
        try:
            # First locate the data availability section
            section_text = self._locate_data_availability_section(doc_content)
            
            # If it's the no data response, return early without further processing
            if section_text == self.NO_DATA_RESPONSE:
                return {
                    "section_text": section_text,
                    "data_sources": []
                }

            if not section_text:
                logger.warning("No data availability section found")
                return {
                    "section_text": self.NO_DATA_RESPONSE,
                    "data_sources": []
                }

            # Then extract structured data sources from the section
            data_sources = self._extract_data_records(section_text)
            
            return {
                "section_text": section_text,
                "data_sources": data_sources
            }
            
        except Exception as e:
            logger.error(f"Error extracting data availability: {str(e)}")
            return {
                "section_text": self.NO_DATA_RESPONSE,
                "data_sources": []
            }

    def _locate_data_availability_section(self, doc_content: str) -> str:
        """Locate and extract the data availability section text."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_LOCATE},
                    {"role": "user", "content": get_locate_data_availability_prompt(doc_content)}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p
            )
                        
            if response.choices:
                section_text = response.choices[0].message.content.strip()
                return section_text
                
            return self.NO_DATA_RESPONSE

        except Exception as e:
            logger.error(f"Error locating data availability section: {str(e)}")
            return self.NO_DATA_RESPONSE

    def _extract_data_records(self, section_text: str) -> List[Dict]:
        """Extract structured data sources from the data availability section."""
        # Return early if it's the no data response
        if section_text == self.NO_DATA_RESPONSE:
            return []

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_EXTRACT},
                    {"role": "user", "content": get_extract_data_sources_prompt(section_text)},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p
            )
            if not response.choices:
                return []

            # Extract the text content from the first choice
            content = response.choices[0].message.content

            try:
                # Search for code blocks of the form ```json ... ```
                code_blocks = re.findall(r"```json\s*(.*?)\s*```", content, re.DOTALL)

                if code_blocks:
                    # If we found a code block, parse the first occurrence
                    data_str = code_blocks[0]
                else:
                    # Otherwise, use the entire response content as JSON
                    data_str = content

                # Attempt to parse the extracted string as JSON
                data_sources = json.loads(data_str)

                # If the result is not a list, return empty
                if isinstance(data_sources, list):
                    return data_sources

                return []

            except json.JSONDecodeError:
                logger.error("Failed to parse data sources JSON response")
                return []

        except Exception as e:
            logger.error(f"Error extracting data sources: {str(e)}")
            return []
