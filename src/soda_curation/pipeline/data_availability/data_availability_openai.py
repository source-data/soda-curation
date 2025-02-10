"""OpenAI implementation for data availability extraction."""

import logging
import os
from typing import Dict, List

import openai
from pydantic import BaseModel

from ..cost_tracking import update_token_usage
from ..manuscript_structure.manuscript_structure import ZipStructure
from .data_availability_base import DataAvailabilityExtractor

logger = logging.getLogger(__name__)


class LocateDataAvailability(BaseModel):
    data_availability: str


class IndividualDataSource(BaseModel):
    database: str
    accession_number: str
    url: str


class ExtractDataSources(BaseModel):
    figures: List[IndividualDataSource]


class DataAvailabilityExtractorOpenAI(DataAvailabilityExtractor):
    """Implementation of data availability extraction using OpenAI's GPT models."""

    def __init__(self, config: Dict, prompt_handler):
        """Initialize with OpenAI configuration."""
        super().__init__(config, prompt_handler)

        # Initialize OpenAI client
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        self.client = openai.OpenAI(api_key=api_key)

    def _validate_config(self) -> None:
        """Validate OpenAI configuration parameters."""
        # Validate model
        valid_models = ["gpt-4o", "gpt-4o-mini"]
        for step_ in ["locate_data_availability", "extract_data_sources"]:
            config_ = self.config["pipeline"][step_]["openai"]
            model = config_.get("model", "gpt-4o")
            if model not in valid_models:
                raise ValueError(
                    f"Invalid model: {model}. Must be one of {valid_models}"
                )

            # Validate numerical parameters
            if not 0 <= config_.get("temperature", 0.1) <= 2:
                raise ValueError(
                    f"Temperature must be between 0 and 2 for step: `{step_}`, value: `{config_.get('temperature', 1.0)}`"
                )
            if not 0 <= config_.get("top_p", 1.0) <= 1:
                raise ValueError(
                    f"Top_p must be between 0 and 1 for step: `{step_}`, value: `{config_.get('top_p', 1.0)}`"
                )
            if (
                "frequency_penalty" in config_
                and not -2 <= config_["frequency_penalty"] <= 2
            ):
                raise ValueError(
                    f"Frequency penalty must be between -2 and 2 for step: `{step_}`, value: `{config_.get('frequency_penalty', 0.)}`"
                )
            if (
                "presence_penalty" in config_
                and not -2 <= config_["presence_penalty"] <= 2
            ):
                raise ValueError(
                    f"Presence penalty must be between -2 and 2 for step: `{step_}`, value: `{config_.get('presence_penalty', 0.)}`"
                )

    def extract_data_availability(
        self, doc_content: str, zip_structure: ZipStructure
    ) -> ZipStructure:
        """Extract data availability information and update ZipStructure."""
        try:
            # First locate the data availability section
            section_text = self._locate_data_availability_section(
                doc_content, zip_structure
            )
            if not section_text:
                logger.warning("No data availability section found")
                zip_structure.data_availability = {
                    "section_text": "",
                    "data_sources": [],
                }
                return zip_structure

            # Then extract structured data sources
            data_sources = self._extract_data_sources(section_text, zip_structure)

            # Update ZipStructure
            zip_structure.data_availability = {
                "section_text": section_text,
                "data_sources": data_sources,
            }

            return zip_structure

        except Exception as e:
            logger.error(f"Error extracting data availability: {str(e)}")
            zip_structure.data_availability = {"section_text": "", "data_sources": []}
            return zip_structure

    def _locate_data_availability_section(
        self, doc_content: str, zip_structure: ZipStructure
    ) -> str:
        """Locate and extract the data availability section."""
        try:
            # Get prompts for this step
            prompts = self.prompt_handler.get_prompt(
                step="locate_data_availability",
                variables={"manuscript_text": doc_content},
            )

            # Call OpenAI API
            config_ = self.config["pipeline"]["locate_data_availability"]["openai"]
            response = self.client.chat.completions.create(
                model=config_.get("model", "gpt-4o"),
                messages=[
                    {"role": "system", "content": prompts["system"]},
                    {"role": "user", "content": prompts["user"]},
                ],
                response_format=LocateDataAvailability,
                temperature=config_.get("temperature", 0.1),
                top_p=config_.get("top_p", 1.0),
                frequency_penalty=config_.get("frequency_penalty", 0),
                presence_penalty=config_.get("presence_penalty", 0),
            )

            # Updating the token usage
            update_token_usage(
                zip_structure.cost.locate_data_availability,
                response,
                config_.get("model", "gpt-4o"),
            )

            if response.choices:
                return response.choices[0].message.content.strip()

            return ""

        except Exception as e:
            logger.error(f"Error locating data availability section: {str(e)}")
            return ""

    def _extract_data_sources(
        self, section_text: str, zip_structure: ZipStructure
    ) -> List[Dict]:
        """Extract structured data sources from section text."""
        try:
            # Get prompts for this step
            prompts = self.prompt_handler.get_prompt(
                step="extract_data_sources", variables={"section_text": section_text}
            )

            # Call OpenAI API
            config_ = self.config["pipeline"]["extract_data_sources"]["openai"]
            response = self.client.chat.completions.create(
                model=config_.get("model", "gpt-4o"),
                messages=[
                    {"role": "system", "content": prompts["system"]},
                    {"role": "user", "content": prompts["user"]},
                ],
                response_format=ExtractDataSources,
                temperature=config_.get("temperature", 0.1),
                top_p=config_.get("top_p", 1.0),
                frequency_penalty=config_.get("frequency_penalty", 0),
                presence_penalty=config_.get("presence_penalty", 0),
            )
            # Updating the token usage
            update_token_usage(
                zip_structure.cost.extract_data_sources,
                response,
                config_.get("model", "gpt-4o"),
            )

            if response.choices:
                return self._parse_response(response.choices[0].message.content)

            return []

        except Exception as e:
            logger.error(f"Error extracting data sources: {str(e)}")
            return []
