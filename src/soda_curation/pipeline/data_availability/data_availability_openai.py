"""OpenAI implementation for data source extraction."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

import openai

from ..cost_tracking import update_token_usage
from ..manuscript_structure.manuscript_structure import ZipStructure
from .data_availability_base import DataAvailabilityExtractor, ExtractDataSources

logger = logging.getLogger(__name__)


class DataAvailabilityExtractorOpenAI(DataAvailabilityExtractor):
    """Implementation of data source extraction using OpenAI's GPT models."""

    def __init__(self, config: Dict, prompt_handler):
        """Initialize with OpenAI configuration."""
        super().__init__(config, prompt_handler)

        # Initialize OpenAI client
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        self.client = openai.OpenAI(api_key=api_key)

        # Load database registry from identifiers.txt
        self.database_registry = self._load_database_registry()

    def _load_database_registry(self) -> Dict[Any, Any]:
        """Load database registry from identifiers.txt file."""
        try:
            # Path to identifiers.txt file (in the same directory as this module)
            identifiers_path = Path(__file__).parent / "identifiers.txt"

            with open(identifiers_path, "r") as f:
                content = f.read()

            # Parse the JSON content directly
            registry: Dict[Any, Any] = json.loads(content)

            return registry

        except Exception as e:
            logger.error(f"Error loading database registry: {str(e)}")
            return {"databases": []}

    def _validate_config(self) -> None:
        """Validate OpenAI configuration parameters."""
        # Validate model
        valid_models = ["gpt-4o", "gpt-4o-mini"]
        config_ = self.config["pipeline"]["extract_data_sources"]["openai"]
        model = config_.get("model", "gpt-4o")
        if model not in valid_models:
            raise ValueError(f"Invalid model: {model}. Must be one of {valid_models}")

        # Validate numerical parameters
        if not 0 <= config_.get("temperature", 0.1) <= 2:
            raise ValueError(
                f"Temperature must be between 0 and 2, value: `{config_.get('temperature', 1.0)}`"
            )
        if not 0 <= config_.get("top_p", 1.0) <= 1:
            raise ValueError(
                f"Top_p must be between 0 and 1, value: `{config_.get('top_p', 1.0)}`"
            )
        if (
            "frequency_penalty" in config_
            and not -2 <= config_["frequency_penalty"] <= 2
        ):
            raise ValueError(
                f"Frequency penalty must be between -2 and 2, value: `{config_.get('frequency_penalty', 0.)}`"
            )
        if "presence_penalty" in config_ and not -2 <= config_["presence_penalty"] <= 2:
            raise ValueError(
                f"Presence penalty must be between -2 and 2, value: `{config_.get('presence_penalty', 0.)}`"
            )

    def extract_data_sources(
        self, section_text: str, zip_structure: ZipStructure
    ) -> ZipStructure:
        """Extract data sources from data availability section."""
        try:
            # Create database registry information for the prompt
            db_registry_info = self._create_registry_info()

            # Get prompts with variables substituted
            prompts = self.prompt_handler.get_prompt(
                step="extract_data_sources",
                variables={
                    "data_availability": section_text,
                },
            )

            # Prepare messages
            system_prompt = prompts["system"] + f"\n{db_registry_info}"

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompts["user"]},
            ]

            config_ = self.config["pipeline"]["extract_data_sources"]["openai"]
            model_ = config_.get("model", "gpt-4o")

            response = self.client.beta.chat.completions.parse(
                model=model_,
                messages=messages,
                response_format=ExtractDataSources,
                temperature=config_.get("temperature", 0.1),
                top_p=config_.get("top_p", 1.0),
                frequency_penalty=config_.get("frequency_penalty", 0),
                presence_penalty=config_.get("presence_penalty", 0),
            )

            # Update token usage
            update_token_usage(
                zip_structure.cost.extract_data_sources, response, model_
            )

            # Update the ZipStructure with data
            response_data = response.choices[0].message.content
            parsed_data = self._parse_response(response_data)

            zip_structure.data_availability = {
                "section_text": section_text,
                "data_sources": parsed_data,
            }

            return zip_structure

        except Exception as e:
            logger.error(f"Error extracting data sources: {str(e)}")
            zip_structure.data_availability = {"section_text": "", "data_sources": []}
            return zip_structure

    def _create_registry_info(self) -> str:
        """Create a formatted markdown table with database registry information for the prompt."""
        if not self.database_registry["databases"]:
            return ""

        # Create markdown table header
        registry_info = "| Database Name | Identifiers Pattern | URL Pattern | Sample ID | Sample Identifiers URL |\n"
        registry_info += "|--------------|-------------------|------------|-----------|----------------------|\n"

        # Add each database as a row in the table
        for db in self.database_registry["databases"]:
            name = db.get("name", "")
            identifiers_pattern = db.get("identifiers_pattern", "")
            url_pattern = db.get("url_pattern", "")
            sample_id = db.get("sample_id", "")
            sample_identifiers_url = db.get("sample_identifiers_url", "")

            # Format as table row, escaping pipe characters if they exist in the data
            registry_info += f"| {name} | {identifiers_pattern} | {url_pattern} | {sample_id} | {sample_identifiers_url} |\n"

        return registry_info
