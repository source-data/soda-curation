"""OpenAI implementation for data source extraction."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

import openai

from ..cost_tracking import update_token_usage
from ..manuscript_structure.manuscript_structure import ZipStructure
from ..openai_utils import call_openai_with_fallback, validate_model_config
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
            # Path to identifiers.json file (in the same directory as this module)
            identifiers_path = Path(__file__).parent / "identifiers.json"

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
        valid_models = ["gpt-4o", "gpt-4o-mini", "gpt-5"]
        config_ = self.config["pipeline"]["extract_data_sources"]["openai"]
        model = config_.get("model", "gpt-4o")
        if model not in valid_models:
            raise ValueError(f"Invalid model: {model}. Must be one of {valid_models}")

        # Use the utility function for validation
        validate_model_config(model, config_)

    def extract_data_sources(
        self, section_text: str, zip_structure: ZipStructure
    ) -> ZipStructure:
        """Extract data sources from data availability section."""
        try:
            # Provide database registry as JSON string for the prompt
            db_registry_json = self._create_registry_info()

            # Get prompts with variables substituted
            prompts = self.prompt_handler.get_prompt(
                step="extract_data_sources",
                variables={
                    "data_availability": section_text,
                },
            )

            # Add explicit note to the system prompt
            system_prompt = (
                prompts["system"]
                + "\nDatabase Registry Information (as JSON):\n"
                + db_registry_json
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompts["user"]},
            ]

            config_ = self.config["pipeline"]["extract_data_sources"]["openai"]
            model_ = config_.get("model", "gpt-4o")

            response = call_openai_with_fallback(
                client=self.client,
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
            # When using structured responses, the parsed content is in .parsed
            if hasattr(response.choices[0].message, "parsed"):
                parsed_data = response.choices[0].message.parsed
            else:
                # Fallback for non-structured responses
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
        """Return the database registry as a JSON string for the prompt."""
        return json.dumps(self.database_registry, indent=2)
