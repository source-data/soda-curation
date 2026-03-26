"""Anthropic Claude implementation for data source extraction."""

import json
import logging
from pathlib import Path
from typing import Any, Dict

import anthropic

from ..ai_observability import summarize_text
from ..anthropic_utils import call_anthropic, validate_anthropic_model
from ..cost_tracking import update_token_usage
from ..manuscript_structure.manuscript_structure import ZipStructure
from .data_availability_base import DataAvailabilityExtractor, ExtractDataSources

logger = logging.getLogger(__name__)


class DataAvailabilityExtractorAnthropic(DataAvailabilityExtractor):
    """Extract data availability information using Anthropic Claude models."""

    def __init__(self, config: Dict, prompt_handler):
        super().__init__(config, prompt_handler)
        self.client = anthropic.Anthropic()
        self.database_registry = self._load_database_registry()

    def _load_database_registry(self) -> Dict[Any, Any]:
        """Load database registry from identifiers.json file."""
        try:
            identifiers_path = Path(__file__).parent / "identifiers.json"
            with open(identifiers_path, "r") as f:
                content = f.read()
            registry: Dict[Any, Any] = json.loads(content)
            return registry
        except Exception as e:
            logger.error(f"Error loading database registry: {str(e)}")
            return {"databases": []}

    def _validate_config(self) -> None:
        """Validate Anthropic configuration parameters."""
        config_ = self.config["pipeline"]["extract_data_sources"]["anthropic"]
        validate_anthropic_model(config_.get("model", "claude-sonnet-4-6"))

    def extract_data_sources(
        self, section_text: str, zip_structure: ZipStructure
    ) -> ZipStructure:
        """Extract data sources from data availability section."""
        logger.info(
            "Preparing data availability extraction request",
            extra={
                "operation": "main.extract_data_sources",
                "provider": "anthropic",
                "section_summary": summarize_text(section_text),
            },
        )
        db_registry_json = self._create_registry_info()

        prompts = self.prompt_handler.get_prompt(
            step="extract_data_sources",
            variables={"data_availability": section_text},
        )

        system_prompt = (
            prompts["system"]
            + "\nDatabase Registry Information (as JSON):\n"
            + db_registry_json
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompts["user"]},
        ]

        config_ = self.config["pipeline"]["extract_data_sources"]["anthropic"]
        model_ = config_.get("model", "claude-sonnet-4-6")

        response = call_anthropic(
            client=self.client,
            model=model_,
            messages=messages,
            response_format=ExtractDataSources,
            temperature=config_.get("temperature", 0.1),
            max_tokens=config_.get("max_tokens", 2048),
            operation="main.extract_data_sources",
            request_metadata={
                "registry_database_count": len(
                    self.database_registry.get("databases", [])
                )
            },
        )

        update_token_usage(zip_structure.cost.extract_data_sources, response, model_)

        if response.choices[0].message.parsed is not None:
            parsed_data = response.choices[0].message.parsed
            if hasattr(parsed_data, "model_dump"):
                parsed_data = parsed_data.model_dump()
        else:
            response_data = response.choices[0].message.content
            sources_list = self._parse_response(response_data)
            parsed_data = {"sources": sources_list}

        zip_structure.data_availability = {
            "section_text": section_text,
            "data_sources": parsed_data["sources"],
        }

        logger.info(
            "Data availability extraction completed",
            extra={
                "operation": "main.extract_data_sources",
                "provider": "anthropic",
                "data_source_count": len(parsed_data.get("sources", [])),
            },
        )

        return zip_structure

    def _create_registry_info(self) -> str:
        """Return the database registry as a JSON string for the prompt."""
        return json.dumps(self.database_registry, indent=2)
