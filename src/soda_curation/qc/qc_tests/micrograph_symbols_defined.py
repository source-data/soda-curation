"""Micrograph symbols defined analysis for figures."""

import logging
from typing import Any, Dict, Tuple

from pydantic import BaseModel

from ..model_api import ModelAPI
from ..prompt_registry import registry

logger = logging.getLogger(__name__)


class MicrographSymbolsDefinedAnalyzer:
    """Analyzer for micrograph symbols defined in figures."""

    def __init__(self, config: Dict):
        self.config = config
        self.model_api = ModelAPI(config)
        # Get metadata from registry for traceability
        self.metadata = registry.get_prompt_metadata("micrograph_symbols_defined")
        # Get the dynamically generated model
        self.result_model = registry.get_pydantic_model("micrograph_symbols_defined")

    def get_system_prompt(self) -> str:
        """Get the system prompt from the registry."""
        return registry.get_prompt("micrograph_symbols_defined")

    def get_schema(self) -> Dict:
        """Get the JSON schema from the registry."""
        return registry.get_schema("micrograph_symbols_defined")

    def analyze_figure(
        self, figure_label: str, encoded_image: str, figure_caption: str
    ) -> Tuple[bool, BaseModel]:
        logger.info("Analyzing micrograph symbols defined for figure %s", figure_label)

        # Get API config from main config
        test_config = self.config["pipeline"]["micrograph_symbols_defined"]["openai"]

        # Override system prompt with one from registry
        test_config["prompts"]["system"] = self.get_system_prompt()

        response = self.model_api.generate_response(
            encoded_image=encoded_image,
            caption=figure_caption,
            prompt_config=test_config,
            response_type=None,  # No longer needed as we validate manually
        )

        # Use the dynamically generated model for validation
        result = self.result_model.model_validate_json(response)

        # Add metadata to result for traceability
        setattr(
            result,
            "metadata",
            {
                "name": self.metadata.name,
                "description": self.metadata.description,
                "permalink": self.metadata.permalink,
                "version": self.metadata.version,
                "prompt_number": self.metadata.prompt_number,
            },
        )

        # A panel passes if micrograph == "no" or (symbols is not empty and all are defined in caption)
        passed = True
        for panel in result.outputs:
            if getattr(panel, "micrograph") == "yes":
                symbols = getattr(panel, "symbols", [])
                if symbols:
                    symbols_defined = getattr(panel, "symbols_defined_in_caption", [])
                    if not symbols_defined or any(
                        ans != "yes" for ans in symbols_defined
                    ):
                        passed = False
        return passed, result
