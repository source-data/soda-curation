"""Error bars defined analysis for figures."""

import logging
from typing import Dict, Tuple

from pydantic import TypeAdapter

from ..data_types import ErrorBarsDefinedResult
from ..model_api import ModelAPI
from ..prompt_registry import registry

logger = logging.getLogger(__name__)


class ErrorBarsDefinedAnalyzer:
    """Analyzer for error bars defined in figures."""

    def __init__(self, config: Dict):
        self.config = config
        self.model_api = ModelAPI(config)
        # Get metadata from registry for traceability
        self.metadata = registry.get_prompt_metadata("error_bars_defined")

    def get_system_prompt(self) -> str:
        """Get the system prompt from the registry."""
        return registry.get_prompt("error_bars_defined")

    def get_schema(self) -> Dict:
        """Get the JSON schema from the registry."""
        return registry.get_schema("error_bars_defined")

    def analyze_figure(
        self, figure_label: str, encoded_image: str, figure_caption: str
    ) -> Tuple[bool, ErrorBarsDefinedResult]:
        logger.info("Analyzing error bars defined for figure %s", figure_label)

        # Get API config from main config
        test_config = self.config["pipeline"]["error_bars_defined"]["openai"]

        # Override system prompt with one from registry
        test_config["prompts"]["system"] = self.get_system_prompt()

        response = self.model_api.generate_response(
            encoded_image=encoded_image,
            caption=figure_caption,
            prompt_config=test_config,
            response_type=ErrorBarsDefinedResult,
        )

        result: ErrorBarsDefinedResult = TypeAdapter(
            ErrorBarsDefinedResult
        ).validate_json(response)

        # Add metadata to result for traceability
        result.metadata = {
            "name": self.metadata.name,
            "description": self.metadata.description,
            "permalink": self.metadata.permalink,
            "version": self.metadata.version,
            "prompt_number": self.metadata.prompt_number,
        }

        # A panel passes if error_bar_on_figure == "no" or error_bar_defined_in_caption == "yes" or "not needed"
        passed = True
        for panel in result.outputs:
            if (
                panel.error_bar_on_figure == "yes"
                and panel.error_bar_defined_in_caption not in ("yes", "not needed")
            ):
                passed = False
        return passed, result
