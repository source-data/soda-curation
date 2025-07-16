"""Statistical significance level analysis for figures."""

import logging
from typing import Any, Dict, Tuple

from pydantic import BaseModel

from ..model_api import ModelAPI
from ..prompt_registry import registry

logger = logging.getLogger(__name__)


class StatsSignificanceLevelAnalyzer:
    """Analyzer for statistical significance level in figures."""

    def __init__(self, config: Dict):
        """
        Initialize StatsSignificanceLevelAnalyzer.

        Args:
            config: Configuration for the analyzer
        """
        self.config = config
        self.model_api = ModelAPI(config)
        # Get metadata from registry for traceability
        self.metadata = registry.get_prompt_metadata("stat_significance_level")
        # Get the dynamically generated model
        self.result_model = registry.get_pydantic_model("stat_significance_level")

    def get_system_prompt(self) -> str:
        """
        Get the system prompt from the registry.

        Returns:
            The system prompt string
        """
        return registry.get_prompt("stat_significance_level")

    def get_schema(self) -> Dict:
        """
        Get the JSON schema from the registry.

        Returns:
            The JSON schema as a dictionary
        """
        return registry.get_schema("stat_significance_level")

    def analyze_figure(
        self, figure_label: str, encoded_image: str, figure_caption: str
    ) -> Tuple[bool, BaseModel]:
        """
        Analyze a figure for statistical significance level usage.

        Args:
            figure_label: Label of the figure
            encoded_image: Base64 encoded image
            figure_caption: Caption of the figure

        Returns:
            Tuple of (passed, result) where passed is True if the figure passes the test
        """
        logger.info("Analyzing stats significance level for figure %s", figure_label)

        # Get API config from main config
        test_config = self.config["pipeline"]["stats_significance_level"]["openai"]

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

        # A panel passes if all significance symbols on image are defined in caption
        passed = True
        for panel in result.outputs:
            if getattr(panel, "is_a_plot") == "yes":
                symbols_on_image = getattr(
                    panel, "significance_level_symbols_on_image", []
                )
                if symbols_on_image:
                    symbols_defined = getattr(
                        panel, "significance_level_symbols_defined_in_caption", []
                    )
                    if not all(s == "yes" for s in symbols_defined):
                        passed = False
                        break
        return passed, result
