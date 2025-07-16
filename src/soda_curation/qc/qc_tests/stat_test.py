"""Statistical test analysis for figures."""

import logging
from typing import Any, Dict, Tuple

from pydantic import BaseModel

from ..model_api import ModelAPI
from ..prompt_registry import registry

logger = logging.getLogger(__name__)


class StatsTestAnalyzer:
    """Analyzer for statistical tests in figures."""

    def __init__(self, config: Dict):
        """
        Initialize StatsTestAnalyzer.

        Args:
            config: Configuration for the analyzer
        """
        self.config = config
        self.model_api = ModelAPI(config)
        # Get metadata from registry for traceability
        self.metadata = registry.get_prompt_metadata("stat_test")
        # Get the dynamically generated model
        self.result_model = registry.get_pydantic_model("stat_test")

    def get_system_prompt(self) -> str:
        """
        Get the system prompt from the registry.

        Returns:
            The system prompt string
        """
        return registry.get_prompt("stat_test")

    def get_schema(self) -> Dict:
        """
        Get the JSON schema from the registry.

        Returns:
            The JSON schema as a dictionary
        """
        return registry.get_schema("stat_test")

    def analyze_figure(
        self, figure_label: str, encoded_image: str, figure_caption: str
    ) -> Tuple[bool, BaseModel]:
        """
        Analyze a figure for statistical test usage.

        Args:
            figure_label: Label of the figure
            encoded_image: Base64 encoded image
            figure_caption: Caption of the figure

        Returns:
            Tuple of (passed, result) where passed is True if the figure passes the test
        """
        logger.info("Analyzing stats test for figure %s", figure_label)
        logger.debug("StatsTestAnalyzer config: %r", self.config)
        logger.debug(
            "StatsTestAnalyzer config['pipeline']: %r", self.config.get("pipeline")
        )
        logger.debug(
            "StatsTestAnalyzer config['pipeline']['stats_test']: %r",
            self.config.get("pipeline", {}).get("stats_test"),
        )
        logger.debug(
            "Input params: figure_label=%r, encoded_image_present=%r, figure_caption=%r",
            figure_label,
            bool(encoded_image),
            figure_caption,
        )

        # Defensive config extraction with error logging
        try:
            test_config = self.config["pipeline"]["stats_test"]["openai"]

            # Override system prompt with one from registry
            test_config["prompts"]["system"] = self.get_system_prompt()
        except KeyError as e:
            logger.error("Missing config key: %s. Full config: %r", e, self.config)
            # Create an empty result using the model from the registry
            empty_result = self.result_model(outputs=[])
            return False, empty_result

        try:
            response = self.model_api.generate_response(
                encoded_image=encoded_image,
                caption=figure_caption,
                prompt_config=test_config,
                response_type=None,  # No longer needed as we validate manually
            )

            if response is None:
                logger.warning("Model API returned None for figure %s", figure_label)
                # Create an empty result using the model from the registry
                empty_result = self.result_model(outputs=[])
                return False, empty_result

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

            # Check for empty outputs
            outputs = getattr(result, "outputs", None)
            if not outputs:
                logger.warning("No outputs in result for figure %s", figure_label)
                # Create an empty result using the model from the registry
                empty_result = self.result_model(outputs=[])
                return False, empty_result

            # Check which panels need stats tests and whether they have them
            needs_stats = [
                p
                for p in outputs
                if getattr(p, "statistical_test_needed", None) == "yes"
            ]
            missing_stats = [
                p
                for p in needs_stats
                if getattr(p, "statistical_test_mentioned", None) == "no"
            ]
            passed = len(missing_stats) == 0
            return passed, result
        except Exception as e:
            logger.error(
                "Error analyzing stats test for figure %s: %s", figure_label, str(e)
            )
            # Create an empty result using the model from the registry
            empty_result = self.result_model(outputs=[])
            return False, empty_result
