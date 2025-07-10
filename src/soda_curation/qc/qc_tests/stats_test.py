"""Statistical test analysis for figures."""

import logging
from typing import Dict, Tuple

from pydantic import TypeAdapter

from ..data_types import StatsTestResult
from ..model_api import ModelAPI

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

    def analyze_figure(
        self, figure_label: str, encoded_image: str, figure_caption: str
    ) -> Tuple[bool, StatsTestResult]:
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
        except KeyError as e:
            logger.error("Missing config key: %s. Full config: %r", e, self.config)
            return False, StatsTestResult(outputs=[])

        try:
            response = self.model_api.generate_response(
                encoded_image=encoded_image,
                caption=figure_caption,
                prompt_config=test_config,
                response_type=StatsTestResult,
            )

            if response is None:
                logger.warning("Model API returned None for figure %s", figure_label)
                return False, StatsTestResult(outputs=[])
            result: StatsTestResult = TypeAdapter(StatsTestResult).validate_json(
                response
            )
            if not hasattr(result, "outputs") or result.outputs is None:
                logger.warning("No outputs in result for figure %s", figure_label)
                return False, StatsTestResult(outputs=[])
            needs_stats = [
                p
                for p in result.outputs
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
            return False, StatsTestResult(outputs=[])
