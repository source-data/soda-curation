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
        logger.info(f"Analyzing stats test for figure {figure_label}")

        # try:
        # Get stats test specific config
        test_config = self.config["pipeline"]["stats_test"]["openai"]

        # Call model API to analyze the figure
        response = self.model_api.generate_response(
            encoded_image=encoded_image,
            caption=figure_caption,
            prompt_config=test_config,
            response_type=StatsTestResult,
        )

        result: StatsTestResult = TypeAdapter(StatsTestResult).validate_json(response)

        # Determine if the figure passes the test
        # A figure passes if all panels that need statistical tests have them mentioned
        needs_stats = [p for p in result.outputs if p.statistical_test_needed == "yes"]
        missing_stats = [p for p in needs_stats if p.statistical_test_mentioned == "no"]
        passed = len(missing_stats) == 0

        return passed, result

        # except Exception as e:
        #     logger.error(
        #         f"Error analyzing stats test for figure {figure_label}: {str(e)}"
        #     )
        #     return False, StatsTestResult(outputs=[])
