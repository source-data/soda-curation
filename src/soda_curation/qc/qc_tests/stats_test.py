"""Statistical test analysis for figures."""

import logging
from typing import Dict, Tuple

from ..data_types import PanelStatsTest, StatsTestResult
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

        try:
            # Get stats test specific config
            test_config = self.config["pipeline"]["stats_test"]["openai"]

            # Call model API to analyze the figure
            response = self.model_api.generate_response(
                encoded_image=encoded_image,
                caption=figure_caption,
                prompt_config=test_config,
            )

            # Parse the response - handle None or missing outputs
            panel_results = []
            if response and isinstance(response, dict) and "outputs" in response:
                for panel_data in response["outputs"]:
                    panel_results.append(
                        PanelStatsTest(
                            panel_label=panel_data.get("panel_label", ""),
                            is_a_plot=panel_data.get("is_a_plot", "no"),
                            statistical_test_needed=panel_data.get(
                                "statistical_test_needed", "no"
                            ),
                            statistical_test_mentioned=panel_data.get(
                                "statistical_test_mentioned", "not needed"
                            ),
                            justify_why_test_is_missing=panel_data.get(
                                "justify_why_test_is_missing", ""
                            ),
                            from_the_caption=panel_data.get("from_the_caption", ""),
                        )
                    )
            else:
                logger.warning(
                    f"Invalid or missing outputs in response for figure {figure_label}"
                )
                if response:
                    logger.debug(f"Response received: {str(response)[:200]}...")
                else:
                    logger.debug("Response was None")

            # Create result object
            result = StatsTestResult(outputs=panel_results)

            # Determine if the figure passes the test
            # A figure passes if all panels that need statistical tests have them mentioned
            needs_stats = [
                p for p in panel_results if p.statistical_test_needed == "yes"
            ]
            missing_stats = [
                p for p in needs_stats if p.statistical_test_mentioned == "no"
            ]
            passed = len(missing_stats) == 0

            return passed, result

        except Exception as e:
            logger.error(
                f"Error analyzing stats test for figure {figure_label}: {str(e)}"
            )
            return False, StatsTestResult(outputs=[])
