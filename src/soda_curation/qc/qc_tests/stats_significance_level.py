"""Statistical significance level analysis for figures."""

import logging
from typing import Dict, Tuple

from pydantic import TypeAdapter

from ..data_types import StatsSignificanceLevelResult
from ..model_api import ModelAPI

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

    def analyze_figure(
        self, figure_label: str, encoded_image: str, figure_caption: str
    ) -> Tuple[bool, StatsSignificanceLevelResult]:
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

        test_config = self.config["pipeline"]["stats_significance_level"]["openai"]

        response = self.model_api.generate_response(
            encoded_image=encoded_image,
            caption=figure_caption,
            prompt_config=test_config,
            response_type=StatsSignificanceLevelResult,
        )

        result: StatsSignificanceLevelResult = TypeAdapter(
            StatsSignificanceLevelResult
        ).validate_json(response)

        # A panel passes if all significance symbols on image are defined in caption
        passed = True
        for panel in result.outputs:
            if panel.is_a_plot == "yes":
                if panel.significance_level_symbols_on_image:
                    if not all(
                        s == "yes"
                        for s in panel.significance_level_symbols_defined_in_caption
                    ):
                        passed = False
                        break
        return passed, result
