"""Error bars defined analysis for figures."""

import logging
from typing import Dict, Tuple

from pydantic import TypeAdapter

from ..data_types import ErrorBarsDefinedResult
from ..model_api import ModelAPI

logger = logging.getLogger(__name__)


class ErrorBarsDefinedAnalyzer:
    """Analyzer for error bars defined in figures."""

    def __init__(self, config: Dict):
        self.config = config
        self.model_api = ModelAPI(config)

    def analyze_figure(
        self, figure_label: str, encoded_image: str, figure_caption: str
    ) -> Tuple[bool, ErrorBarsDefinedResult]:
        logger.info("Analyzing error bars defined for figure %s", figure_label)
        test_config = self.config["pipeline"]["error_bars_defined"]["openai"]
        response = self.model_api.generate_response(
            encoded_image=encoded_image,
            caption=figure_caption,
            prompt_config=test_config,
            response_type=ErrorBarsDefinedResult,
        )
        result: ErrorBarsDefinedResult = TypeAdapter(
            ErrorBarsDefinedResult
        ).validate_json(response)
        # A panel passes if error_bar_on_figure == "no" or error_bar_defined_in_caption == "yes" or "not needed"
        passed = True
        for panel in result.outputs:
            if (
                panel.error_bar_on_figure == "yes"
                and panel.error_bar_defined_in_caption not in ("yes", "not needed")
            ):
                passed = False
        return passed, result
