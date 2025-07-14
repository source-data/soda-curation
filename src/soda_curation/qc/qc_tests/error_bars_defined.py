"""Error bars defined analysis for figures."""

import logging
from typing import Dict, Tuple

from pydantic import TypeAdapter

from ..data_types import BaseModel
from ..model_api import ModelAPI

logger = logging.getLogger(__name__)


class PanelErrorBarsDefined(BaseModel):
    panel_label: str
    error_bars_present: str  # "yes" or "no"
    error_bars_defined_in_caption: str  # "yes" or "no"
    from_the_caption: str


class ErrorBarsDefinedResult(BaseModel):
    outputs: list


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
        # A panel passes if error bars are present and defined when needed
        passed = True
        for panel in result.outputs:
            if panel.error_bars_present == "yes":
                if panel.error_bars_defined_in_caption != "yes":
                    passed = False
                    break
        return passed, result
