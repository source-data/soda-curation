"""Individual data points analysis for figures."""

import logging
from typing import Dict, Tuple

from pydantic import TypeAdapter

from ..data_types import BaseModel
from ..model_api import ModelAPI

logger = logging.getLogger(__name__)


class PanelIndividualDataPoints(BaseModel):
    panel_label: str
    individual_values: str  # "yes", "no", or "not needed"


class IndividualDataPointsResult(BaseModel):
    outputs: list


class IndividualDataPointsAnalyzer:
    """Analyzer for individual data points in figures."""

    def __init__(self, config: Dict):
        self.config = config
        self.model_api = ModelAPI(config)

    def analyze_figure(
        self, figure_label: str, encoded_image: str, figure_caption: str
    ) -> Tuple[bool, IndividualDataPointsResult]:
        logger.info("Analyzing individual data points for figure %s", figure_label)
        test_config = self.config["pipeline"]["individual_data_points"]["openai"]
        response = self.model_api.generate_response(
            encoded_image=encoded_image,
            caption=figure_caption,
            prompt_config=test_config,
            response_type=IndividualDataPointsResult,
        )
        result: IndividualDataPointsResult = TypeAdapter(
            IndividualDataPointsResult
        ).validate_json(response)
        # A panel passes if individual values are shown or not needed
        passed = True
        for panel in result.outputs:
            if panel.individual_values not in ("yes", "not needed"):
                passed = False
                break
        return passed, result
