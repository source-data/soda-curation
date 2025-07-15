"""Micrograph scale bar analysis for figures."""

import logging
from typing import Dict, Tuple

from pydantic import TypeAdapter

from ..data_types import MicrographScaleBarResult
from ..model_api import ModelAPI

logger = logging.getLogger(__name__)


class MicrographScaleBarAnalyzer:
    """Analyzer for micrograph scale bar in figures."""

    def __init__(self, config: Dict):
        self.config = config
        self.model_api = ModelAPI(config)

    def analyze_figure(
        self, figure_label: str, encoded_image: str, figure_caption: str
    ) -> Tuple[bool, MicrographScaleBarResult]:
        logger.info("Analyzing micrograph scale bar for figure %s", figure_label)
        test_config = self.config["pipeline"]["micrograph_scale_bar"]["openai"]
        response = self.model_api.generate_response(
            encoded_image=encoded_image,
            caption=figure_caption,
            prompt_config=test_config,
            response_type=MicrographScaleBarResult,
        )
        result: MicrographScaleBarResult = TypeAdapter(
            MicrographScaleBarResult
        ).validate_json(response)
        # A panel passes if micrograph == "no" or (scale_bar_on_image == "yes" and scale_bar_defined_in_caption == "yes")
        passed = True
        for panel in result.outputs:
            if panel.micrograph == "yes":
                if (
                    panel.scale_bar_on_image != "yes"
                    or panel.scale_bar_defined_in_caption != "yes"
                ):
                    passed = False
        return passed, result
