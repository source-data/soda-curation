"""Micrograph scale bar analysis for figures."""

import logging
from typing import Dict, Tuple

from pydantic import TypeAdapter

from ..data_types import BaseModel
from ..model_api import ModelAPI

logger = logging.getLogger(__name__)


class PanelMicrographScaleBar(BaseModel):
    panel_label: str
    micrograph: str  # "yes" or "no"
    scale_bar_on_image: str  # "yes" or "no"
    scale_bar_defined_in_caption: str  # "yes" or "no"
    from_the_caption: str


class MicrographScaleBarResult(BaseModel):
    outputs: list


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
        # A panel passes if a scale bar is present and defined when needed
        passed = True
        for panel in result.outputs:
            if panel.micrograph == "yes":
                if panel.scale_bar_on_image == "yes":
                    if panel.scale_bar_defined_in_caption != "yes":
                        passed = False
                        break
        return passed, result
