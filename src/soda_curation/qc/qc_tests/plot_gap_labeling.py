"""Plot gap labeling analysis for figures."""

import logging
from typing import Dict, Tuple

from pydantic import TypeAdapter

from ..data_types import BaseModel
from ..model_api import ModelAPI

logger = logging.getLogger(__name__)


class PanelPlotGapLabeling(BaseModel):
    panel_label: str
    is_a_plot: str  # "yes" or "no"
    gaps_defined: list
    gap_description: list
    justify_why_gaps_are_missing: list


class PlotGapLabelingResult(BaseModel):
    outputs: list


class PlotGapLabelingAnalyzer:
    """Analyzer for plot gap labeling in figures."""

    def __init__(self, config: Dict):
        self.config = config
        self.model_api = ModelAPI(config)

    def analyze_figure(
        self, figure_label: str, encoded_image: str, figure_caption: str
    ) -> Tuple[bool, PlotGapLabelingResult]:
        logger.info("Analyzing plot gap labeling for figure %s", figure_label)
        test_config = self.config["pipeline"]["plot_gap_labeling"]["openai"]
        response = self.model_api.generate_response(
            encoded_image=encoded_image,
            caption=figure_caption,
            prompt_config=test_config,
            response_type=PlotGapLabelingResult,
        )
        result: PlotGapLabelingResult = TypeAdapter(
            PlotGapLabelingResult
        ).validate_json(response)
        # A panel passes if all axes that need gap labeling have answer == "yes" or "not needed"
        passed = True
        for panel in result.outputs:
            if panel.is_a_plot == "yes":
                for axis in panel.gaps_defined:
                    if axis["answer"] not in ("yes", "not needed"):
                        passed = False
                        break
        return passed, result
