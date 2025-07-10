"""Plot axis units analysis for figures."""

import logging
from typing import Dict, Tuple

from pydantic import TypeAdapter

from ..data_types import PlotAxisUnitsResult
from ..model_api import ModelAPI

logger = logging.getLogger(__name__)


class PlotAxisUnitsAnalyzer:
    """Analyzer for plot axis units in figures."""

    def __init__(self, config: Dict):
        self.config = config
        self.model_api = ModelAPI(config)

    def analyze_figure(
        self, figure_label: str, encoded_image: str, figure_caption: str
    ) -> Tuple[bool, PlotAxisUnitsResult]:
        logger.info("Analyzing plot axis units for figure %s", figure_label)
        test_config = self.config["pipeline"]["plot_axis_units"]["openai"]
        response = self.model_api.generate_response(
            encoded_image=encoded_image,
            caption=figure_caption,
            prompt_config=test_config,
            response_type=PlotAxisUnitsResult,
        )
        result: PlotAxisUnitsResult = TypeAdapter(PlotAxisUnitsResult).validate_json(
            response
        )
        # A panel passes if all axes that need units have answer == "yes" or "not needed"
        passed = True
        for panel in result.outputs:
            if panel.is_a_plot == "yes":
                for axis in panel.units_provided:
                    if axis.answer not in ("yes", "not needed"):
                        passed = False
                        break
        return passed, result
