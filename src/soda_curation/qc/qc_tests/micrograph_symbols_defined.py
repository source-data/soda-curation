"""Micrograph symbols defined analysis for figures."""

import logging
from typing import Dict, Tuple

from pydantic import TypeAdapter

from ..data_types import BaseModel
from ..model_api import ModelAPI

logger = logging.getLogger(__name__)


class PanelMicrographSymbolsDefined(BaseModel):
    panel_label: str
    micrograph: str  # "yes" or "no"
    symbols: list
    symbols_defined_in_caption: list
    from_the_caption: list


class MicrographSymbolsDefinedResult(BaseModel):
    outputs: list


class MicrographSymbolsDefinedAnalyzer:
    """Analyzer for micrograph symbols defined in figures."""

    def __init__(self, config: Dict):
        self.config = config
        self.model_api = ModelAPI(config)

    def analyze_figure(
        self, figure_label: str, encoded_image: str, figure_caption: str
    ) -> Tuple[bool, MicrographSymbolsDefinedResult]:
        logger.info("Analyzing micrograph symbols defined for figure %s", figure_label)
        test_config = self.config["pipeline"]["micrograph_symbols_defined"]["openai"]
        response = self.model_api.generate_response(
            encoded_image=encoded_image,
            caption=figure_caption,
            prompt_config=test_config,
            response_type=MicrographSymbolsDefinedResult,
        )
        result: MicrographSymbolsDefinedResult = TypeAdapter(
            MicrographSymbolsDefinedResult
        ).validate_json(response)
        # A panel passes if all symbols present are defined in the caption
        passed = True
        for panel in result.outputs:
            if panel.micrograph == "yes":
                if panel.symbols:
                    if not all(s == "yes" for s in panel.symbols_defined_in_caption):
                        passed = False
                        break
        return passed, result
