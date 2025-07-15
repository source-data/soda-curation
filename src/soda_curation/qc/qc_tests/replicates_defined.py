import logging
from typing import Any, Dict, Tuple

from pydantic import TypeAdapter

from ..data_types import ReplicatesDefinedResult
from ..model_api import ModelAPI

logger = logging.getLogger(__name__)


class ReplicatesDefinedAnalyzer:
    def __init__(self, config: Dict):
        self.config = config
        self.model_api = ModelAPI(config)

    def analyze_figure(
        self, figure_label: str, encoded_image: str, figure_caption: str
    ) -> Tuple[bool, ReplicatesDefinedResult]:
        logger.info("Analyzing replicates defined for figure %s", figure_label)
        test_config = self.config["pipeline"]["replicates_defined"]["openai"]
        response = self.model_api.generate_response(
            encoded_image=encoded_image,
            caption=figure_caption,
            prompt_config=test_config,
            response_type=ReplicatesDefinedResult,
        )
        result: ReplicatesDefinedResult = TypeAdapter(
            ReplicatesDefinedResult
        ).validate_json(response)
        # A panel passes if involves_replicates == "no" or (number and type are not "unknown")
        passed = True
        for panel in result.outputs:
            if panel.involves_replicates == "yes":
                if (
                    not panel.number_of_replicates
                    or panel.number_of_replicates[0] == "unknown"
                    or not panel.type_of_replicates
                    or panel.type_of_replicates[0] == "unknown"
                ):
                    passed = False
        return passed, result
