import base64
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict

import yaml

from src.soda_curation.pipeline.match_caption_panel.match_caption_panel_openai import (
    MatchPanelCaptionOpenAI,
)
from src.soda_curation.pipeline.prompt_handler import PromptHandler

from .base_runner import BaseBenchmarkRunner

logger = logging.getLogger(__name__)


class PanelAssignmentBenchmarkRunner(BaseBenchmarkRunner):
    """Runner for panel assignment benchmark tests."""

    def run_test(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Run panel assignment test."""
        # Check cache first
        cache_path = self.get_cache_path(test_case)
        cached_result = self.get_cached_result(cache_path)
        if cached_result is not None:
            logger.info(f"Using cached result for {test_case['msid']}")
            return cached_result

        # Not cached, run the test
        ground_truth = self._load_ground_truth(test_case["msid"])

        result = self._run_panel_assignment_test(test_case, ground_truth)

        # Cache the result
        self.cache_result(cache_path, result)
        return result

    @staticmethod
    def _calculate_match_score(list1, list2):
        if not list1:  # If list is empty
            return 1.0 if not list2 else 0.0

        matches = sum(1 for x, y in zip(list1, list2) if x == y)
        total_elements = len(list1)
        score = matches / total_elements
        return score

    def make_serializable(self, result_dict):
        if isinstance(result_dict, dict):
            return {k: self.make_serializable(v) for k, v in result_dict.items()}
        elif isinstance(result_dict, list):
            return [self.make_serializable(item) for item in result_dict]
        elif hasattr(result_dict, "__dict__"):
            # Convert custom objects to dictionaries
            return self.make_serializable(result_dict.__dict__)
        else:
            # Try to convert to string representation if not a basic type
            try:
                json.dumps(result_dict)
                return result_dict
            except (TypeError, OverflowError):
                return str(result_dict)

    def _run_panel_assignment_test(
        self, test_case: Dict[str, Any], ground_truth: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run panel assignment test."""
        # try:

        # Get manuscript path
        msid = test_case["msid"]
        temp_extract_dir = self.create_temp_extract_dir(msid)
        # Load prompts from development configuration
        with open(self.config["prompts_source"]) as f:
            dev_config = yaml.safe_load(f)
        # Get the default configuration which contains our pipeline setup
        pipeline_config = dev_config.get("default", {}).get("pipeline", {})
        if not pipeline_config:
            raise ValueError("No pipeline configuration found in default profile")
        # Provider-specific configuration
        provider = test_case["provider"]
        # Create extractor configuration
        extractor_config = {
            "api_key": None,  # We don't need to pass the API key here
            "pipeline": {
                "match_caption_panel": {
                    provider: {
                        "model": test_case["model"],
                        "temperature": test_case["temperature"],
                        "top_p": test_case["top_p"],
                        "prompts": {
                            "system": pipeline_config["match_caption_panel"][provider][
                                "prompts"
                            ]["system"],
                            "user": pipeline_config["match_caption_panel"][provider][
                                "prompts"
                            ]["user"],
                        },
                    }
                }
            },
            "extraction_dir": str(temp_extract_dir),
        }

        # Initialize the panel matcher
        prompt_handler = PromptHandler(extractor_config["pipeline"])
        panel_matcher = MatchPanelCaptionOpenAI(
            extractor_config, prompt_handler, Path(self.config["manuscript_dir"])
        )

        # Iterate over each figure in the ground truth

        results = []

        msid_path = (
            Path(self.config["ground_truth_dir"])
            / "match_panel_captions"
            / test_case["msid"]
        )
        for figure in ground_truth["figures"]:
            start_time = time.time()
            figure_caption = figure["figure_caption"]
            # Use regex to extract the figure number
            match = re.search(r"\d+", figure["figure_label"])
            if match:
                figure_label = match.group()
            else:
                logger.warning(
                    f"Could not extract figure number from label: {figure['figure_label']}"
                )
                continue
            expected_output = []
            actual_output = []
            for panel in figure["panels"]:
                panel_label = panel["panel_label"]

                # Construct the path to the panel image
                panel_image_path = msid_path / f"{figure_label}_{panel_label}.png"

                if not panel_image_path.exists():
                    logger.warning(f"Panel image not found: {panel_image_path}")
                    continue

                # Read the image and encode it
                with open(panel_image_path, "rb") as image_file:
                    encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

                # Get the AI-assigned panel label
                panel_str = panel_matcher._match_panel_caption(
                    encoded_image, figure_caption
                )
                panel_object = json.loads(panel_str)
                ai_panel_label = panel_object.get("panel_label", "")

                expected_output.append(panel_label)
                actual_output.append(ai_panel_label)

            duration_ms = (time.time() - start_time) * 1000

            # Calculate the score for the figure
            score = self._calculate_match_score(actual_output, expected_output)
            # Fill results bag for the figure

            results.append(
                self.fill_results_bag(
                    test_case={**test_case, "duration_ms": duration_ms},
                    result={
                        "input": {
                            "image_file": str(msid_path),
                            "caption": figure_caption,
                        }
                    },
                    actual_output=(",").join(actual_output),
                    expected_output=(",").join(expected_output),
                    figure_label=figure_label,
                    task="panel_assignment",
                    score=score,
                )
            )

        return {
            "input": str(msid_path),
            "results": self.make_serializable(results),
        }

        # except Exception as e:
        #     logger.error(f"Error running panel assignment test: {str(e)}", exc_info=True)
        #     return None
