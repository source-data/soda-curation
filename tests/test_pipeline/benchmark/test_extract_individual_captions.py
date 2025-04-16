"""Figure captions extraction benchmark tests."""

import logging
import time
from typing import Any, Dict

import yaml

from src.soda_curation.pipeline.extract_captions.extract_captions_openai import (
    FigureCaptionExtractorOpenAI,
)
from src.soda_curation.pipeline.manuscript_structure.manuscript_structure import (
    Figure,
    ZipStructure,
)
from src.soda_curation.pipeline.prompt_handler import PromptHandler

from .base_runner import BaseBenchmarkRunner

logger = logging.getLogger(__name__)


class CaptionsExtractionBenchmarkRunner(BaseBenchmarkRunner):
    """Runner for figure captions extraction benchmark tests."""

    def run_test(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Run figure captions extraction test."""
        # Check cache first
        cache_path = self.get_cache_path(test_case)
        cached_result = self.get_cached_result(cache_path)
        if cached_result is not None:
            logger.info(f"Using cached result for {test_case['msid']}")
            return cached_result

        # Not cached, run the test
        ground_truth = self._load_ground_truth(test_case["msid"])
        result = self._run_extract_captions_test(test_case, ground_truth)

        # Cache the result
        self.cache_result(cache_path, result)
        return result

    def _run_extract_captions_test(
        self, test_case: Dict[str, Any], ground_truth: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run figure captions extraction test."""
        try:
            start_time = time.time()

            # Load prompts from development configuration
            with open(self.config["prompts_source"]) as f:
                dev_config = yaml.safe_load(f)

            # Get the default configuration which contains our pipeline setup
            pipeline_config = dev_config.get("default", {}).get("pipeline", {})
            if not pipeline_config:
                raise ValueError("No pipeline configuration found in default profile")

            # Provider-specific configuration
            provider = test_case["provider"]

            # Create extractor configuration - Updated for new structure with split extraction steps
            extractor_config = {
                "pipeline": {
                    # Extract caption title configuration
                    "extract_caption_title": {
                        provider: {
                            "model": test_case["model"],
                            "temperature": test_case["temperature"],
                            "top_p": test_case["top_p"],
                            "max_tokens": pipeline_config["extract_caption_title"][
                                provider
                            ].get("max_tokens", 2048),
                            "frequency_penalty": pipeline_config[
                                "extract_caption_title"
                            ][provider].get("frequency_penalty", 0.0),
                            "presence_penalty": pipeline_config[
                                "extract_caption_title"
                            ][provider].get("presence_penalty", 0.0),
                            "json_mode": pipeline_config["extract_caption_title"][
                                provider
                            ].get("json_mode", True),
                            "prompts": {
                                "system": pipeline_config["extract_caption_title"][
                                    provider
                                ]["prompts"]["system"],
                                "user": pipeline_config["extract_caption_title"][
                                    provider
                                ]["prompts"]["user"],
                            },
                        }
                    },
                    # Extract panel sequence configuration
                    "extract_panel_sequence": {
                        provider: {
                            "model": test_case["model"],
                            "temperature": test_case["temperature"],
                            "top_p": test_case["top_p"],
                            "max_tokens": pipeline_config["extract_panel_sequence"][
                                provider
                            ].get("max_tokens", 2048),
                            "frequency_penalty": pipeline_config[
                                "extract_panel_sequence"
                            ][provider].get("frequency_penalty", 0.0),
                            "presence_penalty": pipeline_config[
                                "extract_panel_sequence"
                            ][provider].get("presence_penalty", 0.0),
                            "json_mode": pipeline_config["extract_panel_sequence"][
                                provider
                            ].get("json_mode", True),
                            "prompts": {
                                "system": pipeline_config["extract_panel_sequence"][
                                    provider
                                ]["prompts"]["system"],
                                "user": pipeline_config["extract_panel_sequence"][
                                    provider
                                ]["prompts"]["user"],
                            },
                        }
                    },
                },
            }

            # Create prompt handler
            prompt_handler = PromptHandler(extractor_config["pipeline"])

            # Create captions extractor instance
            if provider == "anthropic":
                raise NotImplementedError("Anthropic is not supported yet")
            elif provider == "openai":
                captions_extractor = FigureCaptionExtractorOpenAI(
                    extractor_config, prompt_handler
                )
            else:
                raise ValueError(f"Unsupported provider: {provider}")

            # Ground truth comes from the input
            modified_figures = [
                Figure(
                    figure_label=fig["figure_label"],
                    img_files=fig.get("img_files", []),
                    sd_files=fig.get("sd_files", []),
                    panels=[p for p in fig.get("panels", [])],
                    unassigned_sd_files=fig.get("unassigned_sd_files", []),
                    _full_img_files=fig.get("_full_img_files", []),
                    _full_sd_files=fig.get("_full_sd_files", []),
                    duplicated_panels=fig.get("duplicated_panels", []),
                    ai_response_panel_source_assign=fig.get(
                        "ai_response_panel_source_assign", ""
                    ),
                    figure_caption="",  # Set to empty string
                    caption_title="",  # Set to empty string
                )
                for fig in ground_truth["figures"]
            ]

            # Use the modified figures in ZipStructure
            zip_structure = ZipStructure(figures=modified_figures)

            # This is where we get the AI extraction
            extracted_zip_structure = captions_extractor.extract_individual_captions(
                doc_content=ground_truth["all_captions"], zip_structure=zip_structure
            )

            # Calculate duration before processing results
            duration_ms = (time.time() - start_time) * 1000

            # For each figure in ground truth, find corresponding AI extraction
            results = []
            for gt_figure in ground_truth["figures"]:
                figure_label = gt_figure["figure_label"]

                # Ground truth values
                expected_caption = gt_figure["figure_caption"]
                expected_title = gt_figure["caption_title"]
                expected_panels = [p["panel_label"] for p in gt_figure["panels"]]

                # Find matching AI extracted figure
                extracted_figure = next(
                    (
                        f
                        for f in extracted_zip_structure.figures
                        if f.figure_label == figure_label
                    ),
                    None,
                )

                if extracted_figure:
                    # AI extracted values
                    actual_caption = extracted_figure.figure_caption
                    actual_title = extracted_figure.caption_title
                    actual_caption = extracted_figure.figure_caption
                    actual_title = extracted_figure.caption_title

                    # Fix: Handle both dictionary and object types for panels
                    actual_panels = []
                    for p in extracted_figure.panels:
                        if isinstance(p, dict):
                            actual_panels.append(p.get("panel_label", ""))
                        else:
                            # Safely access the panel_label attribute
                            try:
                                actual_panels.append(p.panel_label)
                            except AttributeError:
                                # If it's neither a dict nor has panel_label attribute
                                logger.warning(
                                    f"Panel object {p} has no panel_label attribute"
                                )
                                actual_panels.append("")

                    # Calculate score for panel sequence
                    panel_sequence_score = 1.0
                    if expected_panels:
                        missing_panels = len(set(expected_panels)) - len(
                            set(expected_panels).intersection(set(actual_panels))
                        )
                        panel_sequence_score = 1.0 - (
                            missing_panels / len(set(expected_panels))
                        )

                    # Add to results for the DataFrame
                    results.append(
                        self.fill_results_bag(
                            test_case={**test_case, "duration_ms": duration_ms},
                            result={"input": ground_truth["all_captions"]},
                            actual_output=actual_title,
                            expected_output=expected_title,
                            figure_label=figure_label,
                            task="figure_title",
                        )
                    )

                    results.append(
                        self.fill_results_bag(
                            test_case={**test_case, "duration_ms": duration_ms},
                            result={"input": ground_truth["all_captions"]},
                            actual_output=actual_caption,
                            expected_output=expected_caption,
                            figure_label=figure_label,
                            task="figure_caption",
                        )
                    )

                    results.append(
                        self.fill_results_bag(
                            test_case={**test_case, "duration_ms": duration_ms},
                            result={"input": ground_truth["all_captions"]},
                            actual_output=str(actual_panels),
                            expected_output=str(expected_panels),
                            figure_label=figure_label,
                            task="panel_sequence",
                            score=panel_sequence_score,
                        )
                    )

            return {
                "input": ground_truth["all_captions"],
                "results": results,
                "extracted_structure": {
                    "figures": [
                        {
                            "figure_label": fig.figure_label,
                            "figure_caption": fig.figure_caption,
                            "caption_title": fig.caption_title,
                            "panels": [
                                {
                                    "panel_label": panel.panel_label
                                    if not isinstance(panel, dict)
                                    else panel.get("panel_label"),
                                    "panel_caption": panel.panel_caption
                                    if not isinstance(panel, dict)
                                    else panel.get("panel_caption"),
                                }
                                for panel in fig.panels
                            ],
                        }
                        for fig in extracted_zip_structure.figures
                    ]
                },
                "cost": extracted_zip_structure.cost.extract_individual_captions.model_dump()
                if hasattr(
                    extracted_zip_structure.cost.extract_individual_captions,
                    "model_dump",
                )
                else {},
            }

        except Exception as e:
            logger.error(
                f"Error running extract captions test: {str(e)}", exc_info=True
            )
            raise
