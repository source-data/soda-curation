"""Panel source assignment benchmark tests."""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml

from src.soda_curation.pipeline.assign_panel_source.assign_panel_source_openai import (
    PanelSourceAssignerOpenAI,
)
from src.soda_curation.pipeline.manuscript_structure.manuscript_xml_parser import (
    XMLStructureExtractor,
)
from src.soda_curation.pipeline.prompt_handler import PromptHandler

from .base_runner import BaseBenchmarkRunner

logger = logging.getLogger(__name__)


class PanelSourceBenchmarkRunner(BaseBenchmarkRunner):
    """Runner for panel source assignment benchmark tests."""

    def run_test(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Run panel source assignment test."""
        # Check cache first
        cache_path = self.get_cache_path(test_case)
        cached_result = self.get_cached_result(cache_path)
        if cached_result is not None:
            logger.info(f"Using cached result for {test_case['msid']}")
            return cached_result

        # Not cached, run the test
        ground_truth = self._load_ground_truth(test_case["msid"])
        result = self._run_panel_source_test(test_case, ground_truth)

        # Cache the result
        self.cache_result(cache_path, result)
        return result

    def _calculate_panel_score(
        self, expected_files: List[str], actual_files: List[str]
    ) -> float:
        """Calculate score for a single panel."""
        # If no files are expected and none provided, that's correct
        if not expected_files and not actual_files:
            return 1.0

        # If no files are expected but some provided, that's wrong
        if not expected_files and actual_files:
            return 0.0

        # If files are expected but none provided, that's wrong
        if expected_files and not actual_files:
            return 0.0

        # Calculate exact matches
        expected_set = set(expected_files)
        actual_set = set(actual_files)
        correct_matches = len(expected_set.intersection(actual_set))

        # Fix: Make sure denominator is never zero
        return correct_matches / max(1, len(expected_files))

    def _run_panel_source_test(
        self, test_case: Dict[str, Any], ground_truth: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run panel source assignment test."""
        temp_extract_dir = None
        try:
            start_time = time.time()

            # Get manuscript path
            msid = test_case["msid"]
            temp_extract_dir = self.create_temp_extract_dir(msid)

            # Get ZIP path
            zip_path = Path(self.config["manuscript_dir"]) / f"{msid}.zip"
            if not zip_path.exists():
                raise FileNotFoundError(f"ZIP file not found: {zip_path}")

            # Use XML extractor to get structure and extract files
            extractor = XMLStructureExtractor(str(zip_path), str(temp_extract_dir))
            zip_structure = extractor.extract_structure()

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
                    "assign_panel_source": {
                        provider: {
                            "model": test_case["model"],
                            "temperature": test_case["temperature"],
                            "top_p": test_case["top_p"],
                            "prompts": {
                                "system": pipeline_config["assign_panel_source"][
                                    provider
                                ]["prompts"]["system"],
                                "user": pipeline_config["assign_panel_source"][
                                    provider
                                ]["prompts"]["user"],
                            },
                        }
                    }
                },
                "extraction_dir": str(temp_extract_dir),
            }

            # Create prompt handler with the pipeline configuration
            prompt_handler = PromptHandler(extractor_config["pipeline"])

            # Create panel source assigner instance
            if provider == "anthropic":
                raise NotImplementedError("Anthropic is not supported yet")
            elif provider == "openai":
                panel_assigner = PanelSourceAssignerOpenAI(
                    extractor_config,
                    prompt_handler,
                    extract_dir=str(temp_extract_dir),  # Add the extract_dir parameter
                )
            else:
                raise ValueError(f"Unsupported provider: {provider}")

            # Assign panel sources
            logger.info("Assigning panel sources")
            extracted_figures = panel_assigner.assign_panel_source(zip_structure)

            duration_ms = (time.time() - start_time) * 1000

            # Calculate scores and prepare outputs
            scores = {}
            results = []

            for gt_figure in ground_truth["figures"]:
                figure_label = gt_figure["figure_label"]

                # Find matching extracted figure
                ext_figure = next(
                    (f for f in extracted_figures if f.figure_label == figure_label),
                    None,
                )

                if ext_figure:
                    for gt_panel in gt_figure["panels"]:
                        panel_label = gt_panel["panel_label"]

                        # Find matching extracted panel
                        ext_panel = next(
                            (
                                p
                                for p in ext_figure.panels
                                if p.panel_label == panel_label
                            ),
                            None,
                        )

                        panel_key = f"{figure_label}_{panel_label}"
                        expected_files = gt_panel["sd_files"]
                        actual_files = ext_panel.sd_files if ext_panel else []

                        # Calculate score for this panel
                        panel_score = self._calculate_panel_score(
                            expected_files, actual_files
                        )

                        scores[panel_key] = panel_score

                        # Add to results for the DataFrame
                        results.append(
                            self.fill_results_bag(
                                test_case={**test_case, "duration_ms": duration_ms},
                                result={"input": str(zip_structure)},
                                actual_output=str(actual_files),
                                expected_output=str(expected_files),
                                figure_label=figure_label,
                                panel_label=panel_label,
                                task="panel_source_assignment",
                                score=panel_score,
                            )
                        )
                else:
                    logger.warning(
                        f"No matching extracted figure found for {figure_label}"
                    )

            # Calculate average score
            average_score = sum(scores.values()) / len(scores) if scores else 0.0
            scores["average_score"] = average_score

            return {
                "input": str(zip_structure),
                "extracted_figures": [
                    {
                        "figure_label": fig.figure_label,
                        "panels": [
                            {"panel_label": p.panel_label, "sd_files": p.sd_files}
                            for p in fig.panels
                        ],
                    }
                    for fig in extracted_figures
                ],
                "ground_truth_figures": ground_truth["figures"],
                "scores": scores,
                "results": results,
            }

        except Exception as e:
            logger.error(
                f"Error running panel source assignment test: {str(e)}", exc_info=True
            )
            raise

        finally:
            # Cleanup temporary directory after processing
            if temp_extract_dir:
                self.cleanup_temp_dir(temp_extract_dir)
