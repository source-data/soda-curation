"""Sections extraction benchmark tests."""

import logging
import time
from pathlib import Path
from typing import Any, Dict

import yaml

from src.soda_curation.pipeline.extract_sections.extract_sections_openai import (
    SectionExtractorOpenAI,
)
from src.soda_curation.pipeline.manuscript_structure.manuscript_xml_parser import (
    XMLStructureExtractor,
)
from src.soda_curation.pipeline.prompt_handler import PromptHandler

from ..metrics import normalize_text
from .base_runner import BaseBenchmarkRunner

logger = logging.getLogger(__name__)


class SectionsExtractionBenchmarkRunner(BaseBenchmarkRunner):
    """Runner for sections extraction benchmark tests."""

    def run_test(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Run sections extraction test."""
        # Check cache first
        cache_path = self.get_cache_path(test_case)
        cached_result = self.get_cached_result(cache_path)
        if cached_result is not None:
            logger.info(f"Using cached result for {test_case['msid']}")
            # Process cached results for figures and data availability
            results = []

            # Handle figure captions task from cache
            results.append(
                self.fill_results_bag(
                    test_case=test_case,
                    result={"input": cached_result["input"]},
                    actual_output=normalize_text(cached_result["output"]),
                    expected_output=normalize_text(cached_result["expected"]),
                    task="locate_figure_captions",
                )
            )

            # Handle data availability task from cache
            results.append(
                self.fill_results_bag(
                    test_case=test_case,
                    result={"input": cached_result["input"]},
                    actual_output=normalize_text(
                        cached_result["data_availability_output"], True
                    ),
                    expected_output=normalize_text(
                        cached_result["data_availability_expected"], True
                    ),
                    task="extract_data_availability",
                )
            )

            cached_result["results"] = results
            return cached_result

        # Not cached, run the test
        ground_truth = self._load_ground_truth(test_case["msid"])
        result = self._run_extract_sections_test(test_case, ground_truth)

        # Cache the result
        self.cache_result(cache_path, result)
        return result

    def _run_extract_sections_test(
        self, test_case: Dict[str, Any], ground_truth: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run sections extraction test."""
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

            # Get the DOCX content
            docx_content = extractor.extract_docx_content(zip_structure.docx)

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
                    "extract_sections": {
                        provider: {
                            "model": test_case["model"],
                            "temperature": test_case["temperature"],
                            "top_p": test_case["top_p"],
                            "prompts": {
                                "system": pipeline_config["extract_sections"][provider][
                                    "prompts"
                                ]["system"],
                                "user": pipeline_config["extract_sections"][provider][
                                    "prompts"
                                ]["user"],
                            },
                        }
                    }
                },
            }

            # Create prompt handler with the pipeline configuration
            prompt_handler = PromptHandler(extractor_config["pipeline"])

            # Create sections extractor instance
            if provider == "anthropic":
                raise NotImplementedError("Anthropic is not supported yet")
            elif provider == "openai":
                sections_extractor = SectionExtractorOpenAI(
                    extractor_config, prompt_handler
                )
            else:
                raise ValueError(f"Unsupported provider: {provider}")

            # Extract sections
            (
                figure_legends,
                data_availability,
                updated_structure,
            ) = sections_extractor.extract_sections(
                doc_content=docx_content, zip_structure=zip_structure
            )

            duration_ms = (time.time() - start_time) * 1000

            # Get ground truth sections
            ground_truth_figures = ground_truth["all_captions"]
            ground_truth_data_availability = ground_truth.get(
                "data_availability", {}
            ).get("section_text", "")

            # Prepare results for the DataFrame
            results = []

            # Handle figure captions task
            figure_caption_row = self.fill_results_bag(
                test_case={**test_case, "duration_ms": duration_ms},
                result={"input": str(zip_path)},
                actual_output=normalize_text(figure_legends),
                expected_output=normalize_text(ground_truth_figures),
                task="locate_figure_captions",
            )
            results.append(figure_caption_row)

            # Handle data availability task
            data_availability_row = self.fill_results_bag(
                test_case={**test_case, "duration_ms": duration_ms},
                result={"input": str(zip_path)},
                actual_output=normalize_text(data_availability, True),
                expected_output=normalize_text(ground_truth_data_availability, True),
                task="extract_data_availability",
            )
            results.append(data_availability_row)

            return {
                "input": str(zip_path),
                "output": figure_legends,
                "expected": ground_truth_figures,
                "data_availability_output": data_availability,
                "data_availability_expected": ground_truth_data_availability,
                "results": results,
                "cost": updated_structure.cost.extract_sections.model_dump()
                if hasattr(updated_structure.cost.extract_sections, "model_dump")
                else {},
            }

        except Exception as e:
            logger.error(
                f"Error running extract sections test: {str(e)}", exc_info=True
            )
            raise

        finally:
            # Cleanup temporary directory after processing
            if temp_extract_dir:
                self.cleanup_temp_dir(temp_extract_dir)
