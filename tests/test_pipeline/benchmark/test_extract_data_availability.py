"""Data availability extraction benchmark tests."""

import logging
import time
from pathlib import Path
from typing import Any, Dict

import yaml

from src.soda_curation.pipeline.data_availability.data_availability_openai import (
    DataAvailabilityExtractorOpenAI,
)
from src.soda_curation.pipeline.manuscript_structure.manuscript_xml_parser import (
    XMLStructureExtractor,
)
from src.soda_curation.pipeline.prompt_handler import PromptHandler

from .base_runner import BaseBenchmarkRunner

logger = logging.getLogger(__name__)


class DataAvailabilityBenchmarkRunner(BaseBenchmarkRunner):
    """Runner for data availability extraction benchmark tests."""

    def run_test(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Run data availability extraction test."""
        # Check cache first
        cache_path = self.get_cache_path(test_case)
        cached_result = self.get_cached_result(cache_path)
        if cached_result is not None:
            logger.info(f"Using cached result for {test_case['msid']}")
            return cached_result

        # Not cached, run the test
        ground_truth = self._load_ground_truth(test_case["msid"])
        result = self._run_data_availability_test(test_case, ground_truth)

        # Cache the result
        self.cache_result(cache_path, result)
        return result

    def _run_data_availability_test(
        self, test_case: Dict[str, Any], ground_truth: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run data availability extraction test."""
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
                    "extract_data_sources": {
                        provider: {
                            "model": test_case["model"],
                            "temperature": test_case["temperature"],
                            "top_p": test_case["top_p"],
                            "prompts": {
                                "system": pipeline_config["extract_data_sources"][
                                    provider
                                ]["prompts"]["system"],
                                "user": pipeline_config["extract_data_sources"][
                                    provider
                                ]["prompts"]["user"],
                            },
                        }
                    }
                },
            }

            # Create prompt handler with the pipeline configuration
            prompt_handler = PromptHandler(extractor_config["pipeline"])

            # Create data availability extractor instance
            if provider == "anthropic":
                raise NotImplementedError("Anthropic is not supported yet")
            elif provider == "openai":
                data_extractor = DataAvailabilityExtractorOpenAI(
                    extractor_config, prompt_handler
                )
            else:
                raise ValueError(f"Unsupported provider: {provider}")

            # Get the ground truth text and sources
            ground_truth_text = ground_truth["data_availability"]["section_text"]
            ground_truth_sources = ground_truth["data_availability"]["data_sources"]

            # Extract data availability using the correct method
            updated_structure = data_extractor.extract_data_sources(
                section_text=ground_truth_text, zip_structure=zip_structure
            )

            # Get the extracted data
            data_availability_text = updated_structure.data_availability["section_text"]
            data_sources = updated_structure.data_availability["data_sources"]

            duration_ms = (time.time() - start_time) * 1000

            # Extract database values correctly from both sources
            actual_databases = [
                source.get("database", "")
                for source in data_sources
                if "database" in source
            ]
            expected_databases = [
                source.get("database", "")
                for source in ground_truth_sources
                if "database" in source
            ]

            # Extract URL values correctly from both sources
            actual_urls = [
                source.get("url", "") for source in data_sources if "url" in source
            ]
            expected_urls = [
                source.get("url", "")
                for source in ground_truth_sources
                if "url" in source
            ]

            # Extract accession number values correctly from both sources
            actual_accessions = [
                source.get("accession_number", "")
                for source in data_sources
                if "accession_number" in source
            ]
            expected_accessions = [
                source.get("accession_number", "")
                for source in ground_truth_sources
                if "accession_number" in source
            ]

            # Custom scoring logic that respects empty lists
            database_score = (
                1.0
                if (not expected_databases and not actual_databases)
                else float(len(set(expected_databases) & set(actual_databases)))
                / max(1, len(set(expected_databases)))
            )

            url_score = (
                1.0
                if (not expected_urls and not actual_urls)
                else float(len(set(expected_urls) & set(actual_urls)))
                / max(1, len(set(expected_urls)))
            )

            accession_score = (
                1.0
                if (not expected_accessions and not actual_accessions)
                else float(len(set(expected_accessions) & set(actual_accessions)))
                / max(1, len(set(expected_accessions)))
            )

            scores = {
                "database_score": database_score,
                "url_score": url_score,
                "accession_score": accession_score,
            }

            # Base result info
            base_result = {"input": docx_content, "ai_response": str(updated_structure)}

            # Collect rows for results DataFrame
            results = []

            # Database extraction task
            results.append(
                self.fill_results_bag(
                    test_case={**test_case, "duration_ms": duration_ms},
                    result=base_result,
                    actual_output=str(actual_databases),
                    expected_output=str(expected_databases),
                    task="database_extraction",
                    score=scores["database_score"],
                )
            )

            # URL extraction task
            results.append(
                self.fill_results_bag(
                    test_case={**test_case, "duration_ms": duration_ms},
                    result=base_result,
                    actual_output=str(actual_urls),
                    expected_output=str(expected_urls),
                    task="url_extraction",
                    score=scores["url_score"],
                )
            )

            # Accession number extraction task
            results.append(
                self.fill_results_bag(
                    test_case={**test_case, "duration_ms": duration_ms},
                    result=base_result,
                    actual_output=str(actual_accessions),
                    expected_output=str(expected_accessions),
                    task="access_number_extraction",
                    score=scores["accession_score"],
                )
            )

            return {
                "input": docx_content,
                "output": data_availability_text,
                "expected": ground_truth_text,
                "extracted_sources": data_sources,
                "ground_truth_sources": ground_truth_sources,
                "scores": scores,
                "results": results,  # Include the results rows
                "cost": updated_structure.cost.extract_data_sources.model_dump()
                if hasattr(updated_structure.cost.extract_data_sources, "model_dump")
                else {},
            }
        except Exception as e:
            logger.error(
                f"Error running data availability test: {str(e)}", exc_info=True
            )
            raise

        finally:
            # Cleanup temporary directory after processing
            if temp_extract_dir:
                self.cleanup_temp_dir(temp_extract_dir)
