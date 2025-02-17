"""
Benchmark testing framework for SODA curation pipeline.
"""

import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import nltk
import pandas as pd
import pytest
import yaml
from deepeval.test_case import LLMTestCase

from src.soda_curation.pipeline.extract_captions.extract_captions_openai import (
    FigureCaptionExtractorOpenAI,
)
from src.soda_curation.pipeline.extract_sections.extract_sections_openai import (
    SectionExtractorOpenAI,
)
from src.soda_curation.pipeline.manuscript_structure.manuscript_structure import (
    ZipStructure,
)
from src.soda_curation.pipeline.manuscript_structure.manuscript_xml_parser import (
    XMLStructureExtractor,
)
from src.soda_curation.pipeline.prompt_handler import PromptHandler

from .metrics import get_metrics_for_task

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt")
    nltk.download("punkt_tab")

# Define a results bag to store test results
results_bag = []


def preprocess_text(text: str) -> str:
    """
    Preprocess text for comparison by:
    1. Removing escape characters
    2. Normalizing whitespace
    3. Stripping extra spaces

    Args:
        text (str): Input text to process

    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return text

    # Replace common escape sequences with their actual characters
    text = text.replace("\\n", " ")
    text = text.replace("\\t", " ")
    text = text.replace("\\r", " ")

    # Normalize whitespace
    text = " ".join(text.split())

    return text.strip()


class BenchmarkCache:
    """Handle caching of benchmark results."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_path(
        self,
        test_name: str,
        msid: str,
        provider: str,
        model: str,
        temperature: float,
        top_p: float,
        run: int,
    ) -> Path:
        """Get path for cached result."""
        return (
            self.cache_dir
            / f"{test_name}_{msid}_{provider}_{model}_{temperature}_{top_p}_{run}.json"
        )

    def get_cached_result(self, cache_path: Path) -> Optional[Dict[str, Any]]:
        """Get cached result if it exists."""
        if cache_path.exists():
            with open(cache_path) as f:
                return json.load(f)
        return None

    def cache_result(self, cache_path: Path, result: Dict[str, Any]) -> None:
        """Cache test result."""
        with open(cache_path, "w") as f:
            json.dump(result, f)


class BenchmarkRunner:
    """Runner for benchmark tests."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize benchmark runner with configuration."""
        self.config = config

        # Setup base directories
        self.base_output_dir = Path(
            self.config.get("output_dir", "/app/data/benchmark")
        )

        # Setup cache directory
        self.cache_dir = self.base_output_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Setup timestamp-based results directory
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.results_dir = self.base_output_dir / timestamp
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize cache
        self.cache = BenchmarkCache(self.cache_dir)

        # Initialize results DataFrame
        self.results_df = pd.DataFrame(
            columns=[
                "pytest_obj",
                "status",
                "duration_ms",
                "strategy",
                "msid",
                "run",
                "task",
                "input",
                "actual",
                "expected",
                "figure_label",
                "ai_response",
                "rouge1",
                "rouge1_success",
                "rouge1_threshold",
                "rouge2",
                "rouge2_success",
                "rouge2_threshold",
                "rougeL",
                "rougeL_success",
                "rougeL_threshold",
                "bleu1",
                "bleu1_success",
                "bleu1_threshold",
                "bleu2",
                "bleu2_success",
                "bleu2_threshold",
                "bleu3",
                "bleu3_success",
                "bleu3_threshold",
                "bleu4",
                "bleu4_success",
                "bleu4_threshold",
                "data_source_accuracy",
                "data_source_accuracy_success",
                "data_source_accuracy_threshold",
                "data_source_accuracy_exact",
                "data_source_accuracy_jaccard",
                "timestamp",
            ]
        )

    def get_test_cases(self) -> Generator[Dict[str, Any], None, None]:
        """Generate test cases from configuration."""
        for provider, prov_config in self.config["providers"].items():
            for model in prov_config["models"]:
                for temp in model["temperatures"]:
                    for top_p in model["top_p"]:
                        for test_name in self.config[
                            "enabled_tests"
                        ]:  # Changed from "tests" to "enabled_tests"
                            # Handle different manuscript configurations
                            manuscripts = self.config["test_runs"]["manuscripts"]
                            if (
                                isinstance(manuscripts, str)
                                and manuscripts.lower() == "all"
                            ):
                                manuscripts = self._get_all_manuscripts()
                            elif isinstance(manuscripts, int):
                                manuscripts = self._get_n_manuscripts(manuscripts)
                            elif isinstance(manuscripts, list):
                                manuscripts = manuscripts
                            else:
                                raise ValueError(
                                    f"Invalid manuscripts configuration: {manuscripts}"
                                )

                            for msid in manuscripts:
                                for run in range(
                                    self.config["test_runs"]["n_runs"]
                                ):  # Changed from "runs" to "test_runs.n_runs"
                                    yield {
                                        "test_name": test_name,
                                        "msid": msid,
                                        "provider": provider,
                                        "model": model["name"],
                                        "temperature": temp,
                                        "top_p": top_p,
                                        "run": run,
                                    }

    def _get_all_manuscripts(self) -> List[str]:
        """Get all available manuscript IDs from ground truth directory."""
        ground_truth_dir = Path(self.config["ground_truth_dir"])
        return [f.stem for f in ground_truth_dir.glob("*.json")]

    def _get_n_manuscripts(self, n: int) -> List[str]:
        """Get first n manuscript IDs from ground truth directory."""
        all_manuscripts = self._get_all_manuscripts()
        return all_manuscripts[:n]

    def _load_ground_truth(self, msid: str) -> Dict[str, Any]:
        """Load ground truth data for manuscript."""
        ground_truth_path = Path(self.config["ground_truth_dir"]) / f"{msid}.json"
        if not ground_truth_path.exists():
            raise FileNotFoundError(f"Ground truth not found: {ground_truth_path}")
        with open(ground_truth_path) as f:
            return json.load(f)

    def run_test(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Run single test case."""
        # Get cache path
        cache_path = self.cache.get_cache_path(
            test_case["test_name"],
            test_case["msid"],
            test_case["provider"],
            test_case["model"],
            test_case["temperature"],
            test_case["top_p"],
            test_case["run"],
        )

        # Check cache
        cached_result = self.cache.get_cached_result(cache_path)
        if cached_result is not None:
            return cached_result

        # Load ground truth
        ground_truth = self._load_ground_truth(test_case["msid"])

        # Run appropriate test
        result = self._run_specific_test(test_case, ground_truth)

        # Cache result
        self.cache.cache_result(cache_path, result)

        return result

    def _run_specific_test(
        self, test_case: Dict[str, Any], ground_truth: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run specific test based on test name."""
        test_name = test_case["test_name"]

        if test_name == "extract_sections":
            return self._run_extract_sections(test_case, ground_truth)
        elif test_name == "extract_individual_captions":
            return self._run_extract_captions(test_case, ground_truth)
        elif test_name == "assign_panel_source":
            return self._run_assign_panel_source(test_case, ground_truth)
        elif test_name == "extract_data_availability":
            return self._run_extract_data_availability(test_case, ground_truth)
        elif test_name == "assign_panel_source":
            return self._run_assign_panel_source(test_case, ground_truth)
        else:
            raise ValueError(f"Unknown test: {test_name}")

    def _run_extract_sections(
        self, test_case: Dict[str, Any], ground_truth: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run extract sections test."""
        try:
            # Get manuscript path
            msid = test_case["msid"]

            # Create temporary extraction directory
            temp_extract_dir = self.base_output_dir / "temp_extracts" / msid
            temp_extract_dir.mkdir(parents=True, exist_ok=True)

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
                "api_key": os.environ.get(f"{provider.upper()}_API_KEY"),
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

            # Get ground truth sections
            ground_truth_figures = "\n".join(
                figure["figure_caption"] for figure in ground_truth["figures"]
            )
            ground_truth_data_availability = ground_truth.get(
                "data_availability", {}
            ).get("section_text", "")

            # Create result dictionary
            result = {
                "input": str(zip_path),
                "output": figure_legends,
                "expected": ground_truth_figures,
                "data_availability_output": data_availability,
                "data_availability_expected": ground_truth_data_availability,
                "cost": updated_structure.cost.extract_sections.model_dump()
                if hasattr(updated_structure.cost.extract_sections, "model_dump")
                else {},
            }

            # Fill results bag
            self._fill_results_bag(
                test_case=test_case,
                result=result,
                actual_output=figure_legends,
                expected_output=ground_truth_figures,
            )

            return result

        except Exception as e:
            logger.error(
                f"Error running extract sections test: {str(e)}", exc_info=True
            )
            raise

        finally:
            # Cleanup temporary directory after processing
            if temp_extract_dir.exists():
                shutil.rmtree(temp_extract_dir)

    def _run_extract_captions(
        self, test_case: Dict[str, Any], ground_truth: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run extract captions test."""
        try:
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
                "api_key": os.environ.get(f"{provider.upper()}_API_KEY"),
                "pipeline": {
                    "extract_individual_captions": {
                        provider: {
                            "model": test_case["model"],
                            "temperature": test_case["temperature"],
                            "top_p": test_case["top_p"],
                            "prompts": {
                                "system": pipeline_config[
                                    "extract_individual_captions"
                                ][provider]["prompts"]["system"],
                                "user": pipeline_config["extract_individual_captions"][
                                    provider
                                ]["prompts"]["user"],
                            },
                        }
                    }
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

            # Create a ZipStructure object with just the figures
            zip_structure = ZipStructure(figures=ground_truth["figures"])

            # Extract captions using all_captions from ground truth
            extracted_zip_structure = captions_extractor.extract_individual_captions(
                doc_content=ground_truth["all_captions"], zip_structure=zip_structure
            )

            # Compare each caption and title to its ground truth
            results = []
            for figure in ground_truth["figures"]:
                figure_label = figure["figure_label"]
                expected_caption = figure["figure_caption"]
                expected_title = figure["caption_title"]

                # Get the actual caption and title from extracted results
                extracted_figure = next(
                    (
                        f
                        for f in extracted_zip_structure.figures
                        if f["figure_label"] == figure_label
                    ),
                    None,
                )

                actual_caption = (
                    extracted_figure["figure_caption"] if extracted_figure else ""
                )
                actual_title = (
                    extracted_figure["caption_title"] if extracted_figure else ""
                )

                # Fill results bag with both caption and title metrics
                self._fill_results_bag(
                    test_case=test_case,
                    result={"input": ground_truth["all_captions"]},
                    actual_output=actual_caption,
                    expected_output=expected_caption,
                    actual_title=actual_title,
                    expected_title=expected_title,
                    figure_label=figure_label,
                )

                results.append(
                    {
                        "figure_label": figure_label,
                        "actual_caption": actual_caption,
                        "expected_caption": expected_caption,
                        "actual_title": actual_title,
                        "expected_title": expected_title,
                    }
                )

            return {
                "input": ground_truth["all_captions"],
                "results": results,
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

    def _fill_results_bag(
        self,
        test_case: Dict[str, Any],
        result: Dict[str, Any],
        actual_output: str = "",
        expected_output: str = "",
        actual_title: str = "",
        expected_title: str = "",
        figure_label: str = "",
    ) -> None:
        """Fill the results DataFrame with test metrics and metadata."""
        try:
            # Create base row for DataFrame
            row = {
                "pytest_obj": None,
                "status": "completed",
                "duration_ms": 0,
                "strategy": f"{test_case['provider']}_{test_case['model']}",
                "msid": test_case["msid"],
                "run": test_case["run"],
                "task": test_case.get("test_name", ""),
                "input": result.get("input", ""),
                "actual": actual_output,
                "expected": expected_output,
                "figure_label": figure_label,
                "ai_response": result.get("ai_response", ""),
                "timestamp": pd.Timestamp.now(),
            }

            # Get metrics for this task
            metrics = get_metrics_for_task(test_case.get("test_name", ""))

            if test_case.get("test_name") == "extract_individual_captions":
                # For caption extraction, evaluate both caption and title
                test_cases = [
                    (
                        "caption",
                        LLMTestCase(
                            input=result.get("input", ""),
                            actual_output=actual_output,
                            expected_output=expected_output,
                        ),
                    ),
                    (
                        "title",
                        LLMTestCase(
                            input=result.get("input", ""),
                            actual_output=actual_title,
                            expected_output=expected_title,
                        ),
                    ),
                ]

                for prefix, test_case_obj in test_cases:
                    for metric in metrics:
                        score = metric.measure(test_case_obj)
                        row[f"{prefix}_{metric.name}"] = score
                        row[f"{prefix}_{metric.name}_success"] = metric.is_successful()
                        row[f"{prefix}_{metric.name}_threshold"] = metric.threshold

            else:
                # For all other tests, evaluate normally
                test_case_obj = LLMTestCase(
                    input=result.get("input", ""),
                    actual_output=actual_output,
                    expected_output=expected_output,
                )

                for metric in metrics:
                    score = metric.measure(test_case_obj)
                    row[metric.name] = score
                    row[f"{metric.name}_success"] = metric.is_successful()
                    row[f"{metric.name}_threshold"] = metric.threshold

            # Append to DataFrame
            self.results_df = pd.concat(
                [self.results_df, pd.DataFrame([row])], ignore_index=True
            )

        except Exception as e:
            logger.error(f"Error filling results: {str(e)}", exc_info=True)
            raise

    def cleanup(self):
        """Clean up extraction directories."""
        try:
            if self.extracts_dir.exists():
                shutil.rmtree(self.extracts_dir)
        except Exception as e:
            logger.error(f"Error cleaning up extraction directories: {str(e)}")


# Load configuration once at module level
def load_config() -> Dict[str, Any]:
    """Load benchmark configuration."""
    config_path = os.environ.get("BENCHMARK_CONFIG", "config.benchmark.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config


# Initialize runner with configuration
runner = BenchmarkRunner(load_config())

logger.info("Starting test generation")


def pytest_generate_tests(metafunc):
    """Generate test cases."""
    if "test_case" in metafunc.fixturenames:
        logger.info("Generating test cases")
        metafunc.parametrize("test_case", runner.get_test_cases())
        logger.info("Test cases generated")


logger.info("Starting test execution")


def test_pipeline(test_case: Dict[str, Any]) -> None:
    """Run pipeline test with generated test case."""
    try:
        logger.info(f"Running test case: {test_case}")

        # Run the test and collect results
        result = runner.run_test(test_case)
        logger.info("Test case completed")

        # Save results after each test
        results_path = runner.results_dir / "results.json"

        # Update results file
        if results_path.exists():
            # Load existing results
            with open(results_path) as f:
                existing_results = json.load(f)
        else:
            existing_results = []

        # Add new result
        existing_results.append(result)

        # Save updated results
        with open(results_path, "w") as f:
            json.dump(existing_results, f, indent=2)

        logger.info(f"Updated results saved to {results_path}")

    except Exception as e:
        logger.error(f"Error in test: {str(e)}", exc_info=True)
        raise
    finally:
        # Clean up any temporary files
        temp_extracts_dir = runner.base_output_dir / "temp_extracts"
        if temp_extracts_dir.exists():
            shutil.rmtree(temp_extracts_dir)


# Also add this to make sure we see all logs
@pytest.fixture(autouse=True)
def _setup_logging():
    logging.basicConfig(level=logging.INFO)
