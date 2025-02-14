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
from deepeval import assert_test
from deepeval.test_case import LLMTestCase

from src.soda_curation.pipeline.extract_sections.extract_sections_openai import (
    SectionExtractorOpenAI,
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


# Load configuration once at module level
def load_config() -> Dict[str, Any]:
    """Load benchmark configuration."""
    config_path = os.environ.get("BENCHMARK_CONFIG", "config.benchmark.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config


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
        """Generate cache file path."""
        cache_key = (
            f"{test_name}_{msid}_{provider}_{model}_t{temperature}_p{top_p}_r{run}"
        )
        return self.cache_dir / f"{cache_key}.json"

    def get_cached_result(self, cache_path: Path) -> Optional[Any]:
        """Get cached result if it exists."""
        if cache_path.exists():
            with open(cache_path) as f:
                return json.load(f)
        return None

    def cache_result(self, cache_path: Path, result: Dict[str, Any]) -> None:
        """Cache test result."""
        with open(cache_path, "w") as f:
            json.dump(result, f, indent=2)


class BenchmarkRunner:
    """Main class for running benchmarks."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results_dir = Path(config["output_dir"]) / datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S"
        )
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.extracts_dir = self.results_dir / "extracts"
        self.extracts_dir.mkdir(exist_ok=True)
        self.cache = BenchmarkCache(self.results_dir / "cache")

    def get_test_cases(self) -> Generator[Dict[str, Any], None, None]:
        """Generate test cases from configuration."""
        # Get providers and models
        for provider, prov_config in self.config["providers"].items():
            for model in prov_config["models"]:
                for temp in model["temperatures"]:
                    for top_p in model["top_p"]:
                        # Get manuscripts
                        manuscripts = self._get_manuscripts()
                        for msid in manuscripts:
                            # Get enabled tests
                            for test_name in self.config["enabled_tests"]:
                                # Get number of runs
                                for run in range(self.config["test_runs"]["n_runs"]):
                                    yield {
                                        "test_name": test_name,
                                        "msid": msid,
                                        "provider": provider,
                                        "model": model["name"],
                                        "temperature": temp,
                                        "top_p": top_p,
                                        "run": run,
                                    }

    def cleanup(self):
        """Clean up extraction directories."""
        try:
            if self.extracts_dir.exists():
                shutil.rmtree(self.extracts_dir)
                logger.info(f"Cleaned up extraction directory: {self.extracts_dir}")
        except Exception as e:
            logger.error(f"Error cleaning up extraction directory: {str(e)}")

    def cleanup_manuscript(self, msid: str):
        """Clean up extraction directory for a specific manuscript."""
        try:
            manuscript_dir = self.extracts_dir / msid
            if manuscript_dir.exists():
                shutil.rmtree(manuscript_dir)
                logger.info(f"Cleaned up manuscript directory: {manuscript_dir}")
        except Exception as e:
            logger.error(f"Error cleaning up manuscript directory: {str(e)}")

    def _get_manuscripts(self) -> List[str]:
        """Get list of manuscripts to process."""
        manuscripts_config = self.config["test_runs"]["manuscripts"]
        ground_truth_dir = Path(self.config["ground_truth_dir"])

        all_manuscripts = sorted([f.stem for f in ground_truth_dir.glob("*.json")])

        if manuscripts_config == "all":
            return all_manuscripts
        elif isinstance(manuscripts_config, int):
            return all_manuscripts[:manuscripts_config]
        elif isinstance(manuscripts_config, list):
            return manuscripts_config
        raise ValueError(f"Invalid manuscripts configuration: {manuscripts_config}")

    def _load_ground_truth(self, msid: str) -> Dict[str, Any]:
        """Load ground truth data for manuscript."""
        ground_truth_file = Path(self.config["ground_truth_dir"]) / f"{msid}.json"
        with open(ground_truth_file) as f:
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
        elif test_name == "extract_captions":
            return self._run_extract_captions(test_case, ground_truth)
        elif test_name == "assign_panel_source":
            return self._run_assign_panel_source(test_case, ground_truth)
        elif test_name == "match_caption_panel":
            return self._run_match_caption_panel(test_case, ground_truth)
        elif test_name == "extract_data_availability":
            return self._run_extract_data_availability(test_case, ground_truth)
        else:
            raise ValueError(f"Unknown test: {test_name}")

    def save_results(self, results: List[Dict[str, Any]]) -> None:
        """Save benchmark results."""
        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Save detailed results
        results_file = self.results_dir / "results.json"
        df.to_json(results_file, orient="records", indent=2)

        # Generate and save summary
        summary = self._generate_summary(df)
        summary_file = self.results_dir / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # Save configuration
        config_file = self.results_dir / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(self.config, f)

        logger.info(f"Results saved to {self.results_dir}")

    def _generate_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary of results."""
        summary = {}

        # Overall summary
        summary["overall"] = {
            metric: df[metric].mean()
            for metric in self.config["results"]["metrics"]
            if metric in df.columns
        }

        # By test
        summary["by_test"] = {}
        for test in df["test_name"].unique():
            test_df = df[df["test_name"] == test]
            summary["by_test"][test] = {
                metric: test_df[metric].mean()
                for metric in self.config["results"]["metrics"]
                if metric in test_df.columns
            }

        # By model
        summary["by_model"] = {}
        for provider in df["provider"].unique():
            summary["by_model"][provider] = {}
            provider_df = df[df["provider"] == provider]

            for model in provider_df["model"].unique():
                model_df = provider_df[provider_df["model"] == model]
                summary["by_model"][provider][model] = {
                    metric: model_df[metric].mean()
                    for metric in self.config["results"]["metrics"]
                    if metric in model_df.columns
                }

        return summary

    def _run_extract_sections(
        self, test_case: Dict[str, Any], ground_truth: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run extract sections test."""
        try:
            from src.soda_curation.pipeline.manuscript_structure.manuscript_xml_parser import (
                XMLStructureExtractor,
            )

            # Get manuscript path
            msid = test_case["msid"]

            # Setup extraction directory for this test
            extract_dir = self.results_dir / "extracts" / msid
            extract_dir.mkdir(parents=True, exist_ok=True)

            # Get ZIP path
            zip_path = Path(self.config["manuscript_dir"]) / f"{msid}.zip"
            if not zip_path.exists():
                raise FileNotFoundError(f"ZIP file not found: {zip_path}")

            # Use XML extractor to get structure and extract files
            extractor = XMLStructureExtractor(str(zip_path), str(extract_dir))
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

            # Return results in the format expected by the test framework
            return {
                "input": str(zip_path),
                "output": figure_legends,
                "expected": ground_truth_figures,
                "data_availability_output": data_availability,
                "data_availability_expected": ground_truth_data_availability,
                "cost": updated_structure.cost.extract_sections.model_dump()
                if hasattr(updated_structure.cost.extract_sections, "model_dump")
                else {},
            }

        except Exception as e:
            logger.error(
                f"Error running extract sections test: {str(e)}", exc_info=True
            )
            raise


# Initialize benchmark configuration
config = load_config()
runner = BenchmarkRunner(config)


def pytest_generate_tests(metafunc) -> None:
    """Generate test cases for pytest."""
    if "test_case" in metafunc.fixturenames:
        test_cases = list(runner.get_test_cases())
        metafunc.parametrize("test_case", test_cases)


def test_pipeline(test_case: Dict[str, Any]) -> Dict[str, Any]:
    """Run pipeline test with generated test case."""
    try:
        # Run test
        result = runner.run_test(test_case)
        # Preprocess the texts
        if isinstance(result["output"], str):
            result["output"] = preprocess_text(result["output"])
        if isinstance(result["expected"], str):
            result["expected"] = preprocess_text(result["expected"])

        # Create test case
        deepeval_test = LLMTestCase(
            input=result["input"],
            actual_output=result["output"],
            expected_output=result["expected"],
        )

        # Get metrics
        metrics = get_metrics_for_task(test_case["test_name"])

        # Calculate scores
        scores = {}
        for metric in metrics:
            score = metric.measure(deepeval_test)
            scores[metric.name] = score

        # Store results
        test_result = {
            **test_case,
            **scores,
            "input": result["input"],
            "output": result["output"],
            "expected": result["expected"],
        }

        # Add test result to session
        pytest.test_result = test_result

        # Run assertions (this will raise AssertionError if any metric fails)
        assert_test(deepeval_test, metrics=metrics)

        return test_result

    except Exception as e:
        logger.error(f"Error in test: {str(e)}", exc_info=True)
        raise


def pytest_sessionfinish(session, exitstatus) -> None:
    """Save results and clean up after all tests complete."""
    try:
        results = []

        # Collect results from all tests
        for item in session.items:
            if hasattr(pytest, "test_result"):
                results.append(pytest.test_result)
                delattr(pytest, "test_result")  # Clean up after collecting

        if results:  # Only save if we have results
            runner.save_results(results)
            logger.info(f"Saved {len(results)} test results")
        else:
            logger.warning("No test results to save")

        # Clean up all extraction directories
        runner.cleanup()

    except Exception as e:
        logger.error(f"Error in session finish: {str(e)}", exc_info=True)
        # Try cleanup even if there was an error
        try:
            runner.cleanup()
        except Exception:
            pass
