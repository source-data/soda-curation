"""
Benchmark testing framework for SODA curation pipeline.
"""

import json
import logging
import shutil
from datetime import datetime
from typing import Any, Dict, Generator

import nltk
import pandas as pd
import pytest
import yaml

from .benchmark.config import BenchmarkConfig
from .benchmark.test_assign_panel_source import PanelSourceBenchmarkRunner
from .benchmark.test_extract_data_availability import DataAvailabilityBenchmarkRunner
from .benchmark.test_extract_individual_captions import (
    CaptionsExtractionBenchmarkRunner,
)
from .benchmark.test_extract_sections import SectionsExtractionBenchmarkRunner

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

# Set logging level for specific loggers
logging.getLogger("tests.test_pipeline").setLevel(logging.DEBUG)
logging.getLogger("src.soda_curation").setLevel(logging.DEBUG)

# Download required NLTK data
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt")
    nltk.download("punkt_tab")

# Load configuration
config = BenchmarkConfig()

# Initialize the results directory
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
results_dir = config.output_dir / timestamp
results_dir.mkdir(parents=True, exist_ok=True)

# Initialize cache directory
cache_dir = config.output_dir / "cache"
cache_dir.mkdir(parents=True, exist_ok=True)

# Initialize task-specific runners
runners = {
    "extract_sections": SectionsExtractionBenchmarkRunner(
        config.as_dict(), results_dir, cache_dir
    ),
    "extract_individual_captions": CaptionsExtractionBenchmarkRunner(
        config.as_dict(), results_dir, cache_dir
    ),
    "assign_panel_source": PanelSourceBenchmarkRunner(
        config.as_dict(), results_dir, cache_dir
    ),
    "extract_data_availability": DataAvailabilityBenchmarkRunner(
        config.as_dict(), results_dir, cache_dir
    ),
}

# Initialize results DataFrame
results_df = pd.DataFrame()


def get_test_cases() -> Generator[Dict[str, Any], None, None]:
    """Generate test cases from configuration."""
    for provider, prov_config in config.get_providers().items():
        for model in prov_config["models"]:
            for temp in model["temperatures"]:
                for top_p in model["top_p"]:
                    # Get default penalties from dev config
                    with open(config.get_prompts_source()) as f:
                        dev_config = yaml.safe_load(f)
                    default_config = dev_config.get("default", {})

                    for test_name in config.get_enabled_tests():
                        for msid in config.get_manuscript_ids():
                            for run in range(config.get_test_runs().get("n_runs", 1)):
                                yield {
                                    "test_name": test_name,
                                    "msid": msid,
                                    "provider": provider,
                                    "model": model["name"],
                                    "temperature": temp,
                                    "top_p": top_p,
                                    "frequency_penalty": default_config.get(
                                        "frequency_penalty", 0.0
                                    ),
                                    "presence_penalty": default_config.get(
                                        "presence_penalty", 0.0
                                    ),
                                    "run": run,
                                }


def pytest_generate_tests(metafunc):
    """Generate test cases for pytest."""
    if "test_case" in metafunc.fixturenames:
        logger.info("Generating test cases")
        metafunc.parametrize("test_case", get_test_cases())
        logger.info("Test cases generated")


def test_pipeline(test_case: Dict[str, Any]) -> None:
    """Run pipeline test with generated test case."""
    global results_df

    try:
        logger.info(
            f"Starting test case: {test_case['test_name']} - {test_case['msid']} - {test_case['provider']} - {test_case['model']}"
        )

        # Get the appropriate runner for this test
        runner = runners[test_case["test_name"]]

        # Run the test
        result = runner.run_test(test_case)
        logger.info("Test case completed")

        # Add results to DataFrame
        if "results" in result:
            for row in result["results"]:
                if row:  # Check if row is not None
                    results_df = pd.concat(
                        [results_df, pd.DataFrame([row])], ignore_index=True
                    )

        # Save results after each test
        results_path = results_dir / "results.json"

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

        # Save metrics DataFrame
        metrics_path = results_dir / "metrics.csv"
        results_df.to_csv(metrics_path, index=False)

        logger.info(f"Updated results saved to {results_path}")

    except Exception as e:
        logger.error(f"Error in test: {str(e)}", exc_info=True)
        raise
    finally:
        # Clean up any temporary files
        temp_extracts_dir = config.output_dir / "temp_extracts"
        if temp_extracts_dir.exists():
            shutil.rmtree(temp_extracts_dir)


def run_benchmarks_directly():
    """
    Run the benchmarks directly without pytest.
    This can be useful for debugging or running without the test framework.
    """
    for test_case in get_test_cases():
        test_pipeline(test_case)

    logger.info("All benchmarks completed.")
    logger.info(f"Results saved to {results_dir}")


if __name__ == "__main__":
    # This allows running the benchmarks directly with python -m tests.test_pipeline.benchmark
    run_benchmarks_directly()


# Fixture for pytest to setup logging
@pytest.fixture(autouse=True)
def setup_logging():
    """Configure logging for tests."""
    logger.info("Logging configured for tests")
    return None
