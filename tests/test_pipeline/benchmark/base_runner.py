"""Base class for benchmark test runners."""

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from deepeval.test_case import LLMTestCase

from src.soda_curation.pipeline.manuscript_structure.manuscript_structure import (
    CustomJSONEncoder,
)

from ..metrics import get_metrics_for_task, normalize_text

logger = logging.getLogger(__name__)


class BaseBenchmarkRunner:
    """Base class for all task-specific benchmark runners."""

    def __init__(self, config: Dict[str, Any], results_dir: Path, cache_dir: Path):
        """Initialize benchmark runner with configuration."""
        self.config = config
        self.results_dir = results_dir
        self.cache_dir = cache_dir
        self.results_df = None
        # Get cache version from config or use default
        self.cache_version = config.get("cache_version", "v1_default")

    def get_cache_path(self, test_case: Dict[str, Any]) -> Path:
        """Get path for cached result with version info."""
        return (
            self.cache_dir
            / f"{test_case['test_name']}_{test_case['msid']}_{test_case['provider']}_"
            f"{test_case['model']}_{test_case['temperature']}_{test_case['top_p']}_"
            f"{test_case['run']}_{self.cache_version}.json"
        )

    def get_cached_result(self, cache_path: Path) -> Optional[Dict[str, Any]]:
        """Get cached result if it exists."""
        if cache_path.exists():
            try:
                with open(cache_path) as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                logger.warning(f"Corrupt cache file: {cache_path}, error: {str(e)}")
                # Delete the corrupted file
                cache_path.unlink()
                return None
        return None

    def _make_serializable(self, obj):
        """Convert object to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif hasattr(obj, "to_dict"):
            # For pandas dataframes or series
            return self._make_serializable(obj.to_dict())
        elif hasattr(obj, "__dict__"):
            # For custom objects
            return self._make_serializable(obj.__dict__)
        else:
            # Check if it's a basic JSON serializable type
            try:
                json.dumps(obj)
                return obj
            except (TypeError, OverflowError):
                # If it can't be serialized, convert to string
                return str(obj)

    def cache_result(self, cache_path: Path, result: Dict[str, Any]) -> None:
        """Cache test result."""
        try:
            # Use the custom encoder specifically designed for ZipStructure and related objects
            with open(cache_path, "w") as f:
                json.dump(result, f, cls=CustomJSONEncoder)
        except Exception as e:
            logger.error(f"Error caching result: {str(e)}")
            # Delete the potentially corrupted file
            if cache_path.exists():
                cache_path.unlink()

    def _load_ground_truth(self, msid: str) -> Dict[str, Any]:
        """Load ground truth data for manuscript."""
        ground_truth_path = Path(self.config["ground_truth_dir"]) / f"{msid}.json"
        if not ground_truth_path.exists():
            raise FileNotFoundError(f"Ground truth not found: {ground_truth_path}")
        with open(ground_truth_path) as f:
            return json.load(f)

    def create_temp_extract_dir(self, msid: str) -> Path:
        """Create temporary extraction directory."""
        temp_extract_dir = Path(self.config["output_dir"]) / "temp_extracts" / msid
        temp_extract_dir.mkdir(parents=True, exist_ok=True)
        return temp_extract_dir

    def cleanup_temp_dir(self, temp_dir: Path) -> None:
        """Clean up temporary directory."""
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    def fill_results_bag(
        self,
        test_case: Dict[str, Any],
        result: Dict[str, Any],
        actual_output: str = "",
        expected_output: str = "",
        figure_label: str = "",
        panel_label: str = "",
        task: str = "",
        score: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Fill the results DataFrame with test metrics and model parameters."""

        tasks_to_normalize = [
            "locate_figure_captions",
            "extract_data_availability",
            "figure_title",
            "figure_caption",
        ]
        if task in tasks_to_normalize:
            actual_output = normalize_text(
                actual_output,
                strip_whitespace=True,
                is_data_availability="data_availability" in task,
            )
            expected_output = normalize_text(
                expected_output,
                strip_whitespace=True,
                is_data_availability="data_availability" in task,
            )

        try:
            base_row = {
                "status": "completed",
                "duration_ms": test_case.get("duration_ms", 0),
                "strategy": f"{test_case['provider']}_{test_case['model']}",
                "msid": test_case["msid"],
                "run": test_case["run"],
                "input": result.get("input", ""),
                "figure_label": figure_label,
                "panel_label": panel_label,
                "timestamp": str(pd.Timestamp.now()),  # Convert to string immediately
                "task": task,
                "actual": actual_output,
                "expected": expected_output,
                # Add all model parameters
                "model": test_case["model"],
                "temperature": test_case["temperature"],
                "top_p": test_case["top_p"],
                "frequency_penalty": test_case.get("frequency_penalty", 0.0),
                "presence_penalty": test_case.get("presence_penalty", 0.0),
                "seed": test_case["run"],  # Using run as seed
                "provider": test_case["provider"],
            }

            if score is not None:
                base_row["score"] = score
            else:
                # Get the test name from test_case
                test_name = test_case.get("test_name", "")
                if not test_name:
                    logger.warning("No test_name found in test_case, using task name")
                    test_name = task

                metric = get_metrics_for_task(test_name)[0]
                test_case_obj = LLMTestCase(
                    input=result["input"],
                    actual_output=actual_output,
                    expected_output=expected_output,
                )
                base_row["score"] = metric.measure(test_case_obj)

            # Return the row to be added to the DataFrame
            return base_row

        except Exception as e:
            logger.error(f"Error filling results: {str(e)}", exc_info=True)
            raise

    def run_test(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Base method to run a test - should be implemented by child classes."""
        raise NotImplementedError("Child classes must implement run_test")
