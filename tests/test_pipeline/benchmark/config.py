"""Configuration handling for the benchmark system."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


class BenchmarkConfig:
    """Handle loading and accessing benchmark configuration."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration handler with optional path."""
        if config_path is None:
            config_path = os.environ.get("BENCHMARK_CONFIG", "config.benchmark.yaml")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path) as f:
            self._config = yaml.safe_load(f)

        # Initialize cache configuration
        cache_config = self._config.get("cache", {})
        self.cache_version = cache_config.get("version", "v1_default")

        # Set up directories
        self.output_dir = Path(self._config.get("output_dir", "/app/data/benchmark"))
        self.ground_truth_dir = Path(
            self._config.get("ground_truth_dir", "/app/data/ground_truth")
        )
        self.manuscript_dir = Path(
            self._config.get("manuscript_dir", "/app/data/archives")
        )

        # Validate required paths exist
        if not self.ground_truth_dir.exists():
            raise FileNotFoundError(
                f"Ground truth directory not found: {self.ground_truth_dir}"
            )
        if not self.manuscript_dir.exists():
            raise FileNotFoundError(
                f"Manuscript directory not found: {self.manuscript_dir}"
            )

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_enabled_tests(self) -> List[str]:
        """Get list of enabled tests."""
        return self._config.get("enabled_tests", [])

    def get_providers(self) -> Dict[str, Any]:
        """Get provider configurations."""
        return self._config.get("providers", {})

    def get_test_runs(self) -> Dict[str, Any]:
        """Get test run configuration."""
        return self._config.get("test_runs", {})

    def get_prompts_source(self) -> str:
        """Get path to prompts source file."""
        return self._config.get("prompts_source", "/app/config.dev.yaml")

    def get_manuscript_ids(self) -> List[str]:
        """Get list of manuscript IDs to test."""
        manuscripts = self.get_test_runs().get("manuscripts", "all")

        if isinstance(manuscripts, str) and manuscripts.lower() == "all":
            # Get all manuscript IDs from ground truth directory
            return [f.stem for f in self.ground_truth_dir.glob("*.json")]
        elif isinstance(manuscripts, int):
            # Get first N manuscript IDs
            all_manuscripts = [f.stem for f in self.ground_truth_dir.glob("*.json")]
            return all_manuscripts[:manuscripts]
        elif isinstance(manuscripts, list):
            # Use the provided list
            return manuscripts
        else:
            raise ValueError(f"Invalid manuscripts configuration: {manuscripts}")

    def get_cache_version(self) -> str:
        """Get the current cache version."""
        return self.cache_version

    def as_dict(self) -> Dict[str, Any]:
        """Return the entire configuration as a dictionary."""
        return {
            **self._config,
            "output_dir": str(self.output_dir),
            "ground_truth_dir": str(self.ground_truth_dir),
            "manuscript_dir": str(self.manuscript_dir),
            "cache_version": self.cache_version,
        }
