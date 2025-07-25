"""Configuration loading with environment and pipeline support."""

import os
from enum import Enum
from typing import Any, Dict

import yaml
from dotenv import load_dotenv


class PipelineStep(Enum):
    """Enumeration of pipeline steps."""

    MANUSCRIPT_STRUCTURE = "manuscript_structure"
    ASSIGN_PANEL_SOURCE = "assign_panel_source"
    EXTRACT_CAPTIONS = "extract_captions"
    MATCH_CAPTION_PANEL = "match_caption_panel"
    OBJECT_DETECTION = "object_detection"


class ConfigurationError(Exception):
    """Custom exception for configuration errors."""

    pass


class ConfigurationLoader:
    """Handles loading and managing configuration for the pipeline."""

    def __init__(self, config_path: str):
        """
        Initialize configuration loader.

        Args:
            config_path: Path to main configuration YAML
        """
        self.config_path = config_path
        self.environment = os.getenv("ENVIRONMENT", "dev")
        self._load_environment()
        self.config = self._load_yaml_config()

    def _load_environment(self) -> None:
        """Load environment variables from appropriate .env file."""
        env_file = f".env.{self.environment}"

        load_dotenv(env_file)

        # Validate required environment variables
        required_vars = ["OPENAI_API_KEY"]

        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ConfigurationError(
                f"Missing required environment variables: {', '.join(missing_vars)}"
            )

    def _load_yaml_config(self) -> Dict[str, Any]:
        """Load and parse YAML configuration, merging top-level keys into the environment/default config."""
        try:
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
            # Get the environment or default config section
            env_config = config.get(self.environment, config.get("default", {}))
            # Merge in top-level keys (excluding known environment keys)
            merged_config = dict(env_config)  # shallow copy
            for key, value in config.items():
                if key not in (self.environment, "default"):
                    merged_config[key] = value
            return merged_config
        except Exception as e:
            raise ConfigurationError(f"Error loading configuration: {str(e)}")

    def get_pipeline_config(self, step: PipelineStep) -> Dict[str, Any]:
        """
        Get configuration for specific pipeline step.

        Args:
            step: Pipeline step to get configuration for

        Returns:
            Configuration dictionary for the step
        """
        # Special handling for steps that don't need AI configuration
        if step == PipelineStep.MANUSCRIPT_STRUCTURE:
            return {}

        if step == PipelineStep.OBJECT_DETECTION:
            return self.config["pipeline"]["object_detection"]

        # Get provider-specific configuration for AI steps
        provider = os.getenv("MODEL_PROVIDER", "openai")
        step_config = self.config["pipeline"][step.value]

        if provider not in step_config:
            raise ConfigurationError(
                f"Provider {provider} not configured for step {step.value}"
            )

        # Merge provider configuration with API key
        config = step_config[provider].copy()
        config["api_key"] = os.getenv(f"{provider.upper()}_API_KEY")

        return config

    def get_debug_config(self) -> Dict[str, Any]:
        """Get debug configuration for current environment."""
        return self.config.get("debug", {})
