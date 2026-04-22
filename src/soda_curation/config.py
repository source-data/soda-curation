"""Configuration loading with environment and pipeline support."""

import os
from enum import Enum
from typing import Any, Dict, cast

import yaml  # type: ignore[import-untyped]
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
        # Load base .env first (if present), then environment-specific values.
        # This allows shared keys (e.g., LANGFUSE_*) to live in .env while
        # environment-specific overrides stay in .env.<environment>.
        load_dotenv(".env")
        env_file = f".env.{self.environment}"
        load_dotenv(env_file, override=True)

        # Validate required environment variables.
        # If MODEL_PROVIDER is set, require the matching key; otherwise require at
        # least one supported provider key so standalone QC (Anthropic/Gemini) can run.
        provider = os.getenv("MODEL_PROVIDER", "").strip().lower()
        provider_key_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "gemini": "GOOGLE_API_KEY",
            "google": "GOOGLE_API_KEY",
        }
        if provider in provider_key_map:
            required = provider_key_map[provider]
            if not os.getenv(required):
                raise ConfigurationError(
                    f"Missing required environment variable for provider "
                    f"'{provider}': {required}"
                )
            return

        if not any(
            os.getenv(key)
            for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY")
        ):
            raise ConfigurationError(
                "Missing required environment variables: provide at least one of "
                "OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY"
            )

    def _load_yaml_config(self) -> Dict[str, Any]:
        """Load and parse YAML configuration, merging top-level keys into the environment/default config."""
        try:
            with open(self.config_path, "r") as f:
                loaded = yaml.safe_load(f)
            if loaded is None:
                config: Dict[str, Any] = {}
            elif isinstance(loaded, dict):
                config = cast(Dict[str, Any], loaded)
            else:
                raise ConfigurationError(
                    "Top-level YAML config must be a mapping/dictionary."
                )
            # Get the environment or default config section
            env_config = config.get(self.environment, config.get("default", {}))
            if not isinstance(env_config, dict):
                raise ConfigurationError(
                    f"Configuration section '{self.environment}' must be a dictionary."
                )
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
            return cast(Dict[str, Any], self.config["pipeline"]["object_detection"])

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

        return cast(Dict[str, Any], config)

    def get_debug_config(self) -> Dict[str, Any]:
        """Get debug configuration for current environment."""
        debug_cfg = self.config.get("debug", {})
        if isinstance(debug_cfg, dict):
            return cast(Dict[str, Any], debug_cfg)
        return {}


def load_config(config_path: str) -> Dict[str, Any]:
    """Load merged pipeline configuration from a YAML file (backward-compatible API)."""
    loader = ConfigurationLoader(config_path)
    return cast(Dict[str, Any], loader.config)
