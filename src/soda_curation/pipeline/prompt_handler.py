"""Prompt handling utilities for pipeline components."""

import logging
from string import Template
from typing import Dict

logger = logging.getLogger(__name__)

SUPPORTED_PROVIDERS = ["openai", "anthropic"]


class PromptHandler:
    """
    Handle prompt loading and template substitution for pipeline components.

    Each pipeline step has a single set of 'system' and 'user' prompts,
    under a provider-specific config block (e.g., `openai`, `anthropic`).
    """

    def __init__(self, pipeline_config: Dict):
        """
        Initialize with pipeline configuration.

        Args:
            pipeline_config (Dict): A dict where keys are step names,
                                    and values contain provider config, e.g.:
                                    {
                                      "extract_individual_captions": {
                                        "openai": {
                                          "model": "...",
                                          "prompts": {
                                            "system": "some system prompt",
                                            "user": "some user prompt"
                                          }
                                        }
                                      },
                                      ...
                                    }
        """
        self.pipeline_config = pipeline_config or {}
        self.validate_prompts()

    def get_prompt(self, step: str, variables: Dict) -> Dict[str, str]:
        """
        Retrieve the system/user prompts for a given step, substituting variables.

        Args:
            step (str): Pipeline step name (e.g., "extract_individual_captions").
            variables (Dict): Template variables to substitute.

        Returns:
            Dict[str, str]: { "system": "...", "user": "..." }

        Raises:
            KeyError: If step not found or if prompts are missing.
        """
        # Get the step configuration
        step_config = self.pipeline_config.get(step)
        if not step_config:
            raise KeyError(f"No configuration found for step: '{step}'")

        # Identify the provider config (e.g., "openai" or "anthropic")
        provider_config = self._get_provider_config(step_config)
        if not provider_config:
            raise KeyError(f"No recognized provider config found for step '{step}'")

        # Get the single system/user prompt pair
        prompts = provider_config.get("prompts", {})
        if not prompts:
            raise KeyError(f"No prompts found in step '{step}' under provider config")

        system_str = prompts.get("system", "")
        user_str = prompts.get("user", "")

        return {
            "system": Template(system_str).safe_substitute(variables),
            "user": Template(user_str).safe_substitute(variables),
        }

    def validate_prompts(self) -> None:
        """
        Validate that all required prompts are present and well-formed.

        Raises:
            ValueError: If prompts are missing or malformed.
        """
        if not self.pipeline_config:
            raise ValueError("No pipeline configuration provided")

        for step_name, step_config in self.pipeline_config.items():
            # Skip steps that have no recognized provider (like object_detection, etc.)
            provider_config = self._get_provider_config(step_config)
            if not provider_config:
                continue  # Not an AI step or no recognized provider, skip

            # Check there's a 'prompts' dictionary with at least system and user
            prompts = provider_config.get("prompts")
            if not isinstance(prompts, dict):
                raise ValueError(
                    f"No valid 'prompts' dict in step '{step_name}' under provider config"
                )

            # We expect exactly "system" and "user"
            if "system" not in prompts or "user" not in prompts:
                raise ValueError(
                    f"Missing required 'system'/'user' prompts in step '{step_name}'"
                )

            # Validate templates by trying to compile them
            for prompt_type in ["system", "user"]:
                template_str = prompts[prompt_type]
                try:
                    Template(template_str)
                except Exception as e:
                    raise ValueError(
                        f"Invalid template in step '{step_name}' -> '{prompt_type}': {e}"
                    )

    @staticmethod
    def _get_provider_config(step_config: Dict) -> Dict:
        """
        Return the first recognized provider config (e.g., 'openai', 'anthropic')
        within the step config. If none found, returns an empty dict.
        """
        for provider in SUPPORTED_PROVIDERS:
            if provider in step_config:
                return step_config[provider]
        return {}
