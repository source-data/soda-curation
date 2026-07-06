"""Prompt handling utilities for pipeline components."""

import logging
from string import Template
from typing import Dict

from .step_config import get_step_prompts, is_ai_step

logger = logging.getLogger(__name__)


class PromptHandler:
    """
    Handle prompt loading and template substitution for pipeline components.

    Each AI pipeline step has a single shared ``prompts`` block (``system`` + ``user``)
    alongside a ``model`` field. Provider-specific nesting is no longer required.
    """

    def __init__(self, pipeline_config: Dict):
        """
        Initialize with pipeline configuration.

        Args:
            pipeline_config (Dict): ``config["pipeline"]`` mapping step names to
                step config (``model`` + ``prompts`` for AI steps).
        """
        self.pipeline_config = pipeline_config or {}
        self.validate_prompts()

    def get_prompt(self, step: str, variables: Dict) -> Dict[str, str]:
        """
        Retrieve the system/user prompts for a given step, substituting variables.

        Args:
            step (str): Pipeline step name (e.g., ``extract_sections``).
            variables (Dict): Template variables to substitute.

        Returns:
            Dict[str, str]: ``{"system": "...", "user": "..."}``

        Raises:
            KeyError: If step not found or if prompts are missing.
        """
        step_config = self.pipeline_config.get(step)
        if not step_config:
            raise KeyError(f"No configuration found for step: '{step}'")

        prompts = get_step_prompts(step_config)
        return {
            "system": Template(prompts["system"]).safe_substitute(variables),
            "user": Template(prompts["user"]).safe_substitute(variables),
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
            if not is_ai_step(step_config):
                continue

            prompts = get_step_prompts(step_config)
            for prompt_type in ("system", "user"):
                try:
                    Template(prompts[prompt_type])
                except Exception as e:
                    raise ValueError(
                        f"Invalid template in step '{step_name}' -> '{prompt_type}': {e}"
                    ) from e
