"""Prompt handling utilities for pipeline components."""

import logging
from string import Template
from typing import Dict

logger = logging.getLogger(__name__)


class PromptHandler:
    """
    Handle prompt loading and template substitution for pipeline components.

    This class manages prompts for pipeline steps, handling template substitution
    and providing a consistent interface for prompt access.
    """

    def __init__(self, pipeline_config: Dict):
        """
        Initialize with pipeline configuration.

        Args:
            pipeline_config: Pipeline configuration containing steps and their prompts
        """
        self.pipeline_config = pipeline_config
        self.validate_prompts()

    def get_prompt(self, step: str, task: str, variables: Dict) -> Dict[str, str]:
        """
        Get system and user prompts for a specific pipeline step and task.

        Args:
            step: Pipeline step (e.g., 'extract_captions', 'match_caption_panel')
            task: Specific task within the step
            variables: Dictionary of variables to substitute in templates

        Returns:
            Dict with 'system' and 'user' prompts

        Raises:
            KeyError: If prompts for step/task not found in config
        """
        try:
            # Get step configuration
            step_config = self.pipeline_config[step]
            if not step_config:
                raise KeyError(f"No configuration found for step: {step}")

            # Get provider config (e.g., openai, anthropic)
            provider_config = next(
                iter(v for k, v in step_config.items() if k in ["openai", "anthropic"])
            )

            # Get prompts for task
            prompts = provider_config.get("prompts", {}).get(task, {})
            if not prompts:
                raise KeyError(f"No prompts found for task: {task} in step: {step}")

            # Substitute variables in templates
            return {
                "system": Template(prompts.get("system", "")).safe_substitute(
                    variables
                ),
                "user": Template(prompts.get("user", "")).safe_substitute(variables),
            }

        except Exception as e:
            logger.error(f"Error getting prompts for {step}.{task}: {str(e)}")
            raise

    def validate_prompts(self) -> None:
        """
        Validate that all required prompts are present and well-formed.

        Raises:
            ValueError: If prompts are missing or malformed
        """
        if not self.pipeline_config:
            raise ValueError("No pipeline configuration provided")

        # Check each pipeline step
        for step, step_config in self.pipeline_config.items():
            # Skip non-AI steps like object_detection
            if step == "object_detection":
                continue

            # Get provider config
            provider_config = next(
                iter(v for k, v in step_config.items() if k in ["openai", "anthropic"]),
                None,
            )
            if not provider_config:
                raise ValueError(f"No provider configuration found for step {step}")

            # Check prompts structure
            prompts = provider_config.get("prompts", {})
            if not prompts:
                raise ValueError(f"No prompts defined for step {step}")

            # Check each task's prompts
            for task, task_prompts in prompts.items():
                if not isinstance(task_prompts, dict):
                    raise ValueError(f"Invalid structure for task {task} in {step}")

                # Check required prompt types exist
                required_types = ["system", "user"]
                missing = [t for t in required_types if t not in task_prompts]
                if missing:
                    raise ValueError(
                        f"Missing required prompts {missing} for {step}.{task}"
                    )

                # Validate templates
                for prompt_type, template in task_prompts.items():
                    try:
                        Template(template)
                    except Exception as e:
                        raise ValueError(
                            f"Invalid template in {step}.{task}.{prompt_type}: {str(e)}"
                        )
