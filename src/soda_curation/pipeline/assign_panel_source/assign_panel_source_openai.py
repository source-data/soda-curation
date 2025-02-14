import json
import logging
import os
from typing import Any, Dict

import openai
from pydantic import ValidationError

from ..cost_tracking import update_token_usage
from ..prompt_handler import PromptHandler
from .assign_panel_source_base import AsignedFilesList, PanelSourceAssigner

logger = logging.getLogger(__name__)


class PanelSourceAssignerOpenAI(PanelSourceAssigner):
    def __init__(self, config: Dict[str, Any], prompt_handler: PromptHandler):
        """Initialize with OpenAI configuration."""
        super().__init__(config, prompt_handler)

        # Initialize OpenAI client
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        self.client = openai.OpenAI(api_key=api_key)

    def _validate_config(self) -> None:
        """Validate OpenAI configuration parameters."""
        valid_models = [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4o-2024-08-06",
            "gpt-4o-mini-2024-07-18",
        ]
        config_ = self.config["pipeline"]["assign_panel_source"]["openai"]
        model = config_.get("model", "gpt-4o")
        if model not in valid_models:
            raise ValueError(f"Invalid model: {model}. Must be one of {valid_models}")

        # Validate numerical parameters
        if not 0 <= config_.get("temperature", 0.3) <= 2:
            raise ValueError(
                f"Temperature must be between 0 and 2, value: `{config_.get('temperature', 0.3)}`"
            )
        if not 0 <= config_.get("top_p", 1.0) <= 1:
            raise ValueError(
                f"Top_p must be between 0 and 1, value: `{config_.get('top_p', 1.0)}`"
            )
        if (
            "frequency_penalty" in config_
            and not -2 <= config_["frequency_penalty"] <= 2
        ):
            raise ValueError(
                f"Frequency penalty must be between -2 and 2, value: `{config_.get('frequency_penalty', 0)}`"
            )
        if "presence_penalty" in config_ and not -2 <= config_["presence_penalty"] <= 2:
            raise ValueError(
                f"Presence penalty must be between -2 and 2, value: `{config_.get('presence_penalty', 0)}`"
            )

    def call_ai_service(self, prompt: str) -> AsignedFilesList:
        """Call OpenAI service with the given prompt."""
        try:
            # Get both system and user prompts
            prompts = self.prompt_handler.get_prompt("assign_panel_source", {})

            # Prepare messages
            messages = [
                {
                    "role": "system",
                    "content": prompts["system"],
                },
                {"role": "user", "content": prompt},
            ]

            config_ = self.config["pipeline"]["assign_panel_source"]["openai"]
            model_ = config_.get("model", "gpt-4o")

            response = self.client.beta.chat.completions.parse(
                model=model_,
                messages=messages,
                response_format=AsignedFilesList,  # Ensure the response is in JSON format
                temperature=config_.get("temperature", 0.3),
                top_p=config_.get("top_p", 1.0),
                frequency_penalty=config_.get("frequency_penalty", 0),
                presence_penalty=config_.get("presence_penalty", 0),
            )

            # Update token usage
            update_token_usage(
                self.zip_structure.cost.assign_panel_source,
                response,
                model_,
            )

            # Parse response
            response_data = json.loads(response.choices[0].message.content)

            # Validate and create AsignedFilesList object
            assigned_files_list = AsignedFilesList(**response_data)
            return assigned_files_list

        except ValidationError as ve:
            logger.error(f"Validation error when creating AsignedFilesList: {ve}")
            return AsignedFilesList(assigned_files=[], not_assigned_files=[])

        except Exception as e:
            logger.error(f"Error calling OpenAI API: {str(e)}")
            return AsignedFilesList(assigned_files=[], not_assigned_files=[])
