import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import openai
from pydantic import ValidationError

from ..cost_tracking import update_token_usage
from ..openai_utils import call_openai_with_fallback, validate_model_config
from ..prompt_handler import PromptHandler
from .assign_panel_source_base import (
    AsignedFiles,
    AsignedFilesList,
    PanelSourceAssigner,
)

logger = logging.getLogger(__name__)


class PanelSourceAssignerOpenAI(PanelSourceAssigner):
    def __init__(
        self, config: Dict[str, Any], prompt_handler: PromptHandler, extract_dir: Path
    ):
        """Initialize with OpenAI configuration."""
        super().__init__(config, prompt_handler, extract_dir)
        self.client = openai.OpenAI()

    def _validate_config(self) -> None:
        """Validate OpenAI configuration parameters."""
        valid_models = [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4o-2024-08-06",
            "gpt-4o-mini-2024-07-18",
            "gpt-5",
        ]
        config_ = self.config["pipeline"]["assign_panel_source"]["openai"]
        model = config_.get("model", "gpt-4o")
        if model not in valid_models:
            raise ValueError(f"Invalid model: {model}. Must be one of {valid_models}")

        # Use the utility function for validation
        validate_model_config(model, config_)

    def call_ai_service(self, prompt: str, allowed_files: List) -> AsignedFilesList:
        """Call OpenAI service with the given prompt."""
        # Get both system and user prompts
        prompts = self.prompt_handler.get_prompt("assign_panel_source", {})

        # Prepare messages
        messages = [
            {"role": "system", "content": prompts["system"]},
            {"role": "user", "content": prompt},
        ]

        config_ = self.config["pipeline"]["assign_panel_source"]["openai"]
        model_ = config_.get("model", "gpt-4o")

        response = call_openai_with_fallback(
            client=self.client,
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
        # When using structured responses, the parsed content is in .parsed
        if hasattr(response.choices[0].message, "parsed"):
            response_data = response.choices[0].message.parsed
            # response_data is already an AsignedFilesList object
            assigned_files = response_data.assigned_files
            not_assigned_files = response_data.not_assigned_files
        else:
            # Fallback for non-structured responses
            response_data = json.loads(response.choices[0].message.content)
            assigned_files = [
                AsignedFiles(**af) for af in response_data["assigned_files"]
            ]
            not_assigned_files = response_data["not_assigned_files"]

        # Filter out invalid files
        filtered_assigned, filtered_not_assigned = self.filter_files(
            assigned_files=assigned_files,
            not_assigned_files=not_assigned_files,
            allowed_files=allowed_files,
        )

        # Create the filtered AsignedFilesList
        return AsignedFilesList(
            assigned_files=filtered_assigned,
            not_assigned_files=filtered_not_assigned,
        )

    @staticmethod
    def filter_files(
        assigned_files: List[AsignedFiles],
        not_assigned_files: List[str],
        allowed_files: List[str],
    ) -> Tuple[List[AsignedFiles], List[str]]:
        """Remove any files that are not in allowed_files."""
        filtered_assigned_files = [
            AsignedFiles(
                panel_label=af.panel_label,
                panel_sd_files=[
                    file for file in af.panel_sd_files if file in allowed_files
                ],
            )
            for af in assigned_files
        ]

        filtered_not_assigned_files = [
            file for file in not_assigned_files if file in allowed_files
        ]

        return filtered_assigned_files, filtered_not_assigned_files
