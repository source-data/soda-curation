import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import openai
from pydantic import ValidationError

from ..ai_observability import summarize_text
from ..cost_tracking import update_token_usage
from ..openai_utils import call_openai
from ..prompt_handler import PromptHandler
from ..step_config import resolve_step_config
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
        resolved = resolve_step_config(self.config["pipeline"]["assign_panel_source"])
        if resolved["provider"] != "openai":
            raise ValueError(
                f"assign_panel_source model '{resolved['model']}' requires Anthropic."
            )

    def call_ai_service(self, prompt: str, allowed_files: List) -> AsignedFilesList:
        """Call OpenAI service with the given prompt."""
        logger.info(
            "Preparing OpenAI panel-source request",
            extra={
                "operation": "main.assign_panel_source",
                "provider": "openai",
                "prompt_summary": summarize_text(prompt),
                "allowed_file_count": len(allowed_files),
            },
        )
        # Get both system and user prompts
        prompts = self.prompt_handler.get_prompt("assign_panel_source", {})

        # Prepare messages
        messages = [
            {"role": "system", "content": prompts["system"]},
            {"role": "user", "content": prompt},
        ]

        model_ = resolve_step_config(self.config["pipeline"]["assign_panel_source"])[
            "model"
        ]

        response = call_openai(
            client=self.client,
            model=model_,
            messages=messages,
            response_format=AsignedFilesList,  # Ensure the response is in JSON format
            operation="main.assign_panel_source",
            request_metadata={
                "provider": "openai",
                "allowed_file_count": len(allowed_files),
            },
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
