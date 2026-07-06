"""Anthropic Claude implementation for assigning source data files to panels."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import anthropic

from ..ai_observability import summarize_text
from ..anthropic_utils import call_anthropic, validate_anthropic_model
from ..cost_tracking import update_token_usage
from ..prompt_handler import PromptHandler
from ..step_config import resolve_step_config
from .assign_panel_source_base import (
    AsignedFiles,
    AsignedFilesList,
    PanelSourceAssigner,
)

logger = logging.getLogger(__name__)


class PanelSourceAssignerAnthropic(PanelSourceAssigner):
    """Assign source data files to panels using Anthropic Claude."""

    def __init__(
        self, config: Dict[str, Any], prompt_handler: PromptHandler, extract_dir: Path
    ):
        super().__init__(config, prompt_handler, extract_dir)
        self.client = anthropic.Anthropic()

    def _validate_config(self) -> None:
        resolved = resolve_step_config(self.config["pipeline"]["assign_panel_source"])
        if resolved["provider"] != "anthropic":
            raise ValueError(
                f"assign_panel_source model '{resolved['model']}' requires OpenAI."
            )
        validate_anthropic_model(resolved["model"])

    def call_ai_service(self, prompt: str, allowed_files: List) -> AsignedFilesList:
        """Call Claude with the given prompt and return assigned files."""
        logger.info(
            "Preparing Anthropic panel-source request",
            extra={
                "operation": "main.assign_panel_source",
                "provider": "anthropic",
                "prompt_summary": summarize_text(prompt),
                "allowed_file_count": len(allowed_files),
            },
        )
        prompts = self.prompt_handler.get_prompt("assign_panel_source", {})

        messages = [
            {"role": "system", "content": prompts["system"]},
            {"role": "user", "content": prompt},
        ]

        model_ = resolve_step_config(self.config["pipeline"]["assign_panel_source"])[
            "model"
        ]

        response = call_anthropic(
            client=self.client,
            model=model_,
            messages=messages,
            response_format=AsignedFilesList,
            operation="main.assign_panel_source",
            request_metadata={
                "provider": "anthropic",
                "allowed_file_count": len(allowed_files),
            },
        )

        update_token_usage(
            self.zip_structure.cost.assign_panel_source,
            response,
            model_,
        )

        if response.choices[0].message.parsed is not None:
            response_data = response.choices[0].message.parsed
            assigned_files = response_data.assigned_files
            not_assigned_files = response_data.not_assigned_files
        else:
            response_data = json.loads(response.choices[0].message.content)
            assigned_files = [
                AsignedFiles(**af) for af in response_data["assigned_files"]
            ]
            not_assigned_files = response_data["not_assigned_files"]

        filtered_assigned, filtered_not_assigned = self.filter_files(
            assigned_files=assigned_files,
            not_assigned_files=not_assigned_files,
            allowed_files=allowed_files,
        )

        return AsignedFilesList(
            assigned_files=filtered_assigned,
            not_assigned_files=filtered_not_assigned,
        )
