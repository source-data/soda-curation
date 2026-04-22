"""Anthropic Claude implementation for assigning source data files to panels."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import anthropic

from ..ai_observability import summarize_text
from ..anthropic_utils import call_anthropic, validate_anthropic_model
from ..cost_tracking import update_token_usage
from ..prompt_handler import PromptHandler
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
        """Validate Anthropic configuration parameters."""
        config_ = self.config["pipeline"]["assign_panel_source"]["anthropic"]
        validate_anthropic_model(config_.get("model", "claude-sonnet-4-6"))

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

        config_ = self.config["pipeline"]["assign_panel_source"]["anthropic"]
        model_ = config_.get("model", "claude-sonnet-4-6")

        response = call_anthropic(
            client=self.client,
            model=model_,
            messages=messages,
            response_format=AsignedFilesList,
            temperature=config_.get("temperature", 0.3),
            max_tokens=config_.get("max_tokens", 2048),
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
