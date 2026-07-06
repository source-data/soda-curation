"""Anthropic Claude implementation of caption extraction."""

import json
import logging
from typing import Any, Dict, Tuple

import anthropic

from ..ai_observability import summarize_text
from ..anthropic_utils import call_anthropic, validate_anthropic_model
from ..cost_tracking import update_token_usage
from ..manuscript_structure.manuscript_structure import TokenUsage, ZipStructure
from ..step_config import resolve_step_config
from .extract_captions_base import (
    CaptionExtraction,
    FigureCaptionExtractor,
    PanelExtraction,
    PanelInfo,
)

logger = logging.getLogger(__name__)


class FigureCaptionExtractorAnthropic(FigureCaptionExtractor):
    """Extract figure captions using Anthropic Claude models."""

    def __init__(self, config: Dict[str, Any], prompt_handler):
        super().__init__(config, prompt_handler)
        self.client = anthropic.Anthropic()

    def _validate_config(self) -> None:
        for step in ("extract_caption_title", "extract_panel_sequence"):
            resolved = resolve_step_config(self.config["pipeline"][step])
            if resolved["provider"] != "anthropic":
                raise ValueError(
                    f"{step} model '{resolved['model']}' requires the OpenAI extractor."
                )
            validate_anthropic_model(resolved["model"])

    def extract_figure_caption(
        self, figure_label: str, all_captions: str, zip_structure: ZipStructure
    ) -> Tuple[CaptionExtraction, TokenUsage]:
        """Extract caption title and text for a specific figure."""
        logger.info(
            "Preparing figure caption extraction request",
            extra={
                "operation": "main.extract_caption_title",
                "provider": "anthropic",
                "figure_label": figure_label,
                "captions_summary": summarize_text(all_captions),
            },
        )

        prompts = self.prompt_handler.get_prompt(
            step="extract_caption_title",
            variables={
                "figure_label": figure_label,
                "figure_captions": all_captions,
            },
        )

        system_prompt = prompts["system"]
        if "json" not in system_prompt.lower():
            system_prompt += "\n\nProvide your response in JSON format."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompts["user"]},
        ]

        model_ = resolve_step_config(self.config["pipeline"]["extract_caption_title"])[
            "model"
        ]

        response = call_anthropic(
            client=self.client,
            model=model_,
            messages=messages,
            response_format=CaptionExtraction,
            operation="main.extract_caption_title",
            request_metadata={"figure_label": figure_label},
        )

        token_usage = TokenUsage()
        update_token_usage(token_usage, response, model_)

        if response.choices[0].message.parsed is not None:
            caption_result = response.choices[0].message.parsed
        else:
            caption_result = CaptionExtraction(
                **json.loads(response.choices[0].message.content)
            )

        return caption_result, token_usage

    def extract_figure_panels(
        self, figure_label: str, caption_text: str
    ) -> Tuple[PanelExtraction, TokenUsage]:
        """Extract panels for a specific figure."""
        logger.info(
            "Preparing panel sequence extraction request",
            extra={
                "operation": "main.extract_panel_sequence",
                "provider": "anthropic",
                "figure_label": figure_label,
                "caption_summary": summarize_text(caption_text),
            },
        )

        prompts = self.prompt_handler.get_prompt(
            step="extract_panel_sequence",
            variables={
                "figure_label": figure_label,
                "figure_caption": caption_text,
            },
        )

        system_prompt = prompts["system"]
        if "json" not in system_prompt.lower():
            system_prompt += "\n\nProvide your response in JSON format."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompts["user"]},
        ]

        model_ = resolve_step_config(self.config["pipeline"]["extract_panel_sequence"])[
            "model"
        ]

        response = call_anthropic(
            client=self.client,
            model=model_,
            messages=messages,
            response_format=PanelExtraction,
            operation="main.extract_panel_sequence",
            request_metadata={"figure_label": figure_label},
        )

        token_usage = TokenUsage()
        update_token_usage(token_usage, response, model_)

        if response.choices[0].message.parsed is not None:
            panel_extraction = response.choices[0].message.parsed
        else:
            content_json = json.loads(response.choices[0].message.content)
            panel_extraction = PanelExtraction(
                figure_label=content_json.get("figure_label", figure_label),
                panels=[PanelInfo(**p) for p in content_json.get("panels", [])],
            )

        return panel_extraction, token_usage
