"""Anthropic Claude implementation of caption extraction."""

import json
import logging
from typing import Any, Dict, List, Tuple

import anthropic
from pydantic import BaseModel

from ..anthropic_utils import call_anthropic, validate_anthropic_model
from ..cost_tracking import update_token_usage
from ..manuscript_structure.manuscript_structure import (
    Figure,
    Panel,
    TokenUsage,
    ZipStructure,
)
from .extract_captions_base import FigureCaptionExtractor

logger = logging.getLogger(__name__)


class CaptionExtraction(BaseModel):
    """Model for caption extraction result."""

    figure_label: str
    caption_title: str
    figure_caption: str
    is_verbatim: bool


class PanelInfo(BaseModel):
    """Model for panel information."""

    panel_label: str
    panel_caption: str


class PanelExtraction(BaseModel):
    """Model for panel extraction result."""

    figure_label: str
    panels: List[PanelInfo]


class FigureCaptionExtractorAnthropic(FigureCaptionExtractor):
    """Extract figure captions using Anthropic Claude models."""

    def __init__(self, config: Dict[str, Any], prompt_handler):
        super().__init__(config, prompt_handler)
        self.client = anthropic.Anthropic()
        self.caption_config = config["pipeline"]["extract_caption_title"]["anthropic"]
        self.panel_config = config["pipeline"]["extract_panel_sequence"]["anthropic"]

    def _validate_config(self) -> None:
        """Validate Anthropic configuration parameters."""
        for step in ["extract_caption_title", "extract_panel_sequence"]:
            config_ = self.config["pipeline"][step]["anthropic"]
            validate_anthropic_model(config_.get("model", "claude-sonnet-4-6"))

    def is_ev_figure(self, figure_label: str) -> bool:
        """Check if a figure label indicates an Extended View figure."""
        ev_indicators = ["EV", "Extended View", "Extended Data", "Expanded View"]
        return any(
            indicator.lower() in figure_label.lower() for indicator in ev_indicators
        )

    def extract_figure_caption(
        self, figure_label: str, all_captions: str, zip_structure: ZipStructure
    ) -> Tuple[CaptionExtraction, TokenUsage]:
        """Extract caption title and text for a specific figure."""
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

        config_ = self.caption_config
        model_ = config_.get("model", "claude-sonnet-4-6")

        response = call_anthropic(
            client=self.client,
            model=model_,
            messages=messages,
            response_format=CaptionExtraction,
            temperature=config_.get("temperature", 0.1),
            max_tokens=config_.get("max_tokens", 4096),
        )

        token_usage = TokenUsage()
        update_token_usage(token_usage, response, model_)

        if response.choices[0].message.parsed is not None:
            caption_result = response.choices[0].message.parsed
        else:
            caption_extraction = json.loads(response.choices[0].message.content)
            caption_result = CaptionExtraction(**caption_extraction)

        return caption_result, token_usage

    def extract_figure_panels(
        self, figure_label: str, caption_text: str
    ) -> Tuple[PanelExtraction, TokenUsage]:
        """Extract panels for a specific figure."""
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

        config_ = self.panel_config
        model_ = config_.get("model", "claude-sonnet-4-6")

        response = call_anthropic(
            client=self.client,
            model=model_,
            messages=messages,
            response_format=PanelExtraction,
            temperature=config_.get("temperature", 0.1),
            max_tokens=config_.get("max_tokens", 4096),
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

    def process_figure(
        self, figure: Figure, all_captions: str, zip_structure: ZipStructure
    ) -> Tuple[Figure, TokenUsage]:
        """Process a single figure, extracting caption and panels."""
        total_token_usage = TokenUsage()

        if self.is_ev_figure(figure.figure_label):
            logger.info(f"Skipping processing for EV figure: {figure.figure_label}")
            return figure, total_token_usage

        caption_result, caption_token_usage = self.extract_figure_caption(
            figure.figure_label, all_captions, zip_structure
        )

        total_token_usage.prompt_tokens += caption_token_usage.prompt_tokens
        total_token_usage.completion_tokens += caption_token_usage.completion_tokens
        total_token_usage.total_tokens += caption_token_usage.total_tokens
        total_token_usage.cost += caption_token_usage.cost

        if not caption_result.figure_caption:
            logger.error(f"No caption extracted for {figure.figure_label}")
            figure.hallucination_score = 1
            return figure, total_token_usage

        panel_result, panel_token_usage = self.extract_figure_panels(
            figure.figure_label, caption_result.figure_caption
        )

        total_token_usage.prompt_tokens += panel_token_usage.prompt_tokens
        total_token_usage.completion_tokens += panel_token_usage.completion_tokens
        total_token_usage.total_tokens += panel_token_usage.total_tokens
        total_token_usage.cost += panel_token_usage.cost

        figure.caption_title = caption_result.caption_title
        figure.figure_caption = caption_result.figure_caption
        figure.hallucination_score = 0 if caption_result.is_verbatim else 1

        figure.panels = []
        for panel_info in panel_result.panels:
            panel = Panel(
                panel_label=panel_info.panel_label,
                panel_caption=panel_info.panel_caption,
            )
            figure.panels.append(panel)

        logger.info(f"Successfully processed {figure.figure_label}")
        return figure, total_token_usage

    def extract_individual_captions(
        self, doc_content: str, zip_structure: ZipStructure
    ) -> ZipStructure:
        """Extract individual captions for each figure in the structure."""
        logger.info("Starting extraction of individual captions")

        total_token_usage = TokenUsage()

        for figure in zip_structure.figures:
            if self.is_ev_figure(figure.figure_label):
                logger.info(f"Skipping EV figure: {figure.figure_label}")
                continue

            logger.info(f"Processing {figure.figure_label}")
            updated_figure, figure_token_usage = self.process_figure(
                figure, doc_content, zip_structure
            )

            total_token_usage.prompt_tokens += figure_token_usage.prompt_tokens
            total_token_usage.completion_tokens += figure_token_usage.completion_tokens
            total_token_usage.total_tokens += figure_token_usage.total_tokens
            total_token_usage.cost += figure_token_usage.cost

        zip_structure.cost.extract_individual_captions = total_token_usage
        zip_structure.update_total_cost()

        logger.info(
            f"Finished extracting individual captions. "
            f"Total tokens: {total_token_usage.total_tokens}"
        )
        return zip_structure
