"""Base class for figure caption extraction."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel

from ..manuscript_structure.manuscript_structure import (
    Figure,
    Panel,
    TokenUsage,
    ZipStructure,
)
from ..prompt_handler import PromptHandler

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


class PanelList(BaseModel):
    """Model for a list of panels."""

    panel_label: str
    panel_caption: str


class IndividualCaption(BaseModel):
    """Model for individual figure caption."""

    figure_label: str
    caption_title: str
    figure_caption: str
    panels: List[PanelList]


class ExtractedCaptions(BaseModel):
    """Model for extracted captions response."""

    figures: List[IndividualCaption]


class FigureCaptionExtractor(ABC):
    """
    Abstract base class for extracting figure captions from a document.

    Subclasses implement only the two provider-specific API calls:
    ``extract_figure_caption`` and ``extract_figure_panels``. All orchestration
    (EV-figure detection, per-figure iteration, token accumulation) lives here.
    """

    def __init__(self, config: Dict[str, Any], prompt_handler: PromptHandler):
        self.config = config
        self.prompt_handler = prompt_handler
        self._validate_config()

    @abstractmethod
    def _validate_config(self) -> None:
        pass

    @abstractmethod
    def extract_figure_caption(
        self, figure_label: str, all_captions: str, zip_structure: ZipStructure
    ) -> Tuple[CaptionExtraction, TokenUsage]:
        """Extract caption title and text for a specific figure."""
        pass

    @abstractmethod
    def extract_figure_panels(
        self, figure_label: str, caption_text: str
    ) -> Tuple[PanelExtraction, TokenUsage]:
        """Extract panels for a specific figure."""
        pass

    def is_ev_figure(self, figure_label: str) -> bool:
        """Return True if the label indicates an Extended View / Extended Data figure."""
        ev_indicators = ["EV", "Extended View", "Extended Data", "Expanded View"]
        return any(
            indicator.lower() in figure_label.lower() for indicator in ev_indicators
        )

    def process_figure(
        self, figure: Figure, all_captions: str, zip_structure: ZipStructure
    ) -> Tuple[Figure, TokenUsage]:
        """Extract caption and panels for one figure; accumulate token usage."""
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
            logger.warning(
                "Caption extraction returned empty caption",
                extra={
                    "operation": "main.extract_caption_title",
                    "figure_label": figure.figure_label,
                    "severity": "recoverable",
                    "reason": "empty_caption",
                },
            )
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
        figure.panels = [
            Panel(
                panel_label=panel_info.panel_label,
                panel_caption=panel_info.panel_caption,
            )
            for panel_info in panel_result.panels
        ]

        logger.info(f"Successfully processed {figure.figure_label}")
        return figure, total_token_usage

    def extract_individual_captions(
        self, doc_content: str, zip_structure: ZipStructure
    ) -> ZipStructure:
        """Extract individual captions for each figure in the structure."""
        logger.info("Starting extraction of individual captions")
        expected_labels = [fig.figure_label for fig in zip_structure.figures]
        logger.info(
            f"Found {len(expected_labels)} figures to process: {', '.join(expected_labels)}"
        )

        total_token_usage = TokenUsage()

        for figure in zip_structure.figures:
            if self.is_ev_figure(figure.figure_label):
                logger.info(f"Skipping EV figure: {figure.figure_label}")
                continue

            logger.info(f"Processing {figure.figure_label}")
            _, figure_token_usage = self.process_figure(
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

    def _parse_response(self, response: str) -> Dict:
        """Parse AI response containing caption data."""
        try:
            import json
            import re

            json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
            if json_match:
                response = json_match.group(1)
            else:
                json_match = re.search(r"(\{.*\})", response, re.DOTALL)
                if json_match:
                    response = json_match.group(1)

            response = re.sub(r"[\n\r\t]", " ", response)
            response = re.sub(r"\s+", " ", response)

            return json.loads(response)

        except Exception as e:
            logger.error(f"Error parsing captions: {str(e)}")
            return {}

    def _remove_duplicate_panels(self, zip_structure: ZipStructure) -> ZipStructure:
        """Remove duplicate panels (same label) from every figure."""
        for figure in zip_structure.figures:
            seen_panel_labels = set()
            unique_panels = []
            duplicated = []

            for panel in figure.panels:
                if panel.panel_label not in seen_panel_labels:
                    seen_panel_labels.add(panel.panel_label)
                    unique_panels.append(panel)
                else:
                    logger.warning(
                        f"Removing duplicate panel {panel.panel_label} "
                        f"from figure {figure.figure_label}"
                    )
                    duplicated.append(panel)

            figure.panels = unique_panels
            figure.duplicated_panels = duplicated

        return zip_structure

    def _update_figures_with_captions(
        self, zip_structure: ZipStructure, caption_data: list
    ) -> ZipStructure:
        """Update figures in ZipStructure with extracted captions and panels."""
        caption_map = {
            item["figure_label"]: {
                "caption": item["figure_caption"],
                "title": item["caption_title"],
                "panels": item.get("panels", []),
            }
            for item in caption_data
        }

        for figure in zip_structure.figures:
            if figure.figure_label in caption_map:
                caption_info = caption_map[figure.figure_label]
                figure.figure_caption = caption_info["caption"]
                figure.caption_title = caption_info["title"]
                figure.panels = [
                    Panel(
                        panel_label=panel["panel_label"],
                        panel_caption=panel["panel_caption"],
                    )
                    for panel in caption_info["panels"]
                ]
            else:
                logger.warning(f"No caption found for figure {figure.figure_label}")
                figure.figure_caption = "Figure caption not found."
                figure.caption_title = ""
                figure.panels = []

        zip_structure = self._remove_duplicate_panels(zip_structure)
        return zip_structure
