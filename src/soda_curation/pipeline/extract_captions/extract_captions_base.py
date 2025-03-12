"""Base class for figure caption extraction."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from pydantic import BaseModel

from ..manuscript_structure.manuscript_structure import Panel, ZipStructure
from ..prompt_handler import PromptHandler

logger = logging.getLogger(__name__)


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

    This class provides common functionality and defines the interface that all
    figure caption extractors should implement.
    """

    def __init__(self, config: Dict[str, Any], prompt_handler: PromptHandler):
        """
        Initialize with configuration and prompt handler.

        Args:
            config: Configuration dictionary for the extractor
            prompt_handler: Handler for managing prompts

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self.prompt_handler = prompt_handler
        self._validate_config()

    @abstractmethod
    def _validate_config(self) -> None:
        """
        Validate the configuration specific to each implementation.

        Raises:
            ValueError: If configuration is invalid
        """
        pass

    @abstractmethod
    def extract_individual_captions(
        self,
        doc_content: str,
        zip_structure: ZipStructure,
    ) -> ZipStructure:
        """
        Extract captions from the figure legends section and update ZipStructure.

        Args:
            doc_content: Figure legends section content to analyze
            zip_structure: Current ZIP structure to update

        Returns:
            Updated ZipStructure with extracted captions
        """
        pass

    def _parse_response(self, response: str) -> Dict:
        """Parse AI response containing caption data."""
        try:
            import json
            import re

            # Try to extract JSON from code block if present
            json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
            if json_match:
                response = json_match.group(1)
            else:
                # Try to extract any JSON object
                json_match = re.search(r"(\{.*\})", response, re.DOTALL)
                if json_match:
                    response = json_match.group(1)

            # Clean and normalize JSON string
            response = re.sub(r"[\n\r\t]", " ", response)
            response = re.sub(r"\s+", " ", response)

            return json.loads(response)

        except Exception as e:
            logger.error(f"Error parsing captions: {str(e)}")
            return {}

    def _remove_duplicate_panels(self, zip_structure: ZipStructure) -> ZipStructure:
        """
        Remove duplicate panels from figures in the ZipStructure.

        This ensures each panel label appears only once per figure.

        Args:
            zip_structure: The ZipStructure to clean

        Returns:
            ZipStructure with duplicate panels removed
        """
        for figure in zip_structure.figures:
            # Track seen panel labels
            seen_panel_labels = set()
            unique_panels = []
            duplicated = []  # Create temporary list for duplicates

            for panel in figure.panels:
                if panel.panel_label not in seen_panel_labels:
                    seen_panel_labels.add(panel.panel_label)
                    unique_panels.append(panel)
                else:
                    logger.warning(
                        f"Removing duplicate panel {panel.panel_label} from figure {figure.figure_label}"
                    )
                    duplicated.append(panel)

            # Update the figure with unique panels and duplicates
            figure.panels = unique_panels
            figure.duplicated_panels = duplicated

        return zip_structure

    def _update_figures_with_captions(
        self, zip_structure: ZipStructure, caption_data: list
    ) -> ZipStructure:
        """Update figures in ZipStructure with extracted captions and panels.

        Args:
            zip_structure: The ZipStructure to update
            caption_data: List of dictionaries containing caption information
                Each dict should have:
                - figure_label: The label of the figure
                - caption_title: The title of the figure
                - figure_caption: The full caption text
                - panels: List of panel objects with "panel_label" and "panel_caption"

        Returns:
            Updated ZipStructure with captions and panels added to figures
        """
        # Create a mapping of figure labels to caption data for easier lookup
        caption_map = {
            item["figure_label"]: {
                "caption": item["figure_caption"],
                "title": item["caption_title"],
                "panels": item.get("panels", []),
            }
            for item in caption_data
        }

        # Update each figure in place
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

        # Remove any duplicate panels that might have been created
        zip_structure = self._remove_duplicate_panels(zip_structure)

        return zip_structure
