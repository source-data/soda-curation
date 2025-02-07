"""Base class for figure caption extraction."""

import logging
from abc import ABC, abstractmethod
from typing import Dict

from ..manuscript_structure.manuscript_structure import ZipStructure
from ..prompt_handler import PromptHandler

logger = logging.getLogger(__name__)


class FigureCaptionExtractor(ABC):
    """
    Abstract base class for extracting figure captions from a document.

    This class provides common functionality and defines the interface that all
    figure caption extractors should implement.
    """

    def __init__(self, config: Dict, prompt_handler: PromptHandler):
        """
        Initialize with configuration and prompt handler.

        Args:
            config: Configuration dictionary for the extractor
            prompt_handler: Handler for managing prompts
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
    def locate_captions(
        self,
        doc_content: str,
        zip_structure: ZipStructure,
        expected_figure_count: int,
        expected_figure_labels: str,
    ) -> str:
        """
        Extract captions from document content and update ZipStructure.

        This method must be implemented by each provider to define their specific
        approach to caption extraction.

        Args:
            doc_content: Document content to analyze (html formatted)
            zip_structure: Current ZIP structure to update
            expected_figure_count: Expected number of figures
            expected_figure_labels: Expected figure labels

        Returns:
            str: The section containing the figure legends
        """
        pass

    @abstractmethod
    def extract_individual_captions(
        self,
        caption_section: str,
        zip_structure: ZipStructure,
        expected_figure_count: int,
        expected_figure_labels: str,
    ) -> dict:
        """
        Extract captions from the individual figures.

        This method must be implemented by each provider to define their specific
        approach to caption extraction.

        Args:
            doc_content: Document content to analyze
            zip_structure: Current ZIP structure to update
            expected_figure_count: Expected number of figures
            expected_figure_labels: Expected figure labels

        Returns:
            dict: Object including the individual captions, titles, and figure labels.
        """
        pass

    def _parse_captions(self, response: str) -> Dict:
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

    def _update_figures_with_captions(
        self, zip_structure: ZipStructure, caption_data: list
    ) -> ZipStructure:
        """Update figures in ZipStructure with extracted captions.

        Args:
            zip_structure: The ZipStructure to update
            caption_data: List of dictionaries containing caption information
                Each dict should have:
                - figure_label: The label of the figure
                - caption_title: The title of the figure
                - figure_caption: The full caption text

        Returns:
            Updated ZipStructure with captions added to figures
        """
        # Create a mapping of figure labels to caption data for easier lookup
        caption_map = {
            item["figure_label"]: {
                "caption": item["figure_caption"],
                "title": item["caption_title"],
            }
            for item in caption_data
        }

        # Update each figure in place
        for figure in zip_structure.figures:
            if figure.figure_label in caption_map:
                caption_info = caption_map[figure.figure_label]
                figure.figure_caption = caption_info["caption"]
                figure.caption_title = caption_info["title"]
            else:
                logger.warning(f"No caption found for figure {figure.figure_label}")
                figure.figure_caption = "Figure caption not found."
                figure.caption_title = ""

        return zip_structure
