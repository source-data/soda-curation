"""OpenAI implementation of caption extraction."""

import json
import logging
import os
from typing import Any, Dict

import openai

from ..cost_tracking import update_token_usage
from ..manuscript_structure.manuscript_structure import Panel, ZipStructure
from ..prompt_handler import PromptHandler
from .extract_captions_base import ExtractedCaptions, FigureCaptionExtractor

logger = logging.getLogger(__name__)


class FigureCaptionExtractorOpenAI(FigureCaptionExtractor):
    """Implementation of caption extraction using OpenAI's GPT models."""

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
        # Validate model
        valid_models = ["gpt-4o", "gpt-4o-mini"]
        config_ = self.config["pipeline"]["extract_individual_captions"]["openai"]
        model = config_.get("model", "gpt-4o")
        if model not in valid_models:
            raise ValueError(f"Invalid model: {model}. Must be one of {valid_models}")

        # Validate numerical parameters
        if not 0 <= config_.get("temperature", 0.1) <= 2:
            raise ValueError(
                f"Temperature must be between 0 and 2, value: `{config_.get('temperature', 1.0)}`"
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
                f"Frequency penalty must be between -2 and 2, value: `{config_.get('frequency_penalty', 0.)}`"
            )
        if "presence_penalty" in config_ and not -2 <= config_["presence_penalty"] <= 2:
            raise ValueError(
                f"Presence penalty must be between -2 and 2, value: `{config_.get('presence_penalty', 0.)}`"
            )

    def extract_individual_captions(
        self,
        doc_content: str,
        zip_structure: ZipStructure,
    ) -> ZipStructure:
        """Extract captions from the figure legends section."""
        try:
            # Get prompts for caption extraction
            prompts = self.prompt_handler.get_prompt(
                step="extract_individual_captions",
                variables={
                    "expected_figure_count": len(zip_structure.figures),
                    "expected_figure_labels": [
                        figure.figure_label for figure in zip_structure.figures
                    ],
                    "figure_captions": doc_content,
                },
            )

            # Prepare messages
            messages = [
                {"role": "system", "content": prompts["system"]},
                {"role": "user", "content": prompts["user"]},
            ]

            config_ = self.config["pipeline"]["extract_individual_captions"]["openai"]
            model_ = config_.get("model", "gpt-4o")

            response = self.client.beta.chat.completions.parse(
                model=model_,
                messages=messages,
                response_format=ExtractedCaptions,
                temperature=config_.get("temperature", 0.1),
                top_p=config_.get("top_p", 1.0),
                frequency_penalty=config_.get("frequency_penalty", 0),
                presence_penalty=config_.get("presence_penalty", 0),
            )

            # Update token usage
            update_token_usage(
                zip_structure.cost.extract_individual_captions,
                response,
                model_,
            )

            # Parse response and update figures
            response_data = json.loads(response.choices[0].message.content)
            zip_structure = self._update_figures_with_captions(
                zip_structure, response_data["figures"]
            )

            # Store raw response
            zip_structure.ai_response_extract_individual_captions = response.choices[
                0
            ].message.content

            return zip_structure

        except Exception as e:
            logger.error(f"Error extracting individual captions: {str(e)}")
            zip_structure.ai_response_extract_individual_captions = (
                ""  # Ensure this is set to an empty string
            )
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

        return zip_structure
