"""OpenAI implementation of caption extraction."""

import json
import logging
import os
from typing import Any, Dict, List, Tuple

import openai
from pydantic import BaseModel

from ..cost_tracking import update_token_usage
from ..manuscript_structure.manuscript_structure import (
    Figure,
    Panel,
    TokenUsage,
    ZipStructure,
)
from ..openai_utils import call_openai_with_fallback, validate_model_config
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


class FigureCaptionExtractorOpenAI(FigureCaptionExtractor):
    """Implementation of caption extraction using OpenAI's GPT models."""

    def __init__(self, config: Dict[str, Any], prompt_handler):
        """Initialize with OpenAI configuration."""
        super().__init__(config, prompt_handler)

        # Initialize OpenAI client
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        self.client = openai.OpenAI(api_key=api_key)

        self.config = config
        self.prompt_handler = prompt_handler
        self.caption_config = config["pipeline"]["extract_caption_title"]["openai"]
        self.panel_config = config["pipeline"]["extract_panel_sequence"]["openai"]

    def _validate_config(self) -> None:
        """Validate OpenAI configuration parameters."""
        # Validate model
        valid_models = ["gpt-4o", "gpt-4o-mini", "gpt-5"]
        for step in ["extract_caption_title", "extract_panel_sequence"]:
            config_ = self.config["pipeline"][step]["openai"]
            model = config_.get("model", "gpt-4o")
            if model not in valid_models:
                raise ValueError(
                    f"Invalid model: {model}. Must be one of {valid_models}"
                )

            # Use the utility function for validation
            validate_model_config(model, config_)

    def is_ev_figure(self, figure_label: str) -> bool:
        """Check if a figure label indicates an Extended View (EV) figure.

        Args:
            figure_label: The figure label to check

        Returns:
            True if this is an EV figure, False otherwise
        """
        ev_indicators = ["EV", "Extended View", "Extended Data", "Expanded View"]
        return any(
            indicator.lower() in figure_label.lower() for indicator in ev_indicators
        )

    def extract_figure_caption(
        self, figure_label: str, all_captions: str, zip_structure: ZipStructure
    ) -> Tuple[CaptionExtraction, TokenUsage]:
        """Extract caption title and text for a specific figure."""
        # Special case for testing with minimal content

        # Get prompts with variables substituted
        prompts = self.prompt_handler.get_prompt(
            step="extract_caption_title",
            variables={
                "figure_label": figure_label,
                "figure_captions": all_captions,
            },
        )

        # Add JSON instruction to system prompt to satisfy the API requirement
        system_prompt = prompts["system"]
        if "json" not in system_prompt.lower():
            system_prompt += "\n\nProvide your response in JSON format."

        # Prepare messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompts["user"]},
        ]

        config_ = self.caption_config
        model_ = config_.get("model", "gpt-4o")

        response = call_openai_with_fallback(
            client=self.client,
            model=model_,
            messages=messages,
            response_format=CaptionExtraction,
            temperature=config_.get("temperature", 0.1),
            top_p=config_.get("top_p", 1.0),
            frequency_penalty=config_.get("frequency_penalty", 0),
            presence_penalty=config_.get("presence_penalty", 0),
        )
        # Create token usage object
        token_usage = TokenUsage()
        token_usage.prompt_tokens = response.usage.prompt_tokens
        token_usage.completion_tokens = response.usage.completion_tokens
        token_usage.total_tokens = response.usage.total_tokens

        # Update cost based on model and tokens
        token_usage = update_token_usage(
            token_usage,
            {
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            },
            model_,
        )

        # When using structured responses, the parsed content is in .parsed
        if hasattr(response.choices[0].message, "parsed"):
            caption_extraction = response.choices[0].message.parsed
        else:
            # Fallback for non-structured responses
            response_content = response.choices[0].message.content
            caption_extraction = json.loads(response_content)
        caption_result = CaptionExtraction(**caption_extraction)

        return caption_result, token_usage

    def extract_figure_panels(
        self, figure_label: str, caption_text: str
    ) -> Tuple[PanelExtraction, TokenUsage]:
        """Extract panels for a specific figure."""
        # Get prompts with variables substituted
        prompts = self.prompt_handler.get_prompt(
            step="extract_panel_sequence",
            variables={
                "figure_label": figure_label,
                "figure_caption": caption_text,
            },
        )

        # Add JSON instruction to system prompt to satisfy the API requirement
        system_prompt = prompts["system"]
        if "json" not in system_prompt.lower():
            system_prompt += "\n\nProvide your response in JSON format."

        # Prepare messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompts["user"]},
        ]

        config_ = self.panel_config
        model_ = config_.get("model", "gpt-4o")

        # Make direct API call
        response = call_openai_with_fallback(
            client=self.client,
            model=model_,
            messages=messages,
            response_format=PanelExtraction,
            temperature=config_.get("temperature", 0.1),
            top_p=config_.get("top_p", 1.0),
            frequency_penalty=config_.get("frequency_penalty", 0),
            presence_penalty=config_.get("presence_penalty", 0),
        )

        # Create token usage object
        token_usage = TokenUsage()
        update_token_usage(
            token_usage,
            response,
            model_,
        )

        # Parse the response content into PanelExtraction
        # When using structured responses, the parsed content is in .parsed
        if hasattr(response.choices[0].message, "parsed"):
            panel_extraction = response.choices[0].message.parsed
        else:
            # Fallback for non-structured responses
            content_json = json.loads(response.choices[0].message.content)
            panel_extraction = PanelExtraction(
                figure_label=content_json.get("figure_label", figure_label),
                panels=[PanelInfo(**panel) for panel in content_json.get("panels", [])],
            )

        return panel_extraction, token_usage

    def process_figure(
        self, figure: Figure, all_captions: str, zip_structure: ZipStructure
    ) -> Tuple[Figure, TokenUsage]:
        """Process a single figure, extracting caption and panels."""
        total_token_usage = TokenUsage()

        # try:
        # Skip EV figures explicitly
        if self.is_ev_figure(figure.figure_label):
            logger.info(f"Skipping processing for EV figure: {figure.figure_label}")
            return figure, total_token_usage

        # Step 1: Extract caption using direct API call
        caption_result, caption_token_usage = self.extract_figure_caption(
            figure.figure_label, all_captions, zip_structure
        )

        # Accumulate token usage
        total_token_usage.prompt_tokens += caption_token_usage.prompt_tokens
        total_token_usage.completion_tokens += caption_token_usage.completion_tokens
        total_token_usage.total_tokens += caption_token_usage.total_tokens
        total_token_usage.cost += caption_token_usage.cost

        # Skip panel extraction if no caption was found
        if not caption_result.figure_caption:
            logger.error(f"No caption extracted for {figure.figure_label}")
            figure.hallucination_score = 1  # Mark as non-verbatim
            return figure, total_token_usage

        # Step 2: Extract panels using direct API call
        panel_result, panel_token_usage = self.extract_figure_panels(
            figure.figure_label, caption_result.figure_caption
        )

        # Accumulate token usage
        total_token_usage.prompt_tokens += panel_token_usage.prompt_tokens
        total_token_usage.completion_tokens += panel_token_usage.completion_tokens
        total_token_usage.total_tokens += panel_token_usage.total_tokens
        total_token_usage.cost += panel_token_usage.cost

        # Update figure with extracted information
        figure.caption_title = caption_result.caption_title
        figure.figure_caption = caption_result.figure_caption

        # Set verbatim flag for hallucination scoring
        figure.hallucination_score = 0 if caption_result.is_verbatim else 1

        # Add panels to the figure
        figure.panels = []
        for panel_info in panel_result.panels:
            panel = Panel(
                panel_label=panel_info.panel_label,
                panel_caption=panel_info.panel_caption,
            )
            figure.panels.append(panel)

        logger.info(f"Successfully processed {figure.figure_label}")

        return figure, total_token_usage

        # except Exception as e:
        #     logger.error(f"Error processing {figure.figure_label}: {str(e)}")
        #     return figure, total_token_usage

    def extract_individual_captions(
        self, doc_content: str, zip_structure: ZipStructure
    ) -> ZipStructure:
        """Extract individual captions for each figure in the structure."""
        logger.info("Starting extraction of individual captions")

        # Get number of expected figures
        expected_figure_count = len(zip_structure.figures)
        expected_figure_labels = [fig.figure_label for fig in zip_structure.figures]
        logger.info(
            f"Found {expected_figure_count} figures to process: {', '.join(expected_figure_labels)}"
        )

        # Track token usage across all figures
        total_token_usage = TokenUsage()

        try:
            # Process each figure one by one
            for figure in zip_structure.figures:
                # Skip EV figures
                if self.is_ev_figure(figure.figure_label):
                    logger.info(f"Skipping EV figure: {figure.figure_label}")
                    continue

                logger.info(f"Processing {figure.figure_label}")

                # Process the figure directly (no need for async)
                updated_figure, figure_token_usage = self.process_figure(
                    figure, doc_content, zip_structure
                )

                # Update total token usage
                total_token_usage.prompt_tokens += figure_token_usage.prompt_tokens
                total_token_usage.completion_tokens += (
                    figure_token_usage.completion_tokens
                )
                total_token_usage.total_tokens += figure_token_usage.total_tokens
                total_token_usage.cost += figure_token_usage.cost

            # Store token usage in zip structure
            zip_structure.cost.extract_individual_captions = total_token_usage
            zip_structure.update_total_cost()

            logger.info(
                f"Finished extracting individual captions. Total tokens: {total_token_usage.total_tokens}"
            )

            return zip_structure

        except Exception as e:
            logger.error(f"Error extracting individual captions: {str(e)}")
            raise
