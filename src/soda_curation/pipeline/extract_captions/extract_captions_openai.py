"""OpenAI implementation of caption extraction."""

import asyncio
import logging
import os
from typing import Any, Dict, List, Tuple

import openai
from agents import Agent, ModelSettings, Runner
from pydantic import BaseModel

from ...agentic_tools import verify_caption_extraction, verify_panel_sequence
from ..cost_tracking import update_token_usage
from ..manuscript_structure.manuscript_structure import (
    Figure,
    Panel,
    TokenUsage,
    ZipStructure,
)
from ..prompt_handler import PromptHandler
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


class ExtractedCaption(BaseModel):
    """Combined model for caption and panel extraction."""

    figure_label: str
    caption_title: str
    figure_caption: str
    is_verbatim: bool
    panels: List[PanelInfo]


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

        self.config = config
        self.prompt_handler = prompt_handler
        self.caption_config = config["pipeline"]["extract_caption_title"]["openai"]
        self.panel_config = config["pipeline"]["extract_panel_sequence"]["openai"]

    def _validate_config(self) -> None:
        """Validate OpenAI configuration parameters."""
        # Validate model
        valid_models = ["gpt-4o", "gpt-4o-mini"]
        for step in ["extract_caption_title", "extract_panel_sequence"]:
            config_ = self.config["pipeline"][step]["openai"]
            model = config_.get("model", "gpt-4o")
            if model not in valid_models:
                raise ValueError(
                    f"Invalid model: {model}. Must be one of {valid_models}"
                )

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
            if (
                "presence_penalty" in config_
                and not -2 <= config_["presence_penalty"] <= 2
            ):
                raise ValueError(
                    f"Presence penalty must be between -2 and 2, value: `{config_.get('presence_penalty', 0.)}`"
                )

    async def extract_figure_caption(
        self, figure_label: str, all_captions: str
    ) -> Tuple[CaptionExtraction, TokenUsage]:
        """Extract caption title and text for a specific figure."""
        # Get the system and user prompts from the prompt handler
        prompts = self.prompt_handler.get_prompt(
            "extract_caption_title",
            {
                "figure_label": figure_label,
                "figure_captions": all_captions,
            },
        )

        # Create the caption agent with the system prompt
        caption_agent = Agent(
            name="caption_extractor",
            instructions=prompts["system"],
            output_type=CaptionExtraction,
            tools=[verify_caption_extraction],
            model=self.caption_config["model"],
            model_settings=ModelSettings(
                tool_choice="auto",
                temperature=self.caption_config["temperature"],
                top_p=self.caption_config["top_p"],
                max_tokens=self.caption_config["max_tokens"],
            ),
        )

        # Run the caption extraction agent with the user prompt
        result = await Runner.run(caption_agent, prompts["user"], max_turns=10)
        # Calculate token usage
        token_usage = TokenUsage()
        for response in result.raw_responses:
            agent_usage = response.usage
            token_usage.prompt_tokens += agent_usage.input_tokens
            token_usage.completion_tokens += agent_usage.output_tokens
            token_usage.total_tokens += agent_usage.total_tokens

            # Update cost using the update_token_usage function from cost_tracking.py
            # We'll need to create a dict that mimics the OpenAI API response format
            mock_response = {
                "usage": {
                    "prompt_tokens": agent_usage.input_tokens,
                    "completion_tokens": agent_usage.output_tokens,
                    "total_tokens": agent_usage.total_tokens,
                }
            }
            token_usage = update_token_usage(
                token_usage, mock_response, self.caption_config["model"]
            )

        return result.final_output, token_usage

    async def extract_figure_panels(
        self, figure_label: str, caption_text: str
    ) -> Tuple[PanelExtraction, TokenUsage]:
        """Extract panels for a specific figure."""
        # Get the system and user prompts from the prompt handler
        prompts = self.prompt_handler.get_prompt(
            "extract_panel_sequence",
            {
                "figure_label": figure_label,
                "figure_caption": caption_text,
            },
        )

        # Create the panel agent with the system prompt
        panel_agent = Agent(
            name="panel_extractor",
            instructions=prompts["system"],
            output_type=PanelExtraction,
            tools=[verify_panel_sequence],
            model=self.panel_config["model"],
            model_settings=ModelSettings(
                tool_choice="auto",
                temperature=self.panel_config["temperature"],
                top_p=self.panel_config["top_p"],
                max_tokens=self.panel_config["max_tokens"],
            ),
        )

        # Run the panel extraction agent with the user prompt
        result = await Runner.run(panel_agent, prompts["user"], max_turns=10)

        # Calculate token usage
        token_usage = TokenUsage()
        for response in result.raw_responses:
            agent_usage = response.usage
            token_usage.prompt_tokens += agent_usage.input_tokens
            token_usage.completion_tokens += agent_usage.output_tokens
            token_usage.total_tokens += agent_usage.total_tokens

            # Update cost
            mock_response = {
                "usage": {
                    "prompt_tokens": agent_usage.input_tokens,
                    "completion_tokens": agent_usage.output_tokens,
                    "total_tokens": agent_usage.total_tokens,
                }
            }
            token_usage = update_token_usage(
                token_usage, mock_response, self.panel_config["model"]
            )

        return result.final_output, token_usage

    async def process_figure(
        self, figure: Figure, all_captions: str
    ) -> Tuple[Figure, TokenUsage]:
        """Process a single figure, extracting caption and panels."""
        total_token_usage = TokenUsage()

        try:
            # Step 1: Extract caption title and text
            caption_result, caption_token_usage = await self.extract_figure_caption(
                figure.figure_label, all_captions
            )

            # Accumulate token usage
            total_token_usage.prompt_tokens += caption_token_usage.prompt_tokens
            total_token_usage.completion_tokens += caption_token_usage.completion_tokens
            total_token_usage.total_tokens += caption_token_usage.total_tokens
            total_token_usage.cost += caption_token_usage.cost

            # Step 2: Extract panels for this caption
            panel_result, panel_token_usage = await self.extract_figure_panels(
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

        except Exception as e:
            logger.error(f"Error processing {figure.figure_label}: {str(e)}")
            return figure, total_token_usage

    def extract_individual_captions(
        self, doc_content: str, zip_structure: ZipStructure
    ) -> ZipStructure:
        """Extract individual captions for each figure in the structure.

        Args:
            doc_content: The figure legends section content
            zip_structure: The structure containing figures

        Returns:
            Updated zip structure with extracted captions
        """
        logger.info("Starting extraction of individual captions")

        # Get number of expected figures
        expected_figure_count = len(zip_structure.figures)
        expected_figure_labels = [fig.figure_label for fig in zip_structure.figures]
        logger.info(
            f"Found {expected_figure_count} figures to process: {', '.join(expected_figure_labels)}"
        )

        # Track token usage across all figures
        total_token_usage = TokenUsage()

        # Create an event loop for running async tasks
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Process each figure one by one
            for figure in zip_structure.figures:
                logger.info(f"Processing {figure.figure_label}")

                # Run the async processing function
                updated_figure, figure_token_usage = loop.run_until_complete(
                    self.process_figure(figure, doc_content)
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
        finally:
            # Close the event loop
            loop.close()
