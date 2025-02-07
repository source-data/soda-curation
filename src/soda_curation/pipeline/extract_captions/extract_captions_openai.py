"""OpenAI implementation of caption extraction."""

import json
import logging
import os
from typing import Dict, List

import openai
from pydantic import BaseModel

from ..cost_tracking import update_token_usage
from ..manuscript_structure.manuscript_structure import ZipStructure

# from ..prompt_handler import PromptHandler
from .extract_captions_base import FigureCaptionExtractor

logger = logging.getLogger(__name__)


class LocateFigureCaptions(BaseModel):
    figure_legends: str


class IndividualCaption(BaseModel):
    figure_label: str
    caption_title: str
    figure_caption: str


class ExtractIndividualCaptions(BaseModel):
    figures: List[IndividualCaption]


class FigureCaptionExtractorOpenAI(FigureCaptionExtractor):
    """Implementation of caption extraction using OpenAI's GPT models."""

    def __init__(self, config: Dict, prompt_handler):
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
        for step_ in ["locate_captions", "extract_individual_captions"]:
            config_ = self.config["pipeline"][step_]["openai"]
            model_ = config_.get("model", "gpt-4o")
            if model_ not in valid_models:
                raise ValueError(
                    f"Invalid model: {model_}. Must be one of {valid_models}"
                )

            # Validate numerical parameters
            if not 0 <= config_.get("temperature", 0.1) <= 2:
                raise ValueError(
                    f"Temperature must be between 0 and 2 for step: `{step_}`, value: `{config_.get('temperature', 1.0)}`"
                )
            if not 0 <= config_.get("top_p", 1.0) <= 1:
                raise ValueError(
                    f"Top_p must be between 0 and 1 for step: `{step_}`, value: `{config_.get('top_p', 1.0)}`"
                )
            if (
                "frequency_penalty" in config_
                and not -2 <= config_["frequency_penalty"] <= 2
            ):
                raise ValueError(
                    f"Frequency penalty must be between -2 and 2 for step: `{step_}`, value: `{config_.get('frequency_penalty', 0.)}`"
                )
            if (
                "presence_penalty" in config_
                and not -2 <= config_["presence_penalty"] <= 2
            ):
                raise ValueError(
                    f"Presence penalty must be between -2 and 2 for step: `{step_}`, value: `{config_.get('presence_penalty', 0.)}`"
                )

    def locate_captions(
        self,
        doc_content: str,
        zip_structure: ZipStructure,
    ) -> (str, ZipStructure):
        """Call OpenAI API to locate figure captions."""
        # Get prompts with variables substituted
        prompts = self.prompt_handler.get_prompt(
            step="locate_captions",
            variables={
                "expected_figure_count": len(zip_structure.figures),
                "expected_figure_labels": [
                    figure.figure_label for figure in zip_structure.figures
                ],
                "manuscript_text": doc_content,
            },
        )

        # Prepare messages
        messages = [
            {"role": "system", "content": prompts["system"]},
            {"role": "user", "content": prompts["user"]},
        ]
        config_ = self.config["pipeline"]["locate_captions"]["openai"]
        model_ = config_.get("model", "gpt-4o")
        response = self.client.beta.chat.completions.parse(
            model=model_,
            messages=messages,
            response_format=LocateFigureCaptions,
            temperature=config_.get("temperature", 0.1),
            top_p=config_.get("top_p", 1.0),
            frequency_penalty=config_.get("frequency_penalty", 0),
            presence_penalty=config_.get("presence_penalty", 0),
        )

        # Updating the token usage
        update_token_usage(zip_structure.cost.locate_captions, response, model_)

        # Update the `zip_structure` object wit the answer
        zip_structure.ai_response_locate_captions = response.choices[0].message.content

        return (
            json.loads(response.choices[0].message.content)["figure_legends"],
            zip_structure,
        )

    def extract_individual_captions(
        self,
        doc_content: str,
        zip_structure: ZipStructure,
    ) -> dict:
        """Call OpenAI API to extract individual captions."""

        caption_section, zip_structure = self.locate_captions(
            doc_content, zip_structure
        )
        # Get prompts with variables substituted
        prompts = self.prompt_handler.get_prompt(
            step="extract_individual_captions",
            variables={
                "expected_figure_count": len(zip_structure.figures),
                "expected_figure_labels": [
                    figure.figure_label for figure in zip_structure.figures
                ],
                "figure_captions": caption_section,
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
            response_format=ExtractIndividualCaptions,
            temperature=config_.get("temperature", 0.1),
            top_p=config_.get("top_p", 1.0),
            frequency_penalty=config_.get("frequency_penalty", 0),
            presence_penalty=config_.get("presence_penalty", 0),
        )

        # Updating the token usage
        update_token_usage(
            zip_structure.cost.extract_individual_captions, response, model_
        )

        # Updating the figure captions
        self._update_figures_with_captions(
            zip_structure, json.loads(response.choices[0].message.content)["figures"]
        )
        zip_structure.ai_response_extract_individual_captions = response.choices[
            0
        ].message.content

        return zip_structure
