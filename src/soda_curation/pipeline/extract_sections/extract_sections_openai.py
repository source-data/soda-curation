"""OpenAI implementation for extracting manuscript sections."""

import json
import logging
import os
from typing import Dict, Tuple

import openai

from ..cost_tracking import update_token_usage
from ..manuscript_structure.manuscript_structure import ZipStructure
from .extract_sections_base import ExtractedSections, SectionExtractor

logger = logging.getLogger(__name__)


class SectionExtractorOpenAI(SectionExtractor):
    """Implementation of section extraction using OpenAI's GPT models."""

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
        config_ = self.config["pipeline"]["extract_sections"]["openai"]
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

    def extract_sections(
        self,
        doc_content: str,
        zip_structure: ZipStructure,
    ) -> Tuple[str, str, ZipStructure]:
        """Extract figure legends and data availability sections."""
        try:
            # Get prompts with variables substituted
            prompts = self.prompt_handler.get_prompt(
                step="extract_sections",
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

            config_ = self.config["pipeline"]["extract_sections"]["openai"]
            model_ = config_.get("model", "gpt-4o")

            response = self.client.beta.chat.completions.parse(
                model=model_,
                messages=messages,
                response_format=ExtractedSections,
                temperature=config_.get("temperature", 0.1),
                top_p=config_.get("top_p", 1.0),
                frequency_penalty=config_.get("frequency_penalty", 0),
                presence_penalty=config_.get("presence_penalty", 0),
            )

            # Update token usage - we'll split the cost between the two tasks
            usage_cost = response.usage.total_tokens  # Split cost evenly
            update_token_usage(
                zip_structure.cost.extract_sections,
                {
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": usage_cost,
                    }
                },
                model_,
            )

            # Store response in both relevant places for backward compatibility
            response_content = response.choices[0].message.content

            # Parse and return the sections
            result = json.loads(response_content)
            zip_structure.ai_response_locate_captions = result["figure_legends"]
            return (
                result["figure_legends"],
                result["data_availability"],
                zip_structure,
            )

        except Exception as e:
            logger.error(f"Error extracting sections: {str(e)}")
            return "", "", zip_structure
