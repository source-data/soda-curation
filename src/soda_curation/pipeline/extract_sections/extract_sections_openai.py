"""OpenAI implementation for extracting manuscript sections."""

import json
import logging
import os
from typing import Dict, Tuple

import openai

from ..cost_tracking import update_token_usage
from ..manuscript_structure.manuscript_structure import ZipStructure
from ..openai_utils import call_openai_with_fallback, validate_model_config
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
        valid_models = ["gpt-4o", "gpt-4o-mini", "gpt-5"]
        config_ = self.config["pipeline"]["extract_sections"]["openai"]
        model = config_.get("model", "gpt-4o")
        if model not in valid_models:
            raise ValueError(f"Invalid model: {model}. Must be one of {valid_models}")

        # Use the utility function for validation
        validate_model_config(model, config_)

    def extract_sections(
        self,
        doc_content: str,
        zip_structure: ZipStructure,
    ) -> Tuple[str, str, ZipStructure]:
        """Extract figure legends and data availability sections."""
        # try:
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

        response = call_openai_with_fallback(
            client=self.client,
            model=model_,
            messages=messages,
            response_format=ExtractedSections,
            temperature=config_.get("temperature", 0.1),
            top_p=config_.get("top_p", 1.0),
            frequency_penalty=config_.get("frequency_penalty", 0),
            presence_penalty=config_.get("presence_penalty", 0),
        )

        # Update token usage
        update_token_usage(
            zip_structure.cost.extract_sections,
            response,
            model_,
        )

        # Store response in both relevant places for backward compatibility
        # When using structured responses, the parsed content is in .parsed
        logger.info(f"Response type: {type(response)}")
        logger.info(f"Response choices type: {type(response.choices)}")
        logger.info(f"Response choices length: {len(response.choices)}")
        logger.info(f"Response message type: {type(response.choices[0].message)}")
        logger.info(
            f"Response message has parsed: {hasattr(response.choices[0].message, 'parsed')}"
        )

        if hasattr(response.choices[0].message, "parsed"):
            result = response.choices[0].message.parsed
            logger.info(f"Parsed result type: {type(result)}")
            # result is a Pydantic model object, access attributes directly
            figure_legends = result.figure_legends
            data_availability = result.data_availability
        else:
            # Fallback for non-structured responses
            response_content = response.choices[0].message.content
            logger.info(f"Response content type: {type(response_content)}")
            result = json.loads(response_content)
            # result is a dictionary, access with keys
            figure_legends = result["figure_legends"]
            data_availability = result["data_availability"]

        zip_structure.ai_response_locate_captions = figure_legends
        return (
            figure_legends,
            data_availability,
            zip_structure,
        )

        # except Exception as e:
        #     logger.error(f"Error extracting sections: {str(e)}")
        #     return "", "", zip_structure
