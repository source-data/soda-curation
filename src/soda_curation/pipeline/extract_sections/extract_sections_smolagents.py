"""Smolagents implementation for extracting manuscript sections with verification."""

import logging
import os
from typing import Annotated, Dict, Tuple

from pydantic import BaseModel, Field
from smolagents import OpenAIServerModel, ToolAgent
from smolagents.tools import Tool

from ..cost_tracking import update_token_usage
from ..manuscript_structure.manuscript_structure import ZipStructure
from .extract_sections_base import ExtractedSections, SectionExtractor

logger = logging.getLogger(__name__)


class VerificationResult(BaseModel):
    """Model for verification result."""

    is_verbatim: bool = Field(
        description="True if the extracted section is a verbatim substring of the HTML"
    )


class SectionExtractorSmolagents(SectionExtractor):
    """Implementation of section extraction using Smolagents with verification."""

    def __init__(self, config: Dict, prompt_handler):
        """Initialize with configuration and prompt handler."""
        super().__init__(config, prompt_handler)

        # Initialize OpenAI client via smolagents
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        # Define verification tool
        self.verification_tool = Tool(
            name="verify_extraction_in_html",
            description="Verifies that the extracted section is a true verbatim substring of the full HTML input.",
            function=self._verify_extraction_in_html,
            inputs={
                "full_html": str,
                "extracted_section": str,
            },
            output_type=VerificationResult,
        )

        # Setup the OpenAI model
        config_ = self.config["pipeline"]["extract_sections"]["openai"]
        model_ = config_.get("model", "gpt-4o")

        self.model = OpenAIServerModel(
            model_id=model_,
            temperature=config_.get("temperature", 0.1),
            top_p=config_.get("top_p", 1.0),
            frequency_penalty=config_.get("frequency_penalty", 0),
            presence_penalty=config_.get("presence_penalty", 0),
            api_key=api_key,
        )

        # We'll set up the ToolAgent in the extract_sections method

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
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

    def _verify_extraction_in_html(
        self,
        full_html: Annotated[str, "Full manuscript HTML"],
        extracted_section: Annotated[
            str, "The section the agent claims to have extracted verbatim"
        ],
    ) -> VerificationResult:
        """
        Verifies that the extracted section is a true verbatim substring of the full HTML input.
        """
        return VerificationResult(is_verbatim=extracted_section in full_html)

    def extract_sections(
        self,
        doc_content: str,
        zip_structure: ZipStructure,
    ) -> Tuple[str, str, ZipStructure]:
        """Extract figure legends and data availability sections with verification."""
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

        # Modify system prompt to include instructions about verification
        system_prompt = (
            prompts["system"]
            + "\n\nIMPORTANT: You MUST verify that your extracted sections are verbatim from the original text using the verification tool. Only return sections that pass verification."
        )

        # Set up the ToolAgent with our verification tool
        tool_agent = ToolAgent(
            name="Section Extractor",
            system_prompt=system_prompt,
            model=self.model,
            tools=[self.verification_tool],
            output_type=ExtractedSections,
        )

        import pdb

        pdb.set_trace()

        # Run the agent with our query
        agent_response = tool_agent.run(prompts["user"])

        # Access the output through the value property (this is the ExtractedSections object)
        extracted_sections = agent_response.value

        # Get the token usage from agent run
        usage_info = agent_response.metadata.get("usage", {})

        # Update token usage if available
        if usage_info:
            config_ = self.config["pipeline"]["extract_sections"]["openai"]
            model_ = config_.get("model", "gpt-4o")
            update_token_usage(
                zip_structure.cost.extract_sections,
                {
                    "usage": {
                        "prompt_tokens": usage_info.get("prompt_tokens", 0),
                        "completion_tokens": usage_info.get("completion_tokens", 0),
                        "total_tokens": usage_info.get("total_tokens", 0),
                    }
                },
                model_,
            )

        # Store response in the relevant place for backward compatibility
        zip_structure.ai_response_locate_captions = extracted_sections.figure_legends

        return (
            extracted_sections.figure_legends,
            extracted_sections.data_availability,
            zip_structure,
        )

        # except Exception as e:
        #     logger.error(f"Error extracting sections with smolagents: {str(e)}")
        #     return "", "", zip_structure
