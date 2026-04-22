"""Anthropic Claude implementation for extracting manuscript sections."""

import json
import logging
from typing import Dict, Tuple

import anthropic

from ..ai_observability import summarize_text
from ..anthropic_utils import call_anthropic, validate_anthropic_model
from ..cost_tracking import update_token_usage
from ..manuscript_structure.manuscript_structure import ZipStructure
from .extract_sections_base import ExtractedSections, SectionExtractor

logger = logging.getLogger(__name__)


class SectionExtractorAnthropic(SectionExtractor):
    """Extract figure legends and data availability sections using Claude."""

    def __init__(self, config: Dict, prompt_handler):
        super().__init__(config, prompt_handler)
        self.client = anthropic.Anthropic()

    def _validate_config(self) -> None:
        """Validate Anthropic configuration parameters."""
        config_ = self.config["pipeline"]["extract_sections"]["anthropic"]
        validate_anthropic_model(config_.get("model", "claude-sonnet-4-6"))

    def extract_sections(
        self,
        doc_content: str,
        zip_structure: ZipStructure,
    ) -> Tuple[str, str, ZipStructure]:
        """Extract figure legends and data availability sections."""
        logger.info(
            "Preparing section extraction request",
            extra={
                "operation": "main.extract_sections",
                "provider": "anthropic",
                "figure_count": len(zip_structure.figures),
                "manuscript_summary": summarize_text(doc_content),
            },
        )
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

        messages = [
            {"role": "system", "content": prompts["system"]},
            {"role": "user", "content": prompts["user"]},
        ]

        config_ = self.config["pipeline"]["extract_sections"]["anthropic"]
        model_ = config_.get("model", "claude-sonnet-4-6")

        response = call_anthropic(
            client=self.client,
            model=model_,
            messages=messages,
            response_format=ExtractedSections,
            temperature=config_.get("temperature", 0.1),
            max_tokens=config_.get("max_tokens", 2048),
            operation="main.extract_sections",
            request_metadata={"figure_count": len(zip_structure.figures)},
        )

        update_token_usage(
            zip_structure.cost.extract_sections,
            response,
            model_,
        )

        if response.choices[0].message.parsed is not None:
            result = response.choices[0].message.parsed
            figure_legends = result.figure_legends
            data_availability = result.data_availability
        else:
            # Fallback: parse JSON from raw text
            response_content = response.choices[0].message.content
            result = json.loads(response_content)
            figure_legends = result["figure_legends"]
            data_availability = result["data_availability"]

        logger.info(
            "Section extraction completed",
            extra={
                "operation": "main.extract_sections",
                "provider": "anthropic",
                "figure_legends_summary": summarize_text(figure_legends),
                "data_availability_summary": summarize_text(data_availability),
            },
        )

        zip_structure.ai_response_locate_captions = figure_legends
        return figure_legends, data_availability, zip_structure
