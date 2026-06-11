"""Anthropic Claude implementation for matching panel captions with panel images."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import anthropic

from ..ai_observability import summarize_text
from ..anthropic_utils import call_anthropic, validate_anthropic_model
from ..cost_tracking import update_token_usage
from ..manuscript_structure.manuscript_structure import Panel
from .match_caption_panel_base import MatchPanelCaption, PanelObject

logger = logging.getLogger(__name__)


class MatchPanelCaptionAnthropic(MatchPanelCaption):
    """Match panel captions with panel images using Anthropic Claude vision models."""

    def __init__(self, config: Dict[str, Any], prompt_handler: Any, extract_dir: Path):
        super().__init__(config, prompt_handler, extract_dir)
        self.client = anthropic.Anthropic()
        self.anthropic_config = config["pipeline"]["match_caption_panel"]["anthropic"]

    def _validate_config(self) -> None:
        """Validate Anthropic configuration parameters."""
        config_ = self.config["pipeline"]["match_caption_panel"]["anthropic"]
        validate_anthropic_model(config_.get("model", "claude-sonnet-4-6"))

    def _match_panel_caption(
        self,
        encoded_image: str,
        figure_caption: str,
        allowed_panels: Optional[List[Panel]] = None,
    ) -> PanelObject:
        """Pick the caption-derived panel label for this crop; captions are fixed upstream."""
        if not encoded_image:
            logger.error("Encoded image is empty, skipping API call")
            return PanelObject(panel_label="", panel_caption="")

        catalog = []
        if allowed_panels:
            catalog = [
                {
                    "panel_label": p.panel_label,
                    "panel_caption": p.panel_caption,
                }
                for p in allowed_panels
            ]
        variables = {
            "figure_caption": figure_caption,
            "allowed_panel_labels": ", ".join(
                p.panel_label for p in (allowed_panels or [])
            ),
            "allowed_panel_catalog_json": json.dumps(catalog, ensure_ascii=False),
        }
        prompts = self.prompt_handler.get_prompt("match_caption_panel", variables)

        model = self.anthropic_config.get("model", "claude-sonnet-4-6")
        logger.info(
            "Preparing panel-caption vision request",
            extra={
                "operation": "main.match_caption_panel",
                "provider": "anthropic",
                "model": model,
                "figure_caption_summary": summarize_text(figure_caption),
                "encoded_image_chars": len(encoded_image),
            },
        )

        response = call_anthropic(
            client=self.client,
            model=model,
            messages=[
                {"role": "system", "content": prompts["system"]},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompts["user"]},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}"
                            },
                        },
                    ],
                },
            ],
            response_format=PanelObject,
            temperature=self.anthropic_config.get("temperature", 0.1),
            max_tokens=self.anthropic_config.get("max_tokens", 512),
            operation="main.match_caption_panel",
            request_metadata={
                "provider": "anthropic",
                "encoded_image_chars": len(encoded_image),
            },
        )

        if hasattr(self, "zip_structure"):
            update_token_usage(
                self.zip_structure.cost.match_caption_panel,
                response,
                model,
            )

        if response.choices[0].message.parsed is not None:
            return response.choices[0].message.parsed
        return PanelObject(panel_label="", panel_caption="")
