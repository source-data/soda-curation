"""
This module provides functionality for matching panel captions with their corresponding images
using the OpenAI API (GPT model with vision capabilities).

It includes a class that interacts with the OpenAI API to process figure panels and their captions,
matching them based on the visual content and the full figure caption.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import openai

from ..ai_observability import summarize_text
from ..cost_tracking import update_token_usage
from ..manuscript_structure.manuscript_structure import Panel, ZipStructure
from ..openai_utils import call_openai
from ..step_config import resolve_step_config
from .match_caption_panel_base import MatchPanelCaption, PanelObject

logger = logging.getLogger(__name__)


class MatchPanelCaptionOpenAI(MatchPanelCaption):
    """
    A class to match panel captions with their corresponding images using OpenAI's GPT model with vision capabilities.

    This class provides methods to interact with the OpenAI API, process figure panels,
    and match them with their respective captions based on visual content and the full figure caption.

    Attributes:
        config (Dict[str, Any]): Configuration dictionary for the caption matching process.
        openai_config (Dict[str, Any]): Configuration specific to the OpenAI API.
        client (openai.OpenAI): OpenAI API client.
        debug_enabled (bool): Flag indicating whether debug mode is enabled.
        debug_dir (str): Directory for saving debug information.
        extract_dir (str): Directory containing extracted files from the ZIP archive.
        figure_images (Dict): Cache of loaded figure images
    """

    def __init__(self, config: Dict[str, Any], prompt_handler: Any, extract_dir: Path):
        super().__init__(config, prompt_handler, extract_dir)

        # Initialize OpenAI client
        self.client = openai.OpenAI()

    def _validate_config(self) -> None:
        resolved = resolve_step_config(self.config["pipeline"]["match_caption_panel"])
        if resolved["provider"] != "openai":
            raise ValueError(
                f"match_caption_panel model '{resolved['model']}' requires Anthropic."
            )

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
        model = resolve_step_config(self.config["pipeline"]["match_caption_panel"])[
            "model"
        ]
        logger.info(
            "Preparing panel-caption vision request",
            extra={
                "operation": "main.match_caption_panel",
                "provider": "openai",
                "model": model,
                "figure_caption_summary": summarize_text(figure_caption),
                "encoded_image_chars": len(encoded_image),
            },
        )

        response = call_openai(
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
            operation="main.match_caption_panel",
            request_metadata={
                "provider": "openai",
                "encoded_image_chars": len(encoded_image),
            },
        )
        # Track token usage
        if hasattr(self, "zip_structure"):
            update_token_usage(
                self.zip_structure.cost.match_caption_panel,
                response,
                model,
            )

        # When using structured responses, the parsed content is in .parsed
        if hasattr(response.choices[0].message, "parsed"):
            return response.choices[0].message.parsed
        else:
            # Fallback for non-structured responses
            return response.choices[0].message.content
