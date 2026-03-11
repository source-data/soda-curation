"""Anthropic Claude implementation for matching panel captions with panel images."""

import logging
from pathlib import Path
from typing import Any, Dict

import anthropic

from ..anthropic_utils import call_anthropic, validate_anthropic_model
from ..cost_tracking import update_token_usage
from .match_caption_panel_base import MatchPanelCaption, PanelObject

logger = logging.getLogger(__name__)


class MatchPanelCaptionAnthropic(MatchPanelCaption):
    """Match panel captions with panel images using Anthropic Claude vision models."""

    def __init__(self, config: Dict[str, Any], prompt_handler: Any, extract_dir: Path):
        super().__init__(config, prompt_handler, extract_dir)
        self.client = anthropic.Anthropic()
        self.anthropic_config = config["pipeline"]["match_caption_panel"]["anthropic"]
        self.figure_images: Dict = {}

    def _validate_config(self) -> None:
        """Validate Anthropic configuration parameters."""
        config_ = self.config["pipeline"]["match_caption_panel"]["anthropic"]
        validate_anthropic_model(config_.get("model", "claude-sonnet-4-6"))

    def process_figures(self, zip_structure):
        """Override parent to cache figure images."""
        from .object_detection import convert_to_pil_image

        self.figure_images = {}
        result = super().process_figures(zip_structure)

        for figure in zip_structure.figures:
            if figure.figure_label not in self.figure_images and figure.img_files:
                try:
                    full_path = self.extract_dir / figure.img_files[0]
                    if full_path.exists():
                        image, _ = convert_to_pil_image(str(full_path))
                        self.figure_images[figure.figure_label] = image
                except Exception as e:
                    logger.error(
                        f"Error caching figure image {figure.figure_label}: {str(e)}"
                    )

        return result

    def get_figure_images_and_captions(self):
        """Return base64-encoded figure images and their captions."""
        import base64
        import io

        result = []
        if not hasattr(self, "zip_structure") or not self.zip_structure:
            logger.warning("No zip structure available. Run process_figures first.")
            return result

        for figure in self.zip_structure.figures:
            try:
                if figure.figure_label in self.figure_images:
                    image = self.figure_images[figure.figure_label]
                    buffered = io.BytesIO()
                    image.save(buffered, format="PNG")
                    encoded_image = base64.b64encode(buffered.getvalue()).decode(
                        "utf-8"
                    )
                    result.append(
                        (figure.figure_label, encoded_image, figure.figure_caption)
                    )
                else:
                    logger.warning(
                        f"Figure image not found in cache: {figure.figure_label}"
                    )
            except Exception as e:
                logger.error(f"Error encoding figure {figure.figure_label}: {str(e)}")

        return result

    def _match_panel_caption(
        self, encoded_image: str, figure_caption: str
    ) -> PanelObject:
        """Match a panel image with its caption using Claude vision."""
        if not encoded_image:
            logger.error("Encoded image is empty, skipping API call")
            return PanelObject(panel_label="", panel_caption="")

        prompts = self.prompt_handler.get_prompt(
            "match_caption_panel", {"figure_caption": figure_caption}
        )

        model = self.anthropic_config.get("model", "claude-sonnet-4-6")

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
                                "url": f"data:image/png;base64,{encoded_image}"
                            },
                        },
                    ],
                },
            ],
            response_format=PanelObject,
            temperature=self.anthropic_config.get("temperature", 0.1),
            max_tokens=self.anthropic_config.get("max_tokens", 512),
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
