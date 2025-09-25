"""
This module provides functionality for matching panel captions with their corresponding images
using the OpenAI API (GPT model with vision capabilities).

It includes a class that interacts with the OpenAI API to process figure panels and their captions,
matching them based on the visual content and the full figure caption.
"""

import base64
import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import openai

from ..cost_tracking import update_token_usage
from ..manuscript_structure.manuscript_structure import ZipStructure
from ..openai_utils import call_openai_with_fallback, validate_model_config
from .match_caption_panel_base import MatchPanelCaption, PanelObject
from .object_detection import convert_to_pil_image  # Import the function directly

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

        # Get OpenAI specific config
        self.openai_config = config["pipeline"]["match_caption_panel"]["openai"]

        # Cache for figure images
        self.figure_images = {}

    def _validate_config(self) -> None:
        """Validate OpenAI configuration parameters."""
        valid_models = [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4o-2024-08-06",
            "gpt-4o-mini-2024-07-18",
            "gpt-5",
        ]
        config_ = self.config["pipeline"]["match_caption_panel"]["openai"]
        model = config_.get("model", "gpt-4o")
        if model not in valid_models:
            raise ValueError(f"Invalid model: {model}. Must be one of {valid_models}")

        # Use the utility function for validation
        validate_model_config(model, config_)

    def process_figures(self, zip_structure: ZipStructure) -> ZipStructure:
        """Override parent method to cache figure images"""
        self.figure_images = {}  # Clear previous cache
        result = super().process_figures(zip_structure)

        # For each figure, store the image in the cache if not already there
        for figure in zip_structure.figures:
            if figure.figure_label not in self.figure_images and figure.img_files:
                try:
                    full_path = self.extract_dir / figure.img_files[0]
                    if full_path.exists():
                        # Use the imported function, not a method
                        image, _ = convert_to_pil_image(str(full_path))
                        self.figure_images[figure.figure_label] = image
                except Exception as e:
                    logger.error(
                        f"Error caching figure image {figure.figure_label}: {str(e)}"
                    )

        return result

    def get_figure_images_and_captions(self) -> List[Tuple[str, str, str]]:
        """
        Get base64-encoded figure images and their captions.

        Returns:
            List of tuples containing (figure_label, base64_encoded_image, figure_caption)
        """
        result = []

        if not hasattr(self, "zip_structure") or not self.zip_structure:
            logger.warning("No zip structure available. Run process_figures first.")
            return result

        for figure in self.zip_structure.figures:
            try:
                if figure.figure_label in self.figure_images:
                    # Get the PIL image from cache
                    image = self.figure_images[figure.figure_label]

                    # Convert to base64
                    buffered = io.BytesIO()
                    image.save(buffered, format="PNG")
                    encoded_image = base64.b64encode(buffered.getvalue()).decode(
                        "utf-8"
                    )

                    # Add to result list
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
        """Match panel with caption using OpenAI's vision model."""
        if not encoded_image:
            logger.error("Encoded image is empty, skipping API call")
            return PanelObject(panel_label="", panel_caption="")

        try:
            prompts = self.prompt_handler.get_prompt(
                "match_caption_panel", {"figure_caption": figure_caption}
            )

            response = call_openai_with_fallback(
                client=self.client,
                model=self.openai_config.get("model", "gpt-4o"),
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
                temperature=self.openai_config.get("temperature", 0.1),
                top_p=self.openai_config.get("top_p", 1.0),
                frequency_penalty=self.openai_config.get("frequency_penalty", 0),
                presence_penalty=self.openai_config.get("presence_penalty", 0),
                max_tokens=self.openai_config.get("max_tokens", 512),
            )
            # Track token usage
            if hasattr(self, "zip_structure"):
                update_token_usage(
                    self.zip_structure.cost.match_caption_panel,
                    response,
                    self.openai_config["model"],
                )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error in panel caption matching: {str(e)}")
            return PanelObject(panel_label="", panel_caption="")
