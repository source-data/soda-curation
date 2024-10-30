"""
This module provides functionality for matching panel captions with their corresponding images
using the OpenAI API (GPT model with vision capabilities).

It includes a class that interacts with the OpenAI API to process figure panels and their captions,
matching them based on the visual content and the full figure caption.
"""

import base64
import io
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import openai
from PIL import Image

from ..manuscript_structure.manuscript_structure import Figure, Panel, ZipStructure
from ..object_detection.object_detection import convert_to_pil_image
from .match_caption_panel_base import MatchPanelCaption
from .match_caption_panel_prompts import SYSTEM_PROMPT, get_match_panel_caption_prompt

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
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the MatchPanelCaptionOpenAI instance.

        Args:
            config (Dict[str, Any]): Configuration dictionary for the caption matching process.

        Raises:
            ValueError: If required configuration parameters are missing.
        """
        self.config = config
        self.openai_config = config.get("openai", {})

        if not self.openai_config:
            raise ValueError(
                "OpenAI configuration is missing from the main configuration"
            )

        api_key = self.openai_config.get("api_key")
        if not api_key:
            raise ValueError("API key is missing from the OpenAI configuration")

        self.client = openai.OpenAI(api_key=api_key)

        self.debug_enabled = config.get("debug", {}).get("enabled", False)
        self.debug_dir = config.get("debug", {}).get("debug_dir")
        self.extract_dir = config.get("extract_dir")
        self.process_first_figure_only = config.get("debug", {}).get(
            "process_first_figure_only", False
        )

        if not self.extract_dir:
            raise ValueError("extract_dir is not set in the configuration")

        logger.info("MatchPanelCaptionOpenAI initialized successfully")
        logger.debug(f"Debug enabled: {self.debug_enabled}")
        logger.debug(f"Debug directory: {self.debug_dir}")
        logger.debug(f"Extract directory: {self.extract_dir}")
        logger.info(f"Process first figure only: {self.process_first_figure_only}")

    def match_captions(self, figure: Figure) -> Figure:
        """Match panel captions with their corresponding images within a figure.
        
        Args:
            figure (Figure): The figure object containing panels and caption
            
        Returns:
            Figure: Updated figure with matched panel captions
        """
        logger.info(f"Matching captions for figure {figure.figure_label}")
        
        # Skip if no valid caption
        if not figure.figure_caption or figure.figure_caption == "Figure caption not found.":
            logger.warning(f"Skipping {figure.figure_label} - No valid caption")
            return figure
            
        # Skip if possible hallucination
        if figure.possible_hallucination:
            logger.warning(f"Skipping {figure.figure_label} - Caption may be hallucinated")
            return figure
            
        matched_panels = []
        
        # Process each panel
        for i, panel in enumerate(figure.panels):
            try:
                pil_image = None
                # Get the panel image
                if hasattr(figure, '_pil_image'):
                    pil_image = figure._pil_image
                else:
                    figure_path = os.path.join(self.extract_dir, figure.img_files[0])
                    pil_image, _ = convert_to_pil_image(figure_path)
                    
                if not pil_image:
                    logger.error(f"Could not load image for {figure.figure_label}")
                    continue
                    
                # Extract panel image using bbox
                encoded_panel = self._extract_panel_image(pil_image, panel.panel_bbox)
                if not encoded_panel:
                    logger.warning(f"Failed to extract panel {i} from {figure.figure_label}")
                    continue
                    
                # Get AI response for panel caption matching
                ai_response = self._call_openai_api(encoded_panel, figure.figure_caption)
                if not ai_response:
                    logger.warning(f"No AI response for panel {i} of {figure.figure_label}")
                    continue
                    
                # Parse the response to get panel label and caption
                panel_label, panel_caption = self._parse_response(ai_response)
                
                # Create new panel with matched caption
                matched_panel = Panel(
                    panel_label=panel_label if panel_label else chr(65 + i),  # Fallback to A, B, C...
                    panel_caption=panel_caption,
                    panel_bbox=panel.panel_bbox,
                    confidence=panel.confidence,
                    ai_response=ai_response,
                    sd_files=panel.sd_files
                )
                matched_panels.append(matched_panel)
                
                logger.info(f"Successfully matched panel {i} ({panel_label}) of {figure.figure_label}")
                
            except Exception as e:
                logger.error(f"Error processing panel {i} of {figure.figure_label}: {str(e)}")
                # Keep original panel if processing fails
                matched_panels.append(panel)
                
        figure.panels = matched_panels
        return figure

    def _process_figure(self, figure: Figure) -> List[Panel]:
        """
        Process a single figure, matching panel captions with their corresponding images.

        This method extracts panel images, calls the OpenAI API for caption matching,
        and compiles the results for each panel in the figure.

        Args:
            figure (Figure): The figure object to process.

        Returns:
            List[Panel]: A list of Panel objects containing matched panel information.
        """
        matched_panels = []

        for i, panel in enumerate(figure.panels):
            logger.info(f"Processing panel {i+1} of figure {figure.figure_label}")

            try:
                if hasattr(figure, "_pil_image"):
                    pil_image = figure._pil_image
                else:
                    figure_path = os.path.join(self.extract_dir, figure.img_files[0])
                    pil_image, _ = convert_to_pil_image(figure_path)

                encoded_image = self._extract_panel_image(pil_image, panel.panel_bbox)

                if not encoded_image:
                    logger.warning(
                        f"Failed to extract panel image for panel {i+1} of figure {figure.figure_label}"
                    )
                    continue

                if self.debug_enabled and self.debug_dir:
                    self._save_debug_image(
                        encoded_image, f"{figure.figure_label}_panel_{i+1}.png"
                    )

                response = self._call_openai_api(encoded_image, figure.figure_caption)
                panel_label, panel_caption = self._parse_response(response)

                matched_panel = Panel(
                    panel_label=panel_label,
                    panel_caption=panel_caption,
                    panel_bbox=panel.panel_bbox,
                    confidence=panel.confidence,
                    ai_response=response,
                )
                matched_panels.append(matched_panel)
                logger.info(
                    f"Matched panel: Label={panel_label}, Caption={panel_caption[:50]}..."
                )

            except Exception as e:
                logger.error(
                    f"Error processing panel {i+1} of figure {figure.figure_label}: {str(e)}"
                )

        return matched_panels

    def _extract_panel_image(
        self, pil_image: Image.Image, bbox: List[float]
    ) -> Optional[str]:
        """
        Extract a panel image from a figure based on bounding box coordinates.

        This method crops the PIL Image according to the bounding box,
        and returns the panel image as a base64 encoded string.

        Args:
            pil_image (Image.Image): The PIL Image object of the entire figure.
            bbox (List[float]): Bounding box coordinates [x1, y1, x2, y2] in relative format.

        Returns:
            Optional[str]: Base64 encoded string of the panel image, or None if extraction fails.
        """
        try:
            width, height = pil_image.size
            left, top, right, bottom = [
                int(coord * width if i % 2 == 0 else coord * height)
                for i, coord in enumerate(bbox)
            ]
            panel = pil_image.crop((left, top, right, bottom))

            buffered = io.BytesIO()
            panel.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
        except Exception as e:
            logger.error(f"Error extracting panel image: {str(e)}")
            return None

    def _call_openai_api(self, encoded_image: str, figure_caption: str) -> str:
        """
        Call the OpenAI API to match a panel image with its caption.

        This method sends the encoded panel image and figure caption to the OpenAI API
        for caption matching using a vision-capable GPT model.

        Args:
            encoded_image (str): Base64 encoded string of the panel image.
            figure_caption (str): The full caption of the figure.

        Returns:
            str: The AI's response containing the matched panel caption.
        """
        if not encoded_image:
            logger.error("Encoded image is empty, skipping API call")
            return ""

        try:
            prompt = get_match_panel_caption_prompt(figure_caption)
            
            response = self.client.chat.completions.create(
                model=self.openai_config.get("model", "gpt-4-vision-preview"),
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user", 
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{encoded_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=2048,  # Increased token limit for longer captions
                temperature=0.3    # Lower temperature for more consistent results
            )
            
            if response.choices:
                return response.choices[0].message.content
                
            logger.warning("Empty response from OpenAI API")
            return ""
            
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            return ""

    def _parse_response(self, response: str) -> Tuple[str, str]:
        """
        Parse the AI's response to extract the panel label and caption.

        This method interprets the AI's response, extracting the panel label and
        its corresponding caption.

        Args:
            response (str): The AI's response string.

        Returns:
            Tuple[str, str]: A tuple containing the panel label and its caption.
        """
        try:
            # Look for PANEL_X: format
            match = re.search(r'PANEL_([A-Z])\s*:\s*(.*)', response, re.DOTALL)
            if match:
                label = match.group(1)
                caption = match.group(2).strip()
                return label, caption
                
            # Fallback - look for any letter followed by colon
            match = re.search(r'\(?([A-Z])\)?\s*:\s*(.*)', response, re.DOTALL)
            if match:
                return match.group(1), match.group(2).strip()
                
            logger.warning(f"Could not parse panel label/caption from response: {response[:100]}...")
            return "", ""
            
        except Exception as e:
            logger.error(f"Error parsing AI response: {str(e)}")
            return "", ""


    def _save_debug_image(self, encoded_image: str, filename: str):
        """
        Save the encoded image to the debug directory.

        This method decodes the base64 encoded image and saves it to the debug directory
        for inspection and troubleshooting purposes.

        Args:
            encoded_image (str): Base64 encoded string of the image.
            filename (str): The filename to use when saving the debug image.
        """
        if self.debug_enabled and self.debug_dir:
            try:
                image_data = base64.b64decode(encoded_image)
                image = Image.open(io.BytesIO(image_data))
                debug_image_path = os.path.join(self.debug_dir, filename)
                os.makedirs(os.path.dirname(debug_image_path), exist_ok=True)
                image.save(debug_image_path)
                logger.debug(f"Saved debug image: {debug_image_path}")
            except Exception as e:
                logger.error(f"Error saving debug image: {str(e)}")
