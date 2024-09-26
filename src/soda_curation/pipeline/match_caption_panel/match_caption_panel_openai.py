"""
This module provides functionality for matching panel captions with their corresponding images
using the OpenAI API (GPT model with vision capabilities).

It includes a class that interacts with the OpenAI API to process figure panels and their captions,
matching them based on the visual content and the full figure caption.
"""

import openai
from typing import Dict, Any, List, Tuple, Optional
from .match_caption_panel_base import MatchPanelCaption
from ..manuscript_structure.manuscript_structure import ZipStructure, Figure
from .match_caption_panel_prompts import SYSTEM_PROMPT, get_match_panel_caption_prompt
import logging
import os
import json
import base64
from PIL import Image
import io

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
        self.openai_config = config.get('openai', {})
        
        if not self.openai_config:
            raise ValueError("OpenAI configuration is missing from the main configuration")
        
        api_key = self.openai_config.get('api_key')
        if not api_key:
            raise ValueError("API key is missing from the OpenAI configuration")
        
        self.client = openai.OpenAI(api_key=api_key)
        
        self.debug_enabled = config.get('debug', {}).get('enabled', False)
        self.debug_dir = config.get('debug', {}).get('debug_dir')
        self.extract_dir = config.get('extract_dir')
        self.process_first_figure_only = config.get('debug', {}).get('process_first_figure_only', False)
        
        if not self.extract_dir:
            raise ValueError("extract_dir is not set in the configuration")
        
        logger.info("MatchPanelCaptionOpenAI initialized successfully")
        logger.debug(f"Debug enabled: {self.debug_enabled}")
        logger.debug(f"Debug directory: {self.debug_dir}")
        logger.debug(f"Extract directory: {self.extract_dir}")
        logger.info(f"Process first figure only: {self.process_first_figure_only}")

    def match_captions(self, zip_structure: ZipStructure) -> ZipStructure:
        """
        Match panel captions with their corresponding images for all figures in the ZIP structure.

        This method processes all figures in the ZIP structure, matching panel captions
        with their corresponding images based on visual content and the full figure caption.

        Args:
            zip_structure (ZipStructure): The ZIP structure containing figures and their information.

        Returns:
            ZipStructure: Updated ZIP structure with matched panel captions for all figures.
        """
        logger.info("Starting panel caption matching process")
        
        if zip_structure.figures:
            figures_to_process = zip_structure.figures[:1] if self.process_first_figure_only else zip_structure.figures
            for i, figure in enumerate(zip_structure.figures):
                if i < len(figures_to_process):
                    logger.info(f"Processing figure: {figure.figure_label}")
                    matched_panels = self._process_figure(figure)
                    figure.panels = matched_panels
                else:
                    figure.panels = []  # Clear panels for figures not processed in debug mode
                
            if self.process_first_figure_only:
                logger.info("Processed first figure only as per debug configuration")
        else:
            logger.warning("No figures found in the ZIP structure")
        
        logger.info("Panel caption matching process completed")
        return zip_structure

    def _process_figure(self, figure: Figure) -> List[Dict[str, Any]]:
        """
        Process a single figure, matching panel captions with their corresponding images.

        This method extracts panel images, calls the OpenAI API for caption matching,
        and compiles the results for each panel in the figure.

        Args:
            figure (Figure): The figure object to process.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing matched panel information.
        """
        matched_panels = []
        figure_path = os.path.join(self.extract_dir, figure.img_files[0])
        
        for i, panel in enumerate(figure.panels):
            logger.info(f"Processing panel {i+1} of figure {figure.figure_label}")
            
            if 'panel_bbox' not in panel:
                logger.warning(f"No bounding box found for panel {i+1} of figure {figure.figure_label}")
                continue
            
            try:
                encoded_image = self._extract_panel_image(figure_path, panel['panel_bbox'])
                
                if not encoded_image:
                    logger.warning(f"Failed to extract panel image for panel {i+1} of figure {figure.figure_label}")
                    continue
                
                if self.debug_enabled and self.debug_dir:
                    self._save_debug_image(encoded_image, f"{figure.figure_label}_panel_{i+1}.png")
                
                response = self._call_openai_api(encoded_image, figure.figure_caption)
                panel_label, panel_caption = self._parse_response(response)
                
                matched_panel = panel.copy()
                matched_panel.update({
                    'panel_label': panel_label,
                    'panel_caption': panel_caption
                })
                matched_panels.append(matched_panel)
                logger.info(f"Matched panel: Label={panel_label}, Caption={panel_caption[:50]}...")
            
            except Exception as e:
                logger.error(f"Error processing panel {i+1} of figure {figure.figure_label}: {str(e)}")
        
        return matched_panels

    def _extract_panel_image(self, figure_path: str, bbox: List[float]) -> Optional[str]:
        """
        Extract a panel image from a figure based on bounding box coordinates.

        This method opens the figure image, crops it according to the bounding box,
        and returns the panel image as a base64 encoded string.

        Args:
            figure_path (str): Path to the figure image file.
            bbox (List[float]): Bounding box coordinates [x1, y1, x2, y2] in relative format.

        Returns:
            Optional[str]: Base64 encoded string of the panel image, or None if extraction fails.
        """
        try:
            with Image.open(figure_path) as img:
                # Convert to RGB if the image is in CMYK mode
                if img.mode == 'CMYK':
                    img = img.convert('RGB')
                
                width, height = img.size
                left, top, right, bottom = [
                    int(coord * width if i % 2 == 0 else coord * height)
                    for i, coord in enumerate(bbox)
                ]
                panel = img.crop((left, top, right, bottom))
                
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
            logger.info("Calling OpenAI API for panel caption matching")
            logger.info(f"Figure caption: {figure_caption[:100]}...")
            
            model = self.openai_config.get('model', 'gpt-4-vision-preview')
            logger.info(f"Using OpenAI model: {model}")
            
            prompt = get_match_panel_caption_prompt(figure_caption)
            
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}}
                    ]}
                ],
                max_tokens=1024
            )
            
            ai_response = response.choices[0].message.content
            logger.info("Received response from OpenAI API")
            logger.info(f"AI response: {ai_response}")
            
            return ai_response
        except Exception as e:
            logger.error(f"Error in OpenAI API call: {str(e)}")
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
            logger.info("Parsing AI response")
            logger.debug(f"Raw response: {response}")
            
            # Assuming the response is in the format: ```PANEL_{label}: {caption}```
            parts = response.strip('`').split(': ', 1)
            if len(parts) == 2:
                label = parts[0].split('_')[1]
                caption = parts[1]
                logger.info(f"Successfully parsed response. Label: {label}")
                logger.debug(f"Parsed caption: {caption[:50]}...")  # Log first 50 chars of parsed caption
                return label, caption
            else:
                logger.error(f"Unexpected response format: {response}")
                return '', ''
        except Exception as e:
            logger.error(f"Error parsing response: {str(e)}")
            return '', ''

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
