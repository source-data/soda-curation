import os
from anthropic import Anthropic
from typing import Dict, Any, List, Tuple
from .match_caption_panel_base import MatchPanelCaption
from ..zip_structure.zip_structure_base import ZipStructure, Panel, Figure
from .match_caption_panel_prompts import SYSTEM_PROMPT, get_match_panel_caption_prompt
from PIL import Image
import io
import base64
import logging

logger = logging.getLogger(__name__)

class MatchPanelCaptionAnthropic(MatchPanelCaption):
    def __init__(self, config: Dict[str, Any], extract_dir: str):
        self.config = config
        self.client = Anthropic(api_key=self.config['api_key'])
        self.extract_dir = extract_dir

    def match_captions(self, zip_structure: ZipStructure) -> ZipStructure:
        matched_captions = {}
        for figure in zip_structure.figures:
            matched_panels = self._process_figure(figure)
            matched_captions[figure.figure_label] = matched_panels
        return self._update_zip_structure(zip_structure, matched_captions)

    def _process_figure(self, figure: Figure) -> List[Panel]:
        matched_panels = []
        if figure.img_files:
            img_path = os.path.join(self.extract_dir, figure.img_files[0])
            if os.path.exists(img_path):
                for panel in figure.panels:
                    encoded_image = self._extract_panel_image(img_path, panel['panel_bbox'])
                    if encoded_image:
                        response = self._call_anthropic_api(encoded_image, figure.figure_caption)
                        panel_label, panel_caption = self._parse_response(response)
                        matched_panels.append(Panel(
                            panel_label=panel_label,
                            panel_caption=panel_caption,
                            panel_bbox=panel['panel_bbox']
                        ))
                    else:
                        logger.warning(f"Failed to extract panel image for {figure.figure_label}")
            else:
                logger.warning(f"Image file not found: {img_path}")
        else:
            logger.warning(f"No image files found for {figure.figure_label}")
        return matched_panels

    def _extract_panel_image(self, img_path: str, bbox: List[float]) -> str:
        try:
            with Image.open(img_path) as img:
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
            return ""

    def _call_anthropic_api(self, encoded_image: str, figure_caption: str) -> str:
        try:
            response = self.client.messages.create(
                model=self.config['model'],
                max_tokens=self.config['max_tokens_to_sample'],
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "text", "text": get_match_panel_caption_prompt(figure_caption)},
                        {"type": "image", "image_url": f"data:image/png;base64,{encoded_image}"}
                    ]}
                ]
            )
            return response.content
        except Exception as e:
            logger.error(f"Error in Anthropic API call: {str(e)}")
            return ""

    def _parse_response(self, response: str) -> Tuple[str, str]:
        try:
            # Assuming the response is in the format: ```PANEL_{label}: {caption}```
            parts = response.strip('`').split(': ', 1)
            if len(parts) == 2:
                label = parts[0].split('_')[1]
                caption = parts[1]
                return label, caption
            else:
                logger.error("Unexpected response format")
                return '', ''
        except Exception as e:
            logger.error(f"Error parsing response: {str(e)}")
            return '', ''
