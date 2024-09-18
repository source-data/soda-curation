import openai
from typing import Dict, Any, List, Tuple
from .match_caption_panel_base import MatchPanelCaption
from ..zip_structure.zip_structure_base import ZipStructure, Panel, Figure
from .match_caption_panel_prompts import SYSTEM_PROMPT, get_match_panel_caption_prompt
import json
import logging

logger = logging.getLogger(__name__)

class MatchPanelCaptionOpenAI(MatchPanelCaption):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = openai.OpenAI(api_key=self.config['api_key'])

    def match_captions(self, zip_structure: ZipStructure) -> ZipStructure:
        matched_captions = {}
        for figure in zip_structure.figures:
            matched_panels = self._process_figure(figure)
            matched_captions[figure.figure_label] = matched_panels
        return self._update_zip_structure(zip_structure, matched_captions)

    def _process_figure(self, figure: Figure) -> List[Panel]:
        matched_panels = []
        for panel in figure.panels:
            encoded_image = self._extract_panel_image(figure.img_files[0], panel['panel_bbox'])
            response = self._call_openai_api(encoded_image, figure.figure_caption)
            panel_label, panel_caption = self._parse_response(response)
            matched_panels.append(Panel(
                panel_label=panel_label,
                panel_caption=panel_caption,
                panel_bbox=panel['panel_bbox']
            ))
        return matched_panels

    def _call_openai_api(self, encoded_image: str, figure_caption: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.config['model'],
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "text", "text": get_match_panel_caption_prompt(figure_caption)},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}}
                    ]}
                ],
                max_tokens=300
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in OpenAI API call: {str(e)}")
            return "{}"

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
