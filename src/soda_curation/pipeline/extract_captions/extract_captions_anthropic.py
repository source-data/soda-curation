import json
import logging
from typing import Dict, Any, Union
from anthropic import Anthropic
from docx import Document
from .extract_captions_base import FigureCaptionExtractor
from .extract_captions_prompts import get_extract_captions_prompt
from ..zip_structure.zip_structure_base import ZipStructure
import re

logger = logging.getLogger(__name__)

from ..zip_structure.zip_structure_base import CustomJSONEncoder

class CustomJSONDecoder(json.JSONDecoder):
    def decode(self, s):
        result = super().decode(s)
        return self._decode(result)

    def _decode(self, obj):
        if isinstance(obj, dict):
            return {key: self._decode(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._decode(item) for item in obj]
        elif isinstance(obj, str):
            return CustomJSONEncoder.unescape_string(obj)
        return obj

class FigureCaptionExtractorClaude(FigureCaptionExtractor):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = Anthropic(api_key=self.config['api_key'])

    def extract_captions(self, docx_path: str, zip_structure: ZipStructure) -> ZipStructure:
        try:
            logger.info(f"Processing file: {docx_path}")
            
            file_content = self._extract_docx_content(docx_path)
            if not file_content:
                logger.warning(f"No content extracted from {docx_path}")
                return self._update_zip_structure(zip_structure, {})

            prompt = get_extract_captions_prompt(file_content)
            
            logger.debug("Sending request to Anthropic API")
            response = self.client.messages.create(
                model=self.config['model'],
                max_tokens=self.config['max_tokens_to_sample'],
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            logger.debug(f"Response object type: {type(response)}")
            logger.debug(f"Response object attributes: {dir(response)}")
            
            # Extract only the content from the response
            extracted_text = response.content
            if isinstance(extracted_text, list):
                extracted_text = ' '.join(item.text for item in extracted_text if hasattr(item, 'text'))
            elif not isinstance(extracted_text, str):
                extracted_text = str(extracted_text)
            
            logger.debug(f"Extracted text type: {type(extracted_text)}")
            logger.debug(f"Full extracted text: {extracted_text}")
            
            captions = self._parse_response(extracted_text)
            if not captions:
                logger.warning("Failed to extract captions from Claude's response")
                return self._update_zip_structure(zip_structure, {})
            
            updated_structure = self._update_zip_structure(zip_structure, captions)
            logger.info(f"Updated ZIP structure: {updated_structure}")
            
            return updated_structure
        
        except Exception as e:
            logger.exception(f"Error in caption extraction: {str(e)}")
            return self._update_zip_structure(zip_structure, {})

    def _extract_docx_content(self, file_path: str) -> str:
        try:
            doc = Document(file_path)
            paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
            return "\n\n".join(paragraphs)
        except Exception as e:
            logger.exception(f"Error reading DOCX file {file_path}: {str(e)}")
            return ""

    def _parse_response(self, response_text: str) -> Dict[str, str]:
        """
        Parse the response from Claude to extract figure captions.

        Args:
            response_text (str): The response text from Claude.

        Returns:
            Dict[str, str]: A dictionary of figure labels and their captions.
        """
        logger.debug(f"Raw response from Claude: {response_text[:1000]}...")  # Log first 1000 characters

        def extract_json(s):
            """Extract JSON object from a string that may contain other text."""
            json_match = re.search(r'\{[\s\S]*\}', s)
            return json_match.group(0) if json_match else None

        def parse_json(s):
            """Parse JSON string using custom decoder."""
            try:
                return json.loads(s, cls=CustomJSONDecoder)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON: {s}")
                return {}

        # Extract JSON object from the response
        json_str = extract_json(response_text)
        if json_str:
            captions = parse_json(json_str)
        else:
            logger.error("No JSON object found in the response")
            return {}

        if captions:
            logger.info(f"Successfully extracted {len(captions)} captions")
            return captions
        else:
            logger.error("Failed to extract any captions from Claude's response")
            return {}

    def _parse_figure_captions(self, text: str) -> Dict[str, str]:
        """
        Parse figure captions using regex.

        Args:
            text (str): The text to parse for figure captions.

        Returns:
            Dict[str, str]: A dictionary of figure labels and their captions.
        """
        captions = {}
        # Look for patterns like "Figure 1:", "Figure 1.", or just "Figure 1" followed by text
        matches = re.finditer(r'(Figure \d+[.:]?)(.+?)(?=(Figure \d+[.:]?)|\Z)', text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            figure_label = match.group(1).strip().rstrip('.').rstrip(':')
            caption = match.group(2).strip()
            captions[figure_label] = caption
        logger.debug(f"Parsed {len(captions)} captions using regex")
        return captions
    
    def _update_zip_structure(self, zip_structure: ZipStructure, captions: Dict[str, str]) -> ZipStructure:
        for figure in zip_structure.figures:
            if figure.figure_label in captions:
                figure.figure_caption = captions[figure.figure_label]
            else:
                figure.figure_caption = "Figure caption not found."
        return zip_structure
