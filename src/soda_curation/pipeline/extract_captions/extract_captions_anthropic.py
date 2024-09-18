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
            
            extracted_text = response.content
            logger.debug(f"Extracted text type: {type(extracted_text)}")
            logger.debug(f"First 1000 characters of extracted text: {extracted_text[:1000]}...")
            
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

    def _parse_response(self, response_text: Union[str, list]) -> Dict[str, str]:
        """
        Parse the response from Claude to extract figure captions.

        Args:
            response_text (Union[str, list]): The response text from Claude.

        Returns:
            Dict[str, str]: A dictionary of figure labels and their captions.
        """
        logger.debug(f"Type of response_text: {type(response_text)}")
        logger.debug(f"Raw response from Claude: {response_text[:1000]}...")  # Log first 1000 characters

        # If response_text is a list, join it into a single string
        if isinstance(response_text, list):
            response_text = ' '.join(str(item) for item in response_text)

        # First, try to parse the entire response as JSON
        try:
            captions = json.loads(response_text)
            if isinstance(captions, dict):
                logger.info("Successfully parsed entire response as JSON")
                return captions
        except json.JSONDecodeError:
            logger.warning("Failed to parse entire response as JSON. Attempting to extract JSON object.")

        # If parsing the entire response fails, try to extract a JSON object from the text
        try:
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text)
            if json_match:
                captions = json.loads(json_match.group(0))
                if isinstance(captions, dict):
                    logger.info("Successfully extracted and parsed JSON object from response")
                    return captions
        except (re.error, json.JSONDecodeError) as e:
            logger.warning(f"Failed to extract or parse JSON object: {str(e)}")

        # If JSON parsing fails, fall back to regex parsing
        try:
            captions = self._parse_figure_captions(response_text)
            if captions:
                logger.info(f"Extracted {len(captions)} captions using regex parsing")
                return captions
        except Exception as e:
            logger.error(f"Error during regex parsing: {str(e)}")

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
