import json
import os
import re
import shutil
import logging
from typing import Dict, Any
from anthropic import Anthropic
from docx import Document
from docx.opc.exceptions import PackageNotFoundError
from .extract_captions_base import FigureCaptionExtractor
from .extract_captions_prompts import get_extract_captions_prompt
from ..zip_structure.zip_structure_base import ZipStructure

logger = logging.getLogger(__name__)

class FigureCaptionExtractorClaude(FigureCaptionExtractor):
    """
    A class to extract figure captions using Anthropic's Claude model.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the FigureCaptionExtractorClaude instance.

        Args:
            config (Dict[str, Any]): Configuration dictionary for Anthropic API.
        """
        self.config = config
        self.client = Anthropic(api_key=self.config['api_key'])

    def extract_captions(self, docx_path: str, zip_structure: ZipStructure) -> ZipStructure:
        """
        Extract figure captions from the given DOCX file using Anthropic's Claude model.

        Args:
            docx_path (str): Path to the DOCX file.
            zip_structure (ZipStructure): The current ZIP structure.

        Returns:
            ZipStructure: Updated ZIP structure with extracted captions.
        """
        try:
            logger.info(f"Processing file: {docx_path}")
            
            # Copy file to a non-temporary location
            home_dir = os.path.expanduser("~")
            file_name = os.path.basename(docx_path)
            new_file_path = os.path.join(home_dir, file_name)
            shutil.copy2(docx_path, new_file_path)
            logger.debug(f"Copied file to: {new_file_path}")

            file_content = self._extract_docx_content(new_file_path)
            if not file_content:
                logger.warning(f"No content extracted from {new_file_path}")
                return zip_structure

            prompt = get_extract_captions_prompt(file_content)
            
            logger.debug("Sending request to Anthropic API")
            response = self.client.messages.create(
                model=self.config['model'],
                max_tokens=self.config['max_tokens_to_sample'],
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            extracted_text = response.content
            logger.debug(f"Full response from Anthropic API:\n{extracted_text}")
            
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
        """
        Extract content from DOCX file.

        Args:
            file_path (str): Path to the DOCX file.

        Returns:
            str: Extracted content from the DOCX file.
        """
        try:
            doc = Document(file_path)
            paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
            
            # Group paragraphs by figure
            figure_contents = {}
            current_figure = None
            for para in paragraphs:
                if re.match(r'^(Figure|Fig\.?)\s+\d+', para):
                    current_figure = re.match(r'^(Figure|Fig\.?)\s+\d+', para).group()
                    current_figure = current_figure.replace('Fig', 'Figure')
                    figure_contents[current_figure] = [para]
                elif current_figure:
                    figure_contents[current_figure].append(para)
            
            # Join paragraphs for each figure
            return "\n\n".join([f"{fig}\n" + "\n".join(paras) for fig, paras in figure_contents.items()])
        except PackageNotFoundError as e:
            logger.error(f"Unable to open DOCX file. It may be corrupted or not a valid DOCX file: {file_path}")
            logger.error(f"Detailed error: {str(e)}")
            return ""
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
        try:
            # First, try to parse the entire response as JSON
            captions = json.loads(response_text)
            if isinstance(captions, dict):
                return captions
        except json.JSONDecodeError:
            logger.warning("Failed to parse entire response as JSON. Attempting to extract JSON object.")
        
        # If parsing the entire response fails, try to extract a JSON object from the text
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if match:
            try:
                captions = json.loads(match.group(0))
                if isinstance(captions, dict):
                    return captions
            except json.JSONDecodeError:
                logger.warning("Failed to parse extracted JSON object. Falling back to regex parsing.")
        
        # If JSON parsing fails, fall back to regex parsing
        return self._parse_figure_captions(response_text)

    def _parse_figure_captions(self, text: str) -> Dict[str, str]:
        """
        Parse figure captions using regex if JSON parsing fails.

        Args:
            text (str): The text to parse for figure captions.

        Returns:
            Dict[str, str]: A dictionary of figure labels and their captions.
        """
        captions = {}
        matches = re.finditer(r'(Figure \d+[.:])(.+?)(?=(Figure \d+[.:])|\Z)', text, re.DOTALL)
        for match in matches:
            figure_label = match.group(1).strip().rstrip('.').rstrip(':')
            caption = match.group(2).strip()
            captions[figure_label] = caption
        logger.debug(f"Parsed {len(captions)} captions using regex")
        return captions

