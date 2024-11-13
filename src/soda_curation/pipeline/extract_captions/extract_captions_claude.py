"""
This module provides functionality for extracting figure captions from scientific documents
using the Anthropic API (Claude model).

It includes a class that interacts with the Anthropic API to process document content and extract
figure captions, which are then integrated into a ZipStructure object.
"""

import json
import logging
import re
from typing import Any, Dict, Tuple

from anthropic import Anthropic
from docx import Document
from fuzzywuzzy import fuzz

from ..manuscript_structure.manuscript_structure import Figure, ZipStructure
from .extract_captions_base import FigureCaptionExtractor
from .extract_captions_prompts import (
    CLAUDE_EXTRACT_CAPTIONS_PROMPT,
    CLAUDE_LOCATE_CAPTIONS_PROMPT,
    get_claude_extract_captions_prompt,
    get_claude_locate_captions_prompt,
)

logger = logging.getLogger(__name__)

class FigureCaptionExtractorClaude(FigureCaptionExtractor):
    """
    A class to extract figure captions using Anthropic's Claude model.

    This class provides methods to interact with the Anthropic API, process DOCX files,
    and extract figure captions using the Claude model.

    Attributes:
        config (Dict[str, Any]): Configuration dictionary for Anthropic API.
        client (Anthropic): Anthropic API client.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the FigureCaptionExtractorClaude instance.

        Args:
            config (Dict[str, Any]): Configuration dictionary for Anthropic API.
        """
        self.config = config
        self.client = Anthropic(api_key=self.config["api_key"])

    def _extract_text_from_response(self, response) -> str:
        """Helper method to extract text from Claude's response."""
        try:
            # The content is a list of message contents
            if hasattr(response, 'content') and response.content:
                # Extract text from the first content block
                content = response.content[0]
                if hasattr(content, 'text'):
                    return content.text
                return str(content)
            return ""
        except Exception as e:
            logger.error(f"Error extracting text from response: {str(e)}")
            return ""

    def _locate_figure_captions(self, doc_string: str, expected_figure_count: int, expected_figure_labels) -> str:
        """
        Finds figure captions in the document using the Anthropic API.
        
        Args:
            doc_string (str): Text content of the document.
            expected_figure_count (int): Expected number of figures in the document.
            expected_figure_labels (str): Expected figure labels.

        Returns:
            str: Extracted figure captions.
        """
        try:
            message_content = get_claude_locate_captions_prompt(
                manuscript_text=doc_string,
                expected_figure_count=expected_figure_count,
                expected_figure_labels=expected_figure_labels
            )
            response = self.client.messages.create(
                model=self.config["model"],
                system=CLAUDE_LOCATE_CAPTIONS_PROMPT,
                max_tokens=self.config["max_tokens_to_sample"],
                messages=[
                    {"role": "user", "content": message_content},
                ],
            )
            
            extracted_text = self._extract_text_from_response(response)
            logger.info(f"EXTRACTED TEXT LOCATE CAPTIONS: \n {extracted_text}")
            
            return extracted_text
        
        except Exception as e:
            logger.error(f"Error locating figure captions: {str(e)}")
            return ""

    def extract_captions(self,
            docx_path: str,
            zip_structure: ZipStructure,
            expected_figure_count: int,
            expected_figure_labels: str
        ) -> ZipStructure:
        """
        Extract figure captions from the given DOCX file using Anthropic's Claude model.

        This method processes the DOCX file, sends its content to the Anthropic API,
        and updates the ZipStructure with the extracted captions.

        Args:
            docx_path (str): Path to the DOCX file.
            zip_structure (ZipStructure): The current ZIP structure.
            expected_figure_count (int): The expected number of figures in the document.

        Returns:
            ZipStructure: Updated ZIP structure with extracted captions.
        """
        try:
            logger.info(f"Processing file: {docx_path}")

            file_content = self._extract_docx_content(docx_path)
            logger.debug("Sending request to Anthropic API")
            
            all_figure_captions = self._locate_figure_captions(
                file_content,
                expected_figure_count=expected_figure_count,
                expected_figure_labels=expected_figure_labels
            )
            
            if not all_figure_captions:
                logger.error("Failed to locate figure captions")
                return zip_structure
            
            logger.debug(f"Answer from Anthropic: {all_figure_captions}")
            
            # Store raw captions
            zip_structure.all_captions_extracted = all_figure_captions
            logger.info(f"Successfully located captions section ({len(all_figure_captions)} characters)")

            # Extract individual captions
            captions = self._extract_individual_captions(
                all_figure_captions,
                expected_figure_count,
                expected_figure_labels)
            logger.info(f"Extracted {len(captions)} individual captions")

            # Update structure with validated captions
            for figure in zip_structure.figures:
                logger.info(f"Processing {figure.figure_label}")
                
                if figure.figure_label in captions:
                    caption_text = captions[figure.figure_label]
                    is_valid, score = self._validate_caption(docx_path, caption_text)
                    
                    figure.figure_caption = caption_text
                    figure.caption_fuzzy_score = score
                    figure.possible_hallucination = score < 85
                else:
                    logger.warning(f"No caption found for {figure.figure_label}")
                    figure.figure_caption = "Figure caption not found."
                    figure.caption_fuzzy_score = 0
                    figure.possible_hallucination = True

            return zip_structure

        except Exception as e:
            logger.error(f"Error in caption extraction: {str(e)}", exc_info=True)
            return zip_structure
        
    def _extract_individual_captions(self,
        all_captions: str,
        expected_figure_count: int,
        expected_figure_labels: str) -> Dict[str, str]:
        """Extract individual figure captions using AI."""
        try:
            message_content = get_claude_extract_captions_prompt(
                figure_captions=all_captions,
                expected_figure_count=expected_figure_count,
                expected_figure_labels=expected_figure_labels
                )
            
            response = self.client.messages.create(
                model=self.config["model"],
                system=CLAUDE_EXTRACT_CAPTIONS_PROMPT,
                max_tokens=self.config["max_tokens_to_sample"],
                messages=[
                    {"role": "user", "content": message_content},
                ],
            )

            extracted_text = self._extract_text_from_response(response)
            logger.info(f"EXTRACTED TEXT INDIVIDUAL CAPTIONS: \n {extracted_text}")

            if extracted_text:
                return self._parse_response(extracted_text)
            return {}
            
        except Exception as e:
            logger.error(f"Error extracting individual captions: {str(e)}")
            return {}

