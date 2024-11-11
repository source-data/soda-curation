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
    EXTRACT_CAPTIONS_PROMPT,
    LOCATE_CAPTIONS_PROMPT,
    get_extract_captions_prompt,
    get_locate_captions_prompt,
)

logger = logging.getLogger(__name__)

def normalize_text(text: str) -> str:
    """Normalize text for caption comparison."""
    text = text.lower()
    text = re.sub(r'^(figure|fig\.?)\s*\d+[\.:]?\s*', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[\(\)]', '', text)
    text = re.sub(r'["\']', '', text)
    text = re.sub(r'±', 'plus/minus', text)
    text = re.sub(r'[\(\)\[\]\{\}]', '', text)
    text = re.sub(r'[,;:]', ' ', text)
    text = re.sub(r'(\d+)\s*/\s*(\d+)', r'\1/\2', text)
    text = re.sub(r'(\d+)\s+(mm|nm|µm|ml)', r'\1\2', text)
    return text.strip()

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

    def _extract_docx_content(self, file_path: str) -> str:
        """
        Extract text content from a DOCX file.

        Args:
            file_path (str): Path to the DOCX file.

        Returns:
            str: Extracted text content from the DOCX file.
        """
        try:
            doc = Document(file_path)
            paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
            return " ".join(paragraphs)
        except Exception as e:
            logger.exception(f"Error reading DOCX file {file_path}: {str(e)}")
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
            message_content = get_locate_captions_prompt(
                manuscript_text=doc_string,
                expected_figure_count=expected_figure_count,
                expected_figure_labels=expected_figure_labels
            )
            response = self.client.messages.create(
                model=self.config["model"],
                max_tokens=self.config["max_tokens_to_sample"],
                messages=[
                    {"role": "system", "content": LOCATE_CAPTIONS_PROMPT},
                    {"role": "user", "content": message_content},
                ],
            )
            
            extracted_text = response.content
            extracted_text = extracted_text.encode("utf-8").decode("utf-8", "ignore")
            
            if isinstance(extracted_text, list):
                extracted_text = "\n".join(
                    item.text for item in extracted_text if hasattr(item, "text")
                )
                
            elif not isinstance(extracted_text, str):
                extracted_text = str(extracted_text)

            return response.content
        
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
            
            logger.debug(f"Answer from Anthropic: {all_figure_captions}")
            
            # Store raw captions
            zip_structure.all_captions_extracted = all_figure_captions
            logger.info(f"Successfully located captions section ({len(all_figure_captions)} characters)")

            # Step 2: Extract individual captions
            captions = self._extract_individual_captions(
                all_figure_captions,
                expected_figure_count,
                expected_figure_labels)
            logger.info(f"Extracted {len(captions)} individual captions")

            # Step 3: Update structure with validated captions
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
        
    def _parse_response(self, response_text: str) -> Dict[str, str]:
        """Parse JSON response containing figure captions."""
        try:
            json_match = re.search(r'```json\s*({[\s\S]*?})\s*```', response_text)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_match = re.search(r'({[^{]*})', response_text)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_str = response_text.strip()

            json_str = re.sub(r'[\n\r\t]', ' ', json_str)
            json_str = re.sub(r'\s+', ' ', json_str)

            try:
                captions = json.loads(json_str)
            except json.JSONDecodeError:
                json_str = re.sub(r'([{,])\s*([a-zA-Z0-9_]+)\s*:', r'\1"\2":', json_str)
                captions = json.loads(json_str)

            cleaned_captions = {}
            for key, value in captions.items():
                if isinstance(value, str):
                    value = value.strip()
                    if value and value != "Figure caption not found.":
                        clean_key = re.sub(r'^(Fig\.|Figure)\s*', 'Figure ', key)
                        cleaned_captions[clean_key] = value

            return cleaned_captions

        except Exception as e:
            logger.error(f"Error parsing response: {str(e)}")
            return {}

    def _validate_caption(self, docx_path: str, caption: str) -> Tuple[bool, float]:
        """Validate extracted caption against document text."""
        try:
            doc = Document(docx_path)
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            norm_caption = normalize_text(caption)
            
            max_score = 0
            for para in paragraphs:
                norm_para = normalize_text(para)
                ratio_score = fuzz.ratio(norm_caption, norm_para)
                token_sort = fuzz.token_sort_ratio(norm_caption, norm_para)
                token_set = fuzz.token_set_ratio(norm_caption, norm_para)
                
                score = max(ratio_score, token_sort, token_set)
                max_score = max(max_score, score)
                
                if max_score >= 85:
                    return True, max_score

            return False, max_score

        except Exception as e:
            logger.error(f"Error in caption validation: {str(e)}")
            return False, 0

    def _extract_individual_captions(self,
        all_captions: str,
        expected_figure_count: int,
        expected_figure_labels: str) -> Dict[str, str]:
        """Extract individual figure captions using AI."""
        try:
            # Create thread for extraction
            message_content = get_extract_captions_prompt(
                figure_captions=all_captions,
                expected_figure_count=expected_figure_count,
                expected_figure_labels=expected_figure_labels
                )
            
            response = self.client.beta.threads.create(
                messages = [
                    {"role": "system", "content": EXTRACT_CAPTIONS_PROMPT},
                    {"role": "user", "content": message_content},
                ]
            )

            # Run the assistant
            extracted_text = response.content
            extracted_text = extracted_text.encode("utf-8").decode("utf-8", "ignore")
            
            if isinstance(extracted_text, list):
                extracted_text = "\n".join(
                    item.text for item in extracted_text if hasattr(item, "text")
                )
                
            elif not isinstance(extracted_text, str):
                extracted_text = str(extracted_text)
                
            # Parse response
            if extracted_text:
                extracted_captions = self._parse_response(
                    extracted_text
                )
                return extracted_captions
        except Exception as e:
            logger.error(f"Error extracting individual captions: {str(e)}")
            return {}
