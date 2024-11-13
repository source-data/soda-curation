"""
This module provides the base class for figure caption extraction.

It defines the abstract base class that all specific caption extractor implementations
should inherit from, ensuring a consistent interface across different extraction methods.
"""

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

from docx import Document
from fuzzywuzzy import fuzz

from ..manuscript_structure.manuscript_structure import ZipStructure

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

class FigureCaptionExtractor(ABC):
    """
    Abstract base class for extracting figure captions from a document.

    This class provides common functionality and defines the interface that all
    figure caption extractors should implement.
    """

    def __init__(self, config: Dict):
        """Initialize with configuration."""
        self.config = config

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

    def _parse_response(self, response_text: str) -> Dict[str, str]:
        """
        Parse JSON response containing figure captions.

        Args:
            response_text (str): The response text to parse.

        Returns:
            Dict[str, str]: Dictionary of figure labels and their captions.
        """
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
        """
        Validate extracted caption against document text.

        Args:
            docx_path (str): Path to the DOCX file
            caption (str): Caption to validate

        Returns:
            Tuple[bool, float]: Validation result and confidence score
        """
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

    @abstractmethod
    def _locate_figure_captions(self, doc_string: str, expected_figure_count: int, expected_figure_labels: str) -> Optional[str]:
        """
        Locate figure captions in the document using AI.
        
        Args:
            doc_string (str): Document text content
            expected_figure_count (int): Expected number of figures
            expected_figure_labels (str): Expected figure labels
            
        Returns:
            Optional[str]: Located captions text or None
        """
        pass

    @abstractmethod
    def _extract_individual_captions(self, all_captions: str, expected_figure_count: int, expected_figure_labels: str) -> Dict[str, str]:
        """
        Extract individual figure captions from the located captions text.
        
        Args:
            all_captions (str): Text containing all captions
            expected_figure_count (int): Expected number of figures
            expected_figure_labels (str): Expected figure labels
            
        Returns:
            Dict[str, str]: Dictionary of figure labels and their captions
        """
        pass

    @abstractmethod
    def extract_captions(self, docx_path: str, zip_structure: ZipStructure, expected_figure_count: int, expected_figure_labels: str) -> ZipStructure:
        """
        Extract captions from the document and update ZipStructure.
        
        Args:
            docx_path (str): Path to DOCX file
            zip_structure (ZipStructure): Current ZIP structure
            expected_figure_count (int): Expected number of figures
            expected_figure_labels (str): Expected figure labels
            
        Returns:
            ZipStructure: Updated ZIP structure with extracted captions
        """
        pass
