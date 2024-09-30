"""
This module provides functionality for extracting figure captions from scientific documents
using the Anthropic API (Claude model).

It includes a class that interacts with the Anthropic API to process document content and extract
figure captions, which are then integrated into a ZipStructure object.
"""

import json
import logging
import re
from typing import Any, Dict

from anthropic import Anthropic
from docx import Document

from ..manuscript_structure.manuscript_structure import ZipStructure
from .extract_captions_base import FigureCaptionExtractor
from .extract_captions_prompts import get_extract_captions_prompt

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

    def extract_captions(
        self, docx_path: str, zip_structure: ZipStructure
    ) -> ZipStructure:
        """
        Extract figure captions from the given DOCX file using Anthropic's Claude model.

        This method processes the DOCX file, sends its content to the Anthropic API,
        and updates the ZipStructure with the extracted captions.

        Args:
            docx_path (str): Path to the DOCX file.
            zip_structure (ZipStructure): The current ZIP structure.

        Returns:
            ZipStructure: Updated ZIP structure with extracted captions.
        """
        try:
            logger.info(f"Processing file: {docx_path}")

            file_content = self._extract_docx_content(docx_path)
            if not file_content:
                logger.warning(f"No content extracted from {docx_path}")
                return self._update_zip_structure(
                    zip_structure, {}, "No content extracted"
                )

            prompt = get_extract_captions_prompt(file_content)

            logger.debug("Sending request to Anthropic API")
            response = self.client.messages.create(
                model=self.config["model"],
                max_tokens=self.config["max_tokens_to_sample"],
                messages=[{"role": "user", "content": prompt}],
            )
            logger.debug(f"Answer from Anthropic: {response}")

            extracted_text = response.content
            if isinstance(extracted_text, list):
                extracted_text = " ".join(
                    item.text for item in extracted_text if hasattr(item, "text")
                )
            elif not isinstance(extracted_text, str):
                extracted_text = str(extracted_text)

            extracted_text = extracted_text.encode("utf-8").decode("utf-8", "ignore")

            logger.debug(f"Extracted text: {extracted_text}")

            captions = self._parse_response(extracted_text)
            if not captions:
                logger.warning("Failed to extract captions from Claude's response")

            updated_structure = self._update_zip_structure(
                zip_structure, captions, extracted_text
            )
            logger.info(f"Updated ZIP structure: {updated_structure}")

            return updated_structure

        except Exception as e:
            logger.exception(f"Error in caption extraction: {str(e)}")
            return self._update_zip_structure(zip_structure, {}, str(e))

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
            return "\n\n".join(paragraphs)
        except Exception as e:
            logger.exception(f"Error reading DOCX file {file_path}: {str(e)}")
            return ""

    def _parse_response(self, response_text: str) -> Dict[str, str]:
        """
        Parse the response from Claude to extract figure captions.

        This method attempts to extract a JSON object from the response text and parse it.
        If successful, it returns a dictionary of figure labels and their captions.

        Args:
            response_text (str): The response text from Claude.

        Returns:
            Dict[str, str]: A dictionary of figure labels and their captions.
        """
        logger.debug(f"Raw response from Claude: {response_text[:1000]}...")

        def extract_json(s):
            """Extract JSON object from a string that may contain other text."""
            json_match = re.search(r"\{[\s\S]*\}", s)
            return json_match.group(0) if json_match else None

        def parse_json(s):
            """Parse JSON string."""
            try:
                # Ensure the JSON string is properly encoded
                return json.loads(s.encode("utf-8").decode("utf-8", "ignore"))
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON: {s}")
                return {}

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

    def _update_zip_structure(
        self, zip_structure: ZipStructure, captions: Dict[str, str], ai_response: str
    ) -> ZipStructure:
        """
        Update the ZipStructure with extracted captions.

        This method updates the figure captions in the ZipStructure based on the extracted captions.
        If a caption is not found for a figure, it sets the caption to "Figure caption not found."

        Args:
            zip_structure (ZipStructure): The current ZIP structure.
            captions (Dict[str, str]): Dictionary of figure labels and their captions.
            ai_response (str): The raw response from the AI model.

        Returns:
            ZipStructure: Updated ZIP structure with new captions.
        """
        for figure in zip_structure.figures:
            if figure.figure_label in captions:
                figure.figure_caption = (
                    captions[figure.figure_label]
                    .encode("utf-8")
                    .decode("utf-8", "ignore")
                )
            else:
                figure.figure_caption = "Figure caption not found."
            figure.ai_response = ai_response
        return zip_structure
