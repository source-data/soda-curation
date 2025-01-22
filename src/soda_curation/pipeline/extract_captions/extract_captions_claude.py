"""
This module provides functionality for extracting figure captions from scientific documents
using the Anthropic API (Claude model).

It includes a class that interacts with the Anthropic API to process document content and extract
figure captions, which are then integrated into a ZipStructure object.
"""

import json
import logging
import re
import time
from typing import Any, Dict, Optional, Tuple

import backoff  # You'll need to add this to your pyproject.toml dependencies
import requests.exceptions
from anthropic import (
    Anthropic,
    APIConnectionError,
    APIError,
    APITimeoutError,
    RateLimitError,
)
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
        super().__init__(config)
        self.client = Anthropic(api_key=self.config["api_key"])
        self.model = self.config.get("model", "claude-3-5-sonnet-20240620")
        self.max_tokens = self.config.get("max_tokens_to_sample", 8192)
        self.temperature = self.config.get("temperature", 0.5)
        self.top_p = self.config.get("top_p", 1.0)
        self.top_k = self.config.get("top_k", 128)
        self.max_retries = self.config.get("max_retries", 5)
        self.initial_wait = self.config.get("initial_wait", 1)
        logger.info(f"Initialized FigureCaptionExtractorClaude with model: {self.model}")

    def _backoff_hdlr(details):
        """Handler for backoff events"""
        logger.warning(f"Backing off {details['wait']:0.1f} seconds after {details['tries']} tries. Error: {details['exception']}")

    def _giveup_hdlr(details):
        """Handler for deciding when to give up"""
        logger.error(f"Giving up after {details['tries']} tries. Last error: {details['exception']}")
        
    @backoff.on_exception(
        backoff.expo,
        (APIConnectionError, APITimeoutError, RateLimitError, APIError, requests.exceptions.RequestException),
        max_tries=5,
        on_backoff=_backoff_hdlr,
        on_giveup=_giveup_hdlr,
        base=2,
        factor=1
    )

    def _make_anthropic_call(self, message_content: str, system_prompt: str) -> Optional[str]:
        """
        Make a resilient call to the Anthropic API with retries and error handling.
        
        Args:
            message_content: The content to send to the API
            system_prompt: The system prompt to use
            
        Returns:
            Optional[str]: The API response text or None if all retries failed
        """
        try:
            response = self.client.messages.create(
                model=self.model,
                system=system_prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                messages=[
                    {"role": "user", "content": message_content},
                ]
            )
            return self._extract_text_from_response(response)
            
        except RateLimitError as e:
            logger.warning(f"Rate limit exceeded: {str(e)}", exc_info=True)
            raise  # Let backoff handle the retry
            
        except APITimeoutError as e:
            logger.warning(f"API timeout: {str(e)}", exc_info=True)
            raise  # Let backoff handle the retry
            
        except APIConnectionError as e:
            logger.warning(f"Connection error: {str(e)}", exc_info=True)
            raise  # Let backoff handle the retry
            
        except APIError as e:
            if e.status_code >= 500:
                logger.warning(f"Server error: {str(e)}", exc_info=True)
                raise  # Let backoff handle the retry
            else:
                logger.error(f"API error: {str(e)}", exc_info=True)
                return None
                
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}", exc_info=True)
            return None

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
            
            response = self._make_anthropic_call(
                message_content=message_content,
                system_prompt=CLAUDE_LOCATE_CAPTIONS_PROMPT
            )
            
            if response is None:
                logger.error("Failed to get response from Anthropic API")
                return ""
                
            return response
            
        except Exception as e:
            logger.error(f"Error locating figure captions: {str(e)}")
            return ""

    def extract_captions(self,
        file_content: str,
        zip_structure,
        expected_figure_count: int,
        expected_figure_labels: str) -> ZipStructure:
        """Extract figure captions from document content."""
        
        try:
            logger.info("Processing document content")
            
            logger.info(f"****************")
            logger.info(f"EXTRACTED FILE CONTENT FROM DOCX FILE")
            logger.info(f"****************")
            logger.info(file_content)
            # Locate all captions and store raw response
            located_captions = self._locate_figure_captions(
                file_content,
                expected_figure_count,
                expected_figure_labels
            )
            
            if not located_captions:
                logger.error("Failed to locate figure captions")
                return zip_structure
            
            # Store raw located captions text and AI response
            zip_structure.ai_response_locate_captions = located_captions
            logger.info(f"Successfully located captions section ({len(located_captions)} characters)")
            
            # Extract individual captions
            extracted_captions_response = self._extract_individual_captions(
                located_captions,
                expected_figure_count,
                expected_figure_labels
            )
            
            # Store the raw response from caption extraction
            zip_structure.ai_response_extract_captions = extracted_captions_response
            
            # Parse the response into caption dictionary
            captions = self._parse_response(extracted_captions_response)
            logger.info(f"****************")
            logger.info(f"Extracted {len(captions)} individual captions")
            logger.info(f"****************")
            logger.info(captions)

            # Process each figure
            for figure in zip_structure.figures:
                normalized_label = self.normalize_figure_label(figure.figure_label)
                logger.info(f"Processing {normalized_label}")
                
                if normalized_label in captions:
                    caption_info = captions[normalized_label]
                    caption_text = caption_info["caption"]
                    caption_title = caption_info["title"]
                    
                    # Validate caption and get diff
                    is_valid, rouge_score, diff_text = self._validate_caption(docx_path, caption_text)
                    
                    figure.figure_caption = caption_text
                    figure.caption_title = caption_title
                    figure.rouge_l_score = rouge_score
                    figure.possible_hallucination = not is_valid
                    figure.diff = diff_text  # Store the diff text
                else:
                    logger.warning(f"No caption found for {figure.figure_label}")
                    figure.figure_caption = "Figure caption not found."
                    figure.caption_title = ""
                    figure.rouge_l_score = 0.0
                    figure.possible_hallucination = True
                    figure.diff = ""
            
            return zip_structure
            
        except Exception as e:
            logger.error(f"Error in caption extraction: {str(e)}", exc_info=True)
            return zip_structure

    def _extract_text_from_response(self, response) -> str:
        """Helper method to extract text from Claude's response with better error handling."""
        try:
            if response is None:
                return ""
                
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
            
            response = self._make_anthropic_call(
                message_content=message_content,
                system_prompt=CLAUDE_EXTRACT_CAPTIONS_PROMPT
            )
            
            if response is None:
                logger.error("Failed to get response from Anthropic API")
                return {}
                
            return response
            
        except Exception as e:
            logger.error(f"Error extracting individual captions: {str(e)}")
            return {}


