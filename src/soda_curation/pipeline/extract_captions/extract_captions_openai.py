import json
import logging
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import openai
from docx import Document
from fuzzywuzzy import fuzz
from openai.types.beta import Thread
from openai.types.file_object import FileObject

from ..manuscript_structure.manuscript_structure import Figure, ZipStructure
from .extract_captions_base import FigureCaptionExtractor
from .extract_captions_prompts import (
    EXTRACT_CAPTIONS_PROMPT,
    LOCATE_CAPTIONS_PROMPT,
    get_extract_captions_prompt,
    get_locate_captions_prompt,
)

logger = logging.getLogger(__name__)

class FigureCaptionExtractorGpt(FigureCaptionExtractor):
    """A class to extract figure captions using OpenAI's GPT model."""
    
    def __init__(self, config: Dict):
        """Initialize the extractor with OpenAI configuration."""
        self.config = config
        self.client = openai.OpenAI(api_key=self.config["api_key"])
        self.locate_assistant = self._setup_locate_assistant()
        self.extract_assistant = self._setup_extract_assistant()
        logger.info("FigureCaptionExtractorGpt initialized successfully")

    def _setup_locate_assistant(self):
        """Set up or retrieve the OpenAI assistant for locating captions."""
        assistant_id = self.config.get("caption_location_assistant_id")
        if not assistant_id:
            raise ValueError("caption_location_assistant_id is not set in configuration")
        
        return self.client.beta.assistants.update(
            assistant_id,
            model=self.config["model"],
            instructions=LOCATE_CAPTIONS_PROMPT
        )

    def _setup_extract_assistant(self):
        """Set up or retrieve the OpenAI assistant for extracting individual captions."""
        assistant_id = self.config.get("caption_extraction_assistant_id")
        if not assistant_id:
            raise ValueError("caption_extraction_assistant_id is not set in configuration")
        
        return self.client.beta.assistants.update(
            assistant_id,
            model=self.config["model"],
            instructions=EXTRACT_CAPTIONS_PROMPT
        )

    def _cleanup_thread(self, thread_id: str):
        """
        Helper method to clean up a thread and its messages.
        
        Args:
            thread_id: The ID of the thread to clean up
        """
        try:
            # Delete the thread which also deletes all associated messages
            self.client.beta.threads.delete(thread_id=thread_id)
            logger.info(f"Successfully cleaned up thread {thread_id}")
        except Exception as e:
            logger.error(f"Error cleaning up thread {thread_id}: {str(e)}")

    def _locate_figure_captions(self,
        doc_string: str,
        expected_figure_count: int = None,
        expected_figure_labels: str = "") -> Optional[str]:
        """
        Locate figure captions using AI, optionally using both DOCX and PDF files.
        
        Args:
            docx_path (str): Path to DOCX file
            pdf_path (str): Optional path to PDF file
            expected_figure_count (int): Expected number of figures (from Figure 1 to X)
            
        Returns:
            Optional[str]: Found captions text or None
        """
        try:
            # Create message with enhanced prompt that mentions all available files
            message_content = get_locate_captions_prompt(
                manuscript_text=doc_string,
                expected_figure_count=expected_figure_count,
                expected_figure_labels=expected_figure_labels,
            )
            
            thread = self.client.beta.threads.create(
                messages = [
                    {"role": "user", "content": message_content},
                ]
            )

            # Rest of the assistant communication code...
            run = self.client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=self.locate_assistant.id
            )

            # Wait for completion with timeout
            start_time = time.time()
            timeout = 300  # 5 minutes timeout
            while run.status not in ["completed", "failed"]:
                if time.time() - start_time > timeout:
                    logger.error("Assistant run timed out")
                    return None
                    
                time.sleep(1)
                run = self.client.beta.threads.runs.retrieve(
                    thread_id=thread.id,
                    run_id=run.id
                )
                
                if run.status == "failed":
                    logger.error(f"Assistant run failed: {run.last_error}")
                    return None

            # Get response
            messages = self.client.beta.threads.messages.list(thread_id=thread.id)
            
            if messages.data:
                located_captions =  messages.data[0].content[0].text.value
                self._cleanup_thread(thread.id)
                return located_captions
            return None
        except Exception as e:
            logger.error(f"Error locating figure captions: {str(e)}")
            return None
        
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
            
            thread = self.client.beta.threads.create(
                messages = [
                    # {"role": "system", "content": EXTRACT_CAPTIONS_PROMPT},
                    {"role": "user", "content": message_content},
                ]
            )

            # Run the assistant
            run = self.client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=self.extract_assistant.id
            )

            # Wait for completion with timeout
            start_time = time.time()
            timeout = 300  # 5 minutes timeout
            while run.status not in ["completed", "failed"]:
                if time.time() - start_time > timeout:
                    logger.error("Assistant run timed out")
                    return {}
                    
                time.sleep(1)
                run = self.client.beta.threads.runs.retrieve(
                    thread_id=thread.id,
                    run_id=run.id
                )
                
                if run.status == "failed":
                    logger.error(f"Assistant run failed: {run.last_error}")
                    return {}

            # Get response
            messages = self.client.beta.threads.messages.list(thread_id=thread.id)
            
            if messages.data:
                self._cleanup_thread(thread.id)
                return messages.data[0].content[0].text.value

        except Exception as e:
            logger.error(f"Error extracting individual captions: {str(e)}")

            return {}

    def extract_captions(self,
        docx_path: str,
        zip_structure,
        expected_figure_count: int,
        expected_figure_labels: str) -> ZipStructure:
        """Extract figure captions from document."""
        
        try:
            logger.info(f"Processing file: {docx_path}")
            file_content = self._extract_docx_content(docx_path)
            
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
            logger.info(f"Extracted {len(captions)} individual captions")
            
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

