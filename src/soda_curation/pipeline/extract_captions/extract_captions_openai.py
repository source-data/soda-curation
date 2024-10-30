"""
This module provides functionality for extracting figure captions from scientific documents
using the OpenAI API (GPT model).

It implements a two-step process:
1. Locate the section containing all figure captions
2. Parse individual captions from that section
"""

import json
import logging
import os
import re
import time
from typing import Dict, Optional, Tuple

import openai
from docx import Document
from fuzzywuzzy import fuzz
from openai.types.beta import Thread
from openai.types.file_object import FileObject

from ..manuscript_structure.manuscript_structure import Figure, ZipStructure
from .extract_captions_base import FigureCaptionExtractor
from .extract_captions_prompts import (
    get_extract_captions_prompt,
    get_locate_captions_prompt,
)

logger = logging.getLogger(__name__)

def normalize_text(text: str) -> str:
    """
    Normalize text for comparison by removing extra spaces and standardizing format.
    
    Args:
        text (str): The text to normalize
        
    Returns:
        str: The normalized text
    """
    # Convert to lowercase and remove extra whitespace
    text = ' '.join(text.lower().split())
    
    # Handle various figure reference formats
    text = re.sub(r'fig\.\s*(\d+)', r'figure \1', text)
    text = re.sub(r'fig\s+(\d+)', r'figure \1', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,;:?!-]', '', text)
    
    return text.strip()

def format_caption_differences(original: str, extracted: str) -> str:
    """
    Format the differences between original and extracted captions with HTML.
    
    Args:
        original (str): The original caption from the document
        extracted (str): The extracted caption from the AI
        
    Returns:
        str: HTML formatted string showing the differences
    """
    import difflib

    # Normalize both texts for comparison
    original_norm = normalize_text(original)
    extracted_norm = normalize_text(extracted)
    
    # Generate diff
    diff = difflib.HtmlDiff()
    html_diff = diff.make_file(original_norm.splitlines(), extracted_norm.splitlines(),
                             context=True, numlines=3)
    
    return html_diff

class FigureCaptionExtractorGpt(FigureCaptionExtractor):
    """
    A class to extract figure captions using OpenAI's GPT model.
    
    This implementation uses a two-step process:
    1. First locate the section containing all figure captions
    2. Then extract and parse individual captions from that section
    """
    def __init__(self, config: Dict):
        """
        Initialize the extractor with OpenAI configuration.
        
        Args:
            config (Dict): Configuration dictionary containing OpenAI settings
        """
        self.config = config
        self.client = openai.OpenAI(api_key=self.config["api_key"])
        self.assistant = self._setup_assistant()
        logger.info("FigureCaptionExtractorGpt initialized successfully")

    def _setup_assistant(self):
        """Set up or retrieve the OpenAI assistant."""
        assistant_id = self.config.get("caption_extraction_assistant_id")
        if not assistant_id:
            raise ValueError("caption_extraction_assistant_id is not set in configuration")
        
        return self.client.beta.assistants.update(
            assistant_id,
            model=self.config["model"],
            instructions=get_locate_captions_prompt()
        )

    def _upload_file(self, file_path: str) -> FileObject:
        """
        Upload a file to the OpenAI API for processing.

        Args:
            file_path (str): The path to the file to be uploaded.

        Returns:
            FileObject: The uploaded file object.
        """
        with open(file_path, "rb") as file:
            file_object = self.client.files.create(file=file, purpose="assistants")
        return file_object

    def _prepare_query(self, file_path: str, prompt: str) -> Tuple[str, FileObject]:
        """
        Prepare the query to be sent to the OpenAI assistant.

        Args:
            file_path (str): The file path to the DOCX file.
            prompt (str): The prompt to use for the query.

        Returns:
            Tuple[str, FileObject]: Thread ID and file object.
        """
        file_on_client = self._upload_file(file_path)
        thread = self.client.beta.threads.create(
            messages=[{
                "role": "user",
                "content": prompt,
                "attachments": [{
                    "file_id": file_on_client.id,
                    "tools": [{"type": "file_search"}]
                }]
            }]
        )
        return thread.id, file_on_client

    def extract_captions(self, docx_path: str, zip_structure: ZipStructure, expected_figure_count: int) -> ZipStructure:
        """
        Extract figure captions using a two-step process.
        
        Args:
            docx_path (str): Path to the DOCX file
            zip_structure (ZipStructure): Current ZIP structure
            expected_figure_count (int): Expected number of figures
            
        Returns:
            ZipStructure: Updated structure with extracted captions
        """
        try:
            # Step 1: Locate all figure captions
            logger.info("Step 1: Locating figure captions section")
            all_captions = self._locate_figure_captions(docx_path)
            if not all_captions:
                logger.error("Failed to locate figure captions section")
                return zip_structure
            
            # Store raw extracted captions
            zip_structure.all_captions_extracted = all_captions
            logger.info("Successfully located figure captions section")
            
            # Step 2: Extract individual captions
            logger.info("Step 2: Extracting individual captions")
            captions = self._extract_individual_captions(all_captions, expected_figure_count)
            if not captions:
                logger.error("Failed to extract individual captions")
                return zip_structure
            
            # Update structure with extracted captions
            zip_structure = self._update_structure_with_captions(zip_structure, captions, docx_path)
            
            return zip_structure
            
        except Exception as e:
            logger.error(f"Error in caption extraction: {str(e)}")
            return zip_structure

    def _locate_figure_captions(self, docx_path: str) -> Optional[str]:
        """
        Locate and extract all figure captions from the document using OpenAI's file handling.
        
        Args:
            docx_path (str): Path to the DOCX file
            
        Returns:
            Optional[str]: Text containing all located captions, or None if not found
        """
        try:
            # Prepare query with file upload
            thread_id, file_object = self._prepare_query(docx_path, get_locate_captions_prompt())
            logger.info(f"Created thread {thread_id} for caption location")
            
            try:
                # Run the assistant
                run = self.client.beta.threads.runs.create(
                    thread_id=thread_id,
                    assistant_id=self.assistant.id
                )

                # Wait for completion
                while run.status != "completed":
                    run = self.client.beta.threads.runs.retrieve(
                        thread_id=thread_id,
                        run_id=run.id
                    )
                    time.sleep(1)
                    
                # Get the response
                messages = self.client.beta.threads.messages.list(thread_id=thread_id)
                
                if messages.data:
                    result = messages.data[0].content[0].text.value
                    logger.info("Successfully located figure captions section")
                    return result.strip()
                else:
                    logger.warning("No response received for caption location")
                    return None

            finally:
                # Clean up
                try:
                    self.client.files.delete(file_object.id)
                    logger.debug(f"Deleted file {file_object.id}")
                except Exception as del_err:
                    logger.warning(f"Error deleting file {file_object.id}: {del_err}")
                    
        except Exception as e:
            logger.error(f"Error locating figure captions: {str(e)}")
            return None

    def _extract_individual_captions(self, all_captions: str, expected_figure_count: int) -> Dict[str, str]:
        """
        Extract individual figure captions from the located captions text.
        
        Args:
            all_captions (str): Text containing all figure captions
            expected_figure_count (int): Expected number of figures
            
        Returns:
            Dict[str, str]: Dictionary mapping figure labels to captions
        """
        try:
            # Call API to parse captions
            response = self.client.chat.completions.create(
                model=self.config.get("model", "gpt-4-turbo-preview"),
                messages=[
                    {
                        "role": "system", 
                        "content": get_extract_captions_prompt(all_captions, expected_figure_count)
                    }
                ],
                temperature=0.3
            )
            
            if not response.choices:
                return {}
                
            return self._parse_response(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"Error extracting individual captions: {str(e)}")
            return {}

    def _parse_response(self, response_text: str) -> Dict[str, str]:
        """
        Parse the JSON response containing figure captions.
        
        Args:
            response_text (str): The response text from the API
            
        Returns:
            Dict[str, str]: Dictionary mapping figure labels to captions
        """
        try:
            # Try to find JSON block
            json_match = re.search(r'```json\s*({[\s\S]*?})\s*```', response_text)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find any JSON object
                json_match = re.search(r'({[^{]*})', response_text)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_str = response_text.strip()
            
            # Clean up and parse JSON
            json_str = re.sub(r'[\n\r\t]', ' ', json_str)
            json_str = re.sub(r'\s+', ' ', json_str)
            
            try:
                captions = json.loads(json_str)
            except json.JSONDecodeError:
                # Try to fix common JSON issues
                json_str = re.sub(r'([{,])\s*([a-zA-Z0-9_]+)\s*:', r'\1"\2":', json_str)
                captions = json.loads(json_str)
            
            # Clean up captions
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

    def _update_structure_with_captions(self, zip_structure: ZipStructure, 
                                    captions: Dict[str, str], docx_path: str) -> ZipStructure:
        """
        Update the ZipStructure with extracted captions and validation info.
        
        Args:
            zip_structure (ZipStructure): Structure to update
            captions (Dict[str, str]): Extracted captions
            docx_path (str): Path to original document for validation
            
        Returns:
            ZipStructure: Updated structure
        """
        for figure in zip_structure.figures:
            if figure.figure_label in captions:
                caption = captions[figure.figure_label]
                is_valid, score = self._validate_caption(docx_path, caption)
                
                if score >= 80:
                    figure.figure_caption = caption
                    figure.caption_fuzzy_score = score
                    figure.possible_hallucination = score < 95
                else:
                    logger.warning(f"Low confidence caption for {figure.figure_label}: {score}")
                    figure.figure_caption = "Figure caption not found."
                    figure.caption_fuzzy_score = score
            else:
                logger.warning(f"No caption found for {figure.figure_label}")
                figure.figure_caption = "Figure caption not found."
                figure.caption_fuzzy_score = 0
                
        return zip_structure

    def _validate_caption(self, docx_path: str, caption: str, fuzzy_threshold: int = 85) -> Tuple[bool, float]:
        """
        Validate extracted caption against the original document.
        
        Args:
            docx_path (str): Path to the DOCX file
            caption (str): Caption to validate
            fuzzy_threshold (int): Minimum score for valid caption
            
        Returns:
            Tuple[bool, float]: (is_valid, confidence_score)
        """
        try:
            doc = Document(docx_path)
            paragraphs = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
            
            # Try different comparison methods
            max_score = 0
            for paragraph in paragraphs:
                ratio_score = fuzz.ratio(
                    normalize_text(caption), 
                    normalize_text(paragraph)
                )
                token_sort_score = fuzz.token_sort_ratio(
                    normalize_text(caption), 
                    normalize_text(paragraph)
                )
                max_score = max(max_score, ratio_score, token_sort_score)
                
                if max_score >= fuzzy_threshold:
                    return True, max_score
                    
            return False, max_score
            
        except Exception as e:
            logger.error(f"Error in caption validation: {str(e)}")
            return False, 0
