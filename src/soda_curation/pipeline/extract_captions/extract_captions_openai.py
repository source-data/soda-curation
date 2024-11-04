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
            instructions=get_locate_captions_prompt()
        )

    def _setup_extract_assistant(self):
        """Set up or retrieve the OpenAI assistant for extracting individual captions."""
        assistant_id = self.config.get("caption_extraction_assistant_id")
        if not assistant_id:
            raise ValueError("caption_extraction_assistant_id is not set in configuration")
        
        return self.client.beta.assistants.update(
            assistant_id,
            model=self.config["model"],
            instructions="You will receive text containing figure captions and must extract individual figure captions. You will be provided with the expected number of figures to extract."
        )

    def _locate_figure_captions(self, docx_path: str, pdf_path: str = None, expected_figure_count: int = None) -> Optional[str]:
        """
        Locate figure captions using AI, optionally using both DOCX and PDF files.
        
        Args:
            docx_path (str): Path to DOCX file
            pdf_path (str): Optional path to PDF file
            expected_figure_count (int): Expected number of figures (from Figure 1 to X)
            
        Returns:
            Optional[str]: Found captions text or None
        """
        file_ids = []
        try:
            # Upload all available files
            files_info = []
            if docx_path and os.path.exists(docx_path):
                docx_obj = self._upload_file(docx_path)
                if docx_obj:
                    file_ids.append(docx_obj.id)
                    files_info.append("DOCX document")
            
            if pdf_path and os.path.exists(pdf_path):
                pdf_obj = self._upload_file(pdf_path)
                if pdf_obj:
                    file_ids.append(pdf_obj.id)
                    files_info.append("PDF document")

            if not file_ids:
                logger.error("No valid files could be uploaded")
                return None

            try:
                # Create message with enhanced prompt that mentions all available files
                files_desc = " and ".join(files_info)
                message_content = (
                    f"Please locate all figure captions in the provided {files_desc}. "
                    f"I expect {expected_figure_count} main figures, numbered from Figure 1 to Figure {expected_figure_count}. "
                    f"If multiple documents are provided, they should contain the same information - "
                    f"you can use them to cross-validate the extracted captions.\n\n"
                    f"{get_locate_captions_prompt()}"
                )

                thread = self.client.beta.threads.create(
                    messages=[{
                        "role": "user",
                        "content": message_content,
                        "attachments": [
                            {
                                "file_id": file_id,
                                "tools": [{"type": "file_search"}]
                            } for file_id in file_ids
                        ]
                    }]
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
                    return messages.data[0].content[0].text.value
                return None

            finally:
                # Clean up - only try to delete if we have file_ids
                for file_id in file_ids:
                    try:
                        self.client.files.delete(file_id)
                        logger.debug(f"Deleted file {file_id}")
                    except Exception as del_err:
                        logger.warning(f"Error deleting file {file_id}: {del_err}")

        except Exception as e:
            logger.error(f"Error locating captions: {str(e)}")
            return None

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

    def _extract_individual_captions(self, all_captions: str, expected_figure_count: int) -> Dict[str, str]:
        """Extract individual figure captions using AI."""
        try:
            # Create thread for extraction
            message_content = (
                f"The text provided contains figure captions from a scientific manuscript. "
                f"Please extract and structure individual captions for {expected_figure_count} main figures, "
                f"numbered from Figure 1 to Figure {expected_figure_count}. "
                f"Return them in a JSON format where keys are figure labels and values are their captions.\n\n"
                f"{get_extract_captions_prompt(all_captions, expected_figure_count)}"
            )

            thread = self.client.beta.threads.create(
                messages=[{
                    "role": "user",
                    "content": message_content
                }]
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
                return self._parse_response(messages.data[0].content[0].text.value)
            return {}

        except Exception as e:
            logger.error(f"Error extracting individual captions: {str(e)}")
            return {}

    def extract_captions(self, docx_path: str, zip_structure, expected_figure_count: int) -> ZipStructure:
        """Extract figure captions from document."""
        try:
            # Step 1: Get all captions using both DOCX and PDF if available
            logger.info(f"Starting caption extraction for {len(zip_structure.figures)} figures")
            pdf_path = getattr(zip_structure, '_full_pdf', None)
            all_captions = self._locate_figure_captions(
                docx_path, 
                pdf_path=pdf_path, 
                expected_figure_count=expected_figure_count
            )
            
            if not all_captions:
                logger.error("Failed to locate figure captions section")
                return zip_structure

            # Store raw captions
            zip_structure.all_captions_extracted = all_captions
            logger.info(f"Successfully located captions section ({len(all_captions)} characters)")

            # Step 2: Extract individual captions
            captions = self._extract_individual_captions(all_captions, expected_figure_count)
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

    def _upload_file(self, file_path: str) -> Optional[openai.types.FileObject]:
        """Upload a file to OpenAI API."""
        try:
            with open(file_path, "rb") as file:
                file_object = self.client.files.create(
                    file=file,
                    purpose="assistants"
                )
                logger.info(f"Uploaded file {file_object.id}")
                return file_object
        except Exception as e:
            logger.error(f"Error uploading file: {str(e)}")
            return None
