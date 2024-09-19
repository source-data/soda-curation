"""
This module provides functionality for extracting figure captions from scientific documents
using the OpenAI API (GPT model).

It includes a class that interacts with the OpenAI API to process document content and extract
figure captions, which are then integrated into a ZipStructure object.
"""

import json
import os
from io import BytesIO
from typing import Dict
import openai
from openai.types.beta.assistant import Assistant
from openai.types.beta.thread import Thread
from openai.types.file_object import FileObject
from .extract_captions_base import FigureCaptionExtractor
from .extract_captions_prompts import get_extract_captions_prompt
from ..zip_structure.zip_structure_base import ZipStructure
import re
import logging

logger = logging.getLogger(__name__)

class FigureCaptionExtractorGpt(FigureCaptionExtractor):
    """
    A class to extract figure captions using OpenAI's GPT model and an assistant.

    This class provides methods to interact with the OpenAI API, process DOCX files,
    and extract figure captions using GPT models.

    Attributes:
        config (Dict): Configuration dictionary for OpenAI API.
        client (openai.OpenAI): OpenAI API client.
        assistant (Assistant): OpenAI assistant object for caption extraction.
    """

    def __init__(self, config: Dict):
        """
        Initialize the FigureCaptionExtractorGpt instance.

        Args:
            config (Dict): Configuration dictionary for OpenAI API.

        Raises:
            openai.OpenAIError: If there's an issue initializing the OpenAI client or assistant.
        """
        self.config = config
        self.client = openai.OpenAI(api_key=self.config['api_key'])
        self.assistant = self._setup_assistant()

    def _setup_assistant(self) -> Assistant:
        """
        Set up or retrieve the OpenAI assistant for caption extraction.

        This method either creates a new assistant or updates an existing one
        based on the configuration provided.

        Returns:
            Assistant: The OpenAI assistant object.

        Raises:
            openai.OpenAIError: If there's an issue creating or updating the assistant.
        """
        assistant_id = self.config.get("caption_extraction_assistant_id")
        self.prompt = get_extract_captions_prompt("")

        if assistant_id:
            return self.client.beta.assistants.update(
                assistant_id,
                model=self.config['model'],
                instructions=self.prompt
            )
        else:
            return self.client.beta.assistants.create(
                name="Figure Caption Extractor",
                instructions=self.prompt,
                model=self.config['model']
            )

    def _upload_file(self, file_path: str) -> FileObject:
        """
        Upload a file to the OpenAI API for processing.

        Args:
            file_path (str): The path to the file to be uploaded.

        Returns:
            FileObject: The uploaded file object.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            openai.OpenAIError: If there's an issue uploading the file to OpenAI.
        """
        with open(file_path, "rb") as file:        
            file_object = self.client.files.create(
                file=file,
                purpose="assistants"
            )
        return file_object
    
    def _prepare_query(self, file_path: str) -> Thread:
        """
        Prepare the query to be sent to the OpenAI assistant.

        This method uploads the file and creates a new thread with the initial message.

        Args:
            file_path (str): The file path to the DOCX or PDF file.

        Returns:
            Thread: The thread containing the user prompt and file.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            openai.OpenAIError: If there's an issue creating the thread or uploading the file.
        """
        file_on_client = self._upload_file(file_path)
        thread = self.client.beta.threads.create(
            messages=[
                {
                    "role": "user",
                    "content": self.prompt,
                    "attachments": [
                        {
                            "file_id": file_on_client.id,
                            "tools": [
                                {"type": "file_search"}
                            ]
                        }
                    ]
                }
            ]
        )
        return thread, file_on_client

    def extract_captions(self, docx_path: str, zip_structure: ZipStructure) -> ZipStructure:
        """
        Extract figure captions from the given DOCX file using OpenAI's GPT model.

        This method processes the DOCX file, sends its content to the OpenAI API,
        and updates the ZipStructure with the extracted captions.

        Args:
            docx_path (str): Path to the DOCX file.
            zip_structure (ZipStructure): The current ZIP structure.

        Returns:
            ZipStructure: Updated ZIP structure with extracted captions.
        """
        try:
            logger.info(f"Processing file: {docx_path}")
            
            if not os.path.exists(docx_path):
                raise FileNotFoundError(f"File not found: {docx_path}")

            thread, file_object = self._prepare_query(docx_path)

            run = self.client.beta.threads.runs.create_and_poll(
                thread_id=thread.id,
                assistant_id=self.assistant.id,
            )

            if run.status == 'completed':
                messages = self.client.beta.threads.messages.list(
                    thread_id=thread.id
                )
                result = messages.data[0].content[0].text.value
                captions = self._parse_response(result)
                
                if not captions:
                    logger.warning("Failed to extract captions from GPT's response")
                
                updated_structure = self._update_zip_structure(zip_structure, captions)
                logger.info(f"Updated ZIP structure: {updated_structure}")
            else:
                logger.error(f"Assistant run failed with status: {run.status}")
                return zip_structure

            self.client.files.delete(file_object.id)
            
            return updated_structure
        
        except FileNotFoundError as e:
            logger.error(f"File not found: {str(e)}")
            return zip_structure
        except Exception as e:
            logger.exception(f"Error in caption extraction: {str(e)}")
            return zip_structure

    def _parse_response(self, response_text: str) -> Dict[str, str]:
        """
        Parse the response from GPT to extract figure captions.

        This method attempts to extract a JSON object from the response text and parse it.
        If JSON parsing fails, it falls back to regex-based parsing.

        Args:
            response_text (str): The response text from GPT.

        Returns:
            Dict[str, str]: A dictionary of figure labels and their captions.
        """
        json_match = re.search(r'```json\n(.*?)```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON from code block. Attempting to parse entire response.")
        
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from GPT's response. Attempting regex parsing.")
            return self._parse_figure_captions(response_text)

    def _parse_figure_captions(self, text: str) -> Dict[str, str]:
        """
        Parse figure captions using regex if JSON parsing fails.

        This method uses regular expressions to extract figure labels and captions
        from the text response.

        Args:
            text (str): The text to parse for figure captions.

        Returns:
            Dict[str, str]: A dictionary of figure labels and their captions.
        """
        captions = {}
        matches = re.finditer(r'"(Figure \d+)":\s*"(.*?)"', text, re.DOTALL)
        for match in matches:
            figure_label = match.group(1)
            caption = match.group(2).replace('\\n', '\n').strip()
            captions[figure_label] = caption
        return captions
