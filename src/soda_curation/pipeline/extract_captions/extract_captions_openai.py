"""
This module provides functionality for extracting figure captions from scientific documents
using the OpenAI API (GPT model).

It includes a class that interacts with the OpenAI API to process document content and extract
figure captions, which are then integrated into a ZipStructure object.
"""

import json
import logging
import os
import re
import time
from typing import Dict

import openai
from openai.types.beta.assistant import Assistant
from openai.types.beta.thread import Thread
from openai.types.file_object import FileObject

from ..manuscript_structure.manuscript_structure import ZipStructure
from .extract_captions_base import FigureCaptionExtractor
from .extract_captions_prompts import get_extract_captions_prompt

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
        self.client = openai.OpenAI(api_key=self.config["api_key"])
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
        
        # We'll use a placeholder for expected_figure_count
        self.prompt = get_extract_captions_prompt("", "{expected_figure_count}")

        if assistant_id:
            return self.client.beta.assistants.update(
                assistant_id, model=self.config["model"], instructions=self.prompt
            )
        else:
            return self.client.beta.assistants.create(
                name="Figure Caption Extractor",
                instructions=self.prompt,
                model=self.config["model"],
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
            file_object = self.client.files.create(file=file, purpose="assistants")
        return file_object

    def _prepare_query(self, file_path: str, expected_figure_count: int) -> Thread:
        """
        Prepare the query to be sent to the OpenAI assistant.

        This method uploads the file and creates a new thread with the initial message.

        Args:
            file_path (str): The file path to the DOCX or PDF file.
            expected_figure_count (int): The expected number of figures in the document.

        Returns:
            Thread: The thread containing the user prompt and file.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            openai.OpenAIError: If there's an issue creating the thread or uploading the file.
        """
        file_on_client = self._upload_file(file_path)
        prompt = get_extract_captions_prompt("", expected_figure_count)  # Pass empty string as file_content will be uploaded
        thread = self.client.beta.threads.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "attachments": [
                        {
                            "file_id": file_on_client.id,
                            "tools": [{"type": "file_search"}],
                        }
                    ],
                }
            ]
        )
        return thread, file_on_client

    def extract_captions(self, docx_path: str, zip_structure: ZipStructure, expected_figure_count: int) -> ZipStructure:
        """
        Extract figure captions from the given DOCX file using OpenAI's GPT model.

        This method processes the DOCX file, sends its content to the OpenAI API,
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

            if not os.path.exists(docx_path):
                raise FileNotFoundError(f"File not found: {docx_path}")

            thread, file_object = self._prepare_query(docx_path, expected_figure_count)

            run = self.client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=self.assistant.id,
            )

            while run.status != "completed":
                run = self.client.beta.threads.runs.retrieve(
                    thread_id=thread.id,
                    run_id=run.id
                )
                time.sleep(1)

            messages = self.client.beta.threads.messages.list(thread_id=thread.id)
            
            # Debug logging
            logger.debug(f"Messages data type: {type(messages)}")
            logger.debug(f"Messages content: {messages}")

            if messages.data:
                # Ensure we're getting a string from the message content
                result = messages.data[0].content[0].text if isinstance(messages.data[0].content[0].text, str) else str(messages.data[0].content[0].text)
            else:
                logger.warning("No messages found in the thread")
                result = ""

            logger.debug(f"Result data type: {type(result)}")
            logger.debug(f"Result content: {result[:500]}...")  # Log first 500 characters

            if result:
                captions = self._parse_response(result)

                if not captions:
                    logger.warning("Failed to extract captions from GPT's response")
            else:
                logger.warning("No result to parse")
                captions = {}

            updated_structure = self._update_zip_structure(zip_structure, captions, result)
            logger.info(f"Updated ZIP structure: {updated_structure}")

            self.client.files.delete(file_object.id)

            return updated_structure

        except Exception as e:
            logger.exception(f"Error in caption extraction: {str(e)}")
            return self._update_zip_structure(zip_structure, {}, str(e))

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
        if not isinstance(response_text, str):
            logger.warning(f"Unexpected response type: {type(response_text)}")
            response_text = str(response_text)

        json_match = re.search(r"```json\n(.*?)```", response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                logger.warning(
                    "Failed to parse JSON from code block. Attempting to parse entire response."
                )

        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            logger.warning(
                "Failed to parse JSON from GPT's response. Attempting regex parsing."
            )
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
            caption = match.group(2).replace("\\n", "\n").strip()
            captions[figure_label] = caption
        return captions

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
                figure.figure_caption = captions[figure.figure_label].encode("utf-8").decode("utf-8", "ignore")
            else:
                figure.figure_caption = "Figure caption not found."
        zip_structure.ai_response = ai_response  # Set AI response at ZipStructure level
        return zip_structure

