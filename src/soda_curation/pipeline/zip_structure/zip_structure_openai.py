"""
This module provides functionality for processing ZIP file structures using OpenAI's GPT model.

It includes a class that interacts with the OpenAI API to parse and structure
the contents of a ZIP file containing manuscript data.
"""

import openai
import logging
from typing import List, Dict
from .zip_structure_base import StructureZipFile, ZipStructure
from openai.types.beta.thread import Thread
from .zip_structure_prompts import get_structure_zip_prompt

logger = logging.getLogger(__name__)

class StructureZipFileGPT(StructureZipFile):
    """
    A class to process ZIP file structures using OpenAI's GPT model.

    This class handles the interaction with OpenAI's API to parse and structure
    the contents of a ZIP file.

    Attributes:
        config (Dict): Configuration dictionary for OpenAI API.
        _client (openai.Client): OpenAI API client.
        _assistant (openai.types.beta.assistant.Assistant): OpenAI assistant object.
    """
    def __init__(self, config: Dict):
        """
        Initialize the StructureZipFileGPT instance.

        Args:
            config (Dict): Configuration dictionary for OpenAI API.
        """
        self.config = config
        self._client = openai.Client(api_key=self.config['api_key'])
        self._assistant = self._client.beta.assistants.retrieve(
            config["structure_zip_assistant_id"]
        )
        
        # Update assistant instructions
        prompt = get_structure_zip_prompt(
            file_list="[File list will be provided in each request]",
            custom_instructions=self.config.get('custom_prompt_instructions')
        )
        self._assistant = self._client.beta.assistants.update(
            config["structure_zip_assistant_id"],
            model=self.config['model'],
            temperature=self.config['temperature'],
            top_p=self.config['top_p'],
            instructions=prompt
        )
        logger.info("OpenAI assistant initialized and updated")

    def _prepare_query(self, file_list: List[str]) -> Thread:
        """
        Prepare a query thread for the OpenAI assistant.

        This method creates a new thread with the file list as the initial message.

        Args:
            file_list (List[str]): List of files in the ZIP archive.

        Returns:
            Thread: An OpenAI thread object containing the query.
        """
        thread = self._client.beta.threads.create(
            messages=[
                {
                    "role": "user",
                    "content": f"Process this file list:\n{file_list}",
                }
            ]
        )
        logger.debug(f"Query prepared with file list: {file_list}")
        return thread

    def process_zip_structure(self, file_list: List[str]) -> ZipStructure:
        """
        Process the ZIP file structure using the OpenAI assistant.

        This method sends the file list to the OpenAI assistant, retrieves the response,
        and converts it into a ZipStructure object.

        Args:
            file_list (List[str]): List of files in the ZIP archive.

        Returns:
            ZipStructure: A structured representation of the ZIP contents,
                          or None if processing fails.
        """
        try:
            thread = self._prepare_query(f"""[{", ".join(file_list)}]""")

            run = self._client.beta.threads.runs.create_and_poll(
                thread_id=thread.id,
                assistant_id=self._assistant.id,
            )

            if run.status == 'completed':
                messages = self._client.beta.threads.messages.list(
                    thread_id=thread.id
                )
                result = messages.data[0].content[0].text.value
                logger.debug(f"AI response: {result}")
                return self._json_to_zip_structure(result)
            else:
                logger.error(f"AI processing failed with status: {run.status}")
                return run.status

        except Exception as e:
            logger.exception(f"Error in AI processing: {str(e)}")
            return None
