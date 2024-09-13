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

class FigureCaptionExtractorGpt(FigureCaptionExtractor):
    """
    A class to extract figure captions using OpenAI's GPT model and an assistant.
    """

    def __init__(self, config: Dict):
        """
        Initialize the FigureCaptionExtractorGpt instance.

        Args:
            config (Dict): Configuration dictionary for OpenAI API.
        """
        self.config = config
        self.client = openai.OpenAI(api_key=self.config['api_key'])
        self.assistant = self._setup_assistant()

    def _setup_assistant(self) -> Assistant:
        """
        Set up or retrieve the OpenAI assistant for caption extraction.

        Returns:
            Assistant: The OpenAI assistant object.
        """
        assistant_id = self.config.get("caption_extraction_assistant_id")
        self.prompt = get_extract_captions_prompt("")

        if assistant_id:
            # Update existing assistant
            return self.client.beta.assistants.update(
                assistant_id,
                model=self.config['model'],
                instructions=self.prompt
            )
        else:
            # Create a new assistant if ID is not provided
            return self.client.beta.assistants.create(
                name="Figure Caption Extractor",
                instructions=self.prompt,
                model=self.config['model']
            )

    def _upload_file(self, file_path: str) -> FileObject:
        """
        Upload a file to the assistant.

        Args:
            file_path (str): The path to the file to be uploaded.

        Returns:
            FileObject: The uploaded file object.
        """
        with open(file_path, "rb") as file:        
            file_object = self.client.files.create(
                file=file,
                purpose="assistants"
            )
        return file_object
    
    def _prepare_query(self, file_path: str) -> Thread:
        """
        Prepare the query to be sent to the assistant.

        Args:
            user_prompt (str): The user prompt to be processed by the assistant.
            file_path (str): The file path to the docx or pdf file.

        Returns:
            Thread: The thread containing the user prompt and file.
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

        Args:
            docx_path (str): Path to the DOCX file.
            zip_structure (ZipStructure): The current ZIP structure.

        Returns:
            ZipStructure: Updated ZIP structure with extracted captions.
        """
        try:
            print(f"Debug: Processing file: {docx_path}")
            
            # Prepare the query
            thread, file_object = self._prepare_query(docx_path)

            # Run the assistant
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
                    print("Debug: Failed to extract captions from GPT's response")
                    return zip_structure
                
                updated_structure = self._update_zip_structure(zip_structure, captions)
                print(f"Debug: Updated ZIP structure: {updated_structure}")
            else:
                print(f"Debug: Assistant run failed with status: {run.status}")
                return zip_structure

            # Clean up the file
            self.client.files.delete(file_object.id)
            
            return updated_structure
        
        except Exception as e:
            print(f"Error in caption extraction: {str(e)}")
            return zip_structure

    def _parse_response(self, response_text: str) -> Dict[str, str]:
        """
        Parse the response from GPT to extract figure captions.

        Args:
            response_text (str): The response text from GPT.

        Returns:
            Dict[str, str]: A dictionary of figure labels and their captions.
        """
        # First, try to extract JSON from code block
        json_match = re.search(r'```json\n(.*?)```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                print("Failed to parse JSON from code block. Attempting to parse entire response.")
        
        # If no code block or parsing failed, try to parse the entire response
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            print("Failed to parse JSON from GPT's response. Attempting regex parsing.")
            return self._parse_figure_captions(response_text)

    def _parse_figure_captions(self, text: str) -> Dict[str, str]:
        """
        Parse figure captions using regex if JSON parsing fails.

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

    def _update_zip_structure(self, zip_structure: ZipStructure, captions: Dict[str, str]) -> ZipStructure:
        """
        Update the ZipStructure with extracted captions.

        Args:
            zip_structure (ZipStructure): The current ZIP structure.
            captions (Dict[str, str]): Dictionary of figure labels and their captions.

        Returns:
            ZipStructure: Updated ZIP structure.
        """
        for figure in zip_structure.figures:
            if figure.figure_label in captions:
                figure.figure_caption = captions[figure.figure_label]
            elif figure.figure_caption == "TO BE ADDED IN LATER STEP":
                figure.figure_caption = "Figure caption not found."
        return zip_structure