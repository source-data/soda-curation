from anthropic import Anthropic
import logging
from typing import List, Dict, Union
from .zip_structure_base import StructureZipFile, ZipStructure
from .zip_structure_prompts import get_structure_zip_prompt

logger = logging.getLogger(__name__)

class StructureZipFileClaude(StructureZipFile):
    """
    A class to process ZIP file structures using Anthropic's Claude model.

    This class handles the interaction with Anthropic's API to parse and structure
    the contents of a ZIP file.

    Attributes:
        anthropic (Anthropic): Anthropic API client.
        config (Dict): Configuration dictionary for Anthropic API.
    """
    def __init__(self, config: Dict):
        """
        Initialize the StructureZipFileClaude instance.

        Args:
            config (Dict): Configuration dictionary for Anthropic API.
        """
        self.anthropic = Anthropic(api_key=config['api_key'])
        self.config = config
        logger.info("Anthropic client initialized")

    def process_zip_structure(self, file_list: List[str]) -> ZipStructure:
        """
        Process the ZIP file structure using the Anthropic Claude model.

        This method sends the file list to the Anthropic Claude model, retrieves the response,
        and converts it into a ZipStructure object.

        Args:
            file_list (List[str]): List of files in the ZIP archive.

        Returns:
            ZipStructure: A structured representation of the ZIP contents,
                          or None if processing fails.
        """
        prompt = get_structure_zip_prompt(
            file_list="\n".join(file_list),
            custom_instructions=self.config.get('custom_prompt_instructions')
        )
        
        try:
            response = self.anthropic.messages.create(
                model=self.config['model'],
                max_tokens=self.config['max_tokens_to_sample'],
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config['temperature']
            )

            json_response = response.content
            json_str = self._extract_json(json_response)
            logger.debug(f"AI response: {json_str}")
            return self._json_to_zip_structure(json_str)
        except Exception as e:
            logger.exception(f"Error in AI processing: {str(e)}")
            return None

    def _extract_json(self, response: Union[str, List]) -> str:
        """
        Extract JSON string from the AI model's response.

        This method handles both string and list responses, extracting the JSON
        content from the AI model's output.

        Args:
            response (Union[str, List]): The raw response from the AI model.

        Returns:
            str: Extracted JSON string.

        Raises:
            ValueError: If no valid JSON object is found in the response.
        """
        if isinstance(response, list):
            # Join all elements of the list into a single string
            text = ' '.join(item.text for item in response if hasattr(item, 'text'))
        else:
            text = response

        # Find the first '{' and the last '}'
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            return text[start:end+1]
        else:
            logger.error("No valid JSON object found in the response")
            raise ValueError("No valid JSON object found in the response")
