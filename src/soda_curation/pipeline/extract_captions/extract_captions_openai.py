import json
from typing import Dict
import openai
from docx import Document
from .extract_captions_base import FigureCaptionExtractor
from .extract_captions_prompts import get_extract_captions_prompt
from ..zip_structure.zip_structure_base import ZipStructure

class FigureCaptionExtractorGpt(FigureCaptionExtractor):
    """
    A class to extract figure captions using OpenAI's GPT model.
    """

    def __init__(self, config: Dict):
        """
        Initialize the FigureCaptionExtractorGpt instance.

        Args:
            config (Dict): Configuration dictionary for OpenAI API.
        """
        self.config = config
        self.client = openai.Client(api_key=self.config['api_key'])

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
            doc = Document(docx_path)
            file_content = "\n".join([para.text for para in doc.paragraphs])
            
            prompt = get_extract_captions_prompt(file_content)
            
            response = self.client.chat.completions.create(
                model=self.config['model'],
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that extracts figure captions from scientific manuscripts."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config['temperature'],
                max_tokens=self.config.get('max_tokens', 1000),
                top_p=self.config['top_p']
            )
            
            captions = json.loads(response.choices[0].message.content)
            return self._update_zip_structure(zip_structure, captions)
        
        except Exception as e:
            print(f"Error in caption extraction: {str(e)}")
            return zip_structure
