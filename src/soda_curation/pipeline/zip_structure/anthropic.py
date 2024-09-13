from anthropic import Anthropic
from typing import List, Dict, Union
from .base import StructureZipFile, ZipStructure, Figure
from .prompts import get_structure_zip_prompt

class StructureZipFileClaude(StructureZipFile):
    def __init__(self, config: Dict):
        self.anthropic = Anthropic(api_key=config['api_key'])
        self.config = config

    def process_zip_structure(self, file_list: List[str]) -> ZipStructure:
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
            print(f"Debug - AI response: {json_str}")  # Debug print
            return self._json_to_zip_structure(json_str)
        except Exception as e:
            print(f"Error in AI processing: {str(e)}")
            return None

    def _extract_json(self, response: Union[str, List]) -> str:
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
            raise ValueError("No valid JSON object found in the response")
