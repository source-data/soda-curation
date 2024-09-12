from anthropic import Anthropic
from typing import List, Dict
from .general import StructureZipFile, ZipStructure, Figure

class StructureZipFileClaude(StructureZipFile):
    def __init__(self, config: Dict):
        self.anthropic = Anthropic(api_key=config['api_key'])
        self.config = config

    def process_zip_structure(self, file_list: List[str]) -> ZipStructure:
        prompt = f"Given the following list of files from a ZIP archive, parse the structure and return a JSON object following the ZipStructure format:\n\n{file_list}\n\nEnsure that the JSON object includes 'manuscript_id', 'xml', 'docx', 'pdf', 'appendix', and 'figures' fields. The 'figures' field should be a list of objects, each with 'figure_label', 'img_files', 'sd_files', 'figure_caption', and 'figure_panels' fields."
        
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
            # Extract the JSON string from the TextBlock
            json_str = json_response[0].text if isinstance(json_response, list) else json_response
            # Remove any leading/trailing whitespace and extract the JSON part
            json_str = json_str.strip().split('```json')[-1].split('```')[0].strip()
            print(f"Debug - AI response: {json_str}")  # Debug print
            return self._json_to_zip_structure(json_str)
        except Exception as e:
            print(f"Error in AI processing: {str(e)}")
            return None
