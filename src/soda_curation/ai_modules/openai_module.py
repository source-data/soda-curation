import openai
from typing import List, Dict
from .general import StructureZipFile, ZipStructure

class StructureZipFileGPT(StructureZipFile):
    def __init__(self, config: Dict):
        openai.api_key = config['api_key']
        self.config = config

    def process_zip_structure(self, file_list: List[str]) -> ZipStructure:
        prompt = f"Given the following list of files from a ZIP archive, parse the structure and return a JSON object following the ZipStructure format:\n\n{file_list}\n\nEnsure that the JSON object includes 'manuscript_id', 'xml', 'docx', 'pdf', 'appendix', and 'figures' fields, even if some are empty."
        
        try:
            response = openai.ChatCompletion.create(
                model=self.config['model'],
                messages=[
                    {"role": "system", "content": "You are an AI assistant that parses ZIP file structures and returns them in a specific JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config['temperature'],
                max_tokens=self.config['max_tokens'],
                top_p=self.config['top_p'],
                frequency_penalty=self.config['frequency_penalty'],
                presence_penalty=self.config['presence_penalty']
            )

            json_response = response.choices[0].message['content']
            print(f"Debug - AI response: {json_response}")  # Debug print
            return self._json_to_zip_structure(json_response)
        except Exception as e:
            print(f"Error in AI processing: {str(e)}")
            return None

