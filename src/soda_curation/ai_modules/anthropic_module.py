from anthropic import Anthropic
from typing import List, Dict, Union
from .general import StructureZipFile, ZipStructure, Figure

class StructureZipFileClaude(StructureZipFile):
    def __init__(self, config: Dict):
        self.anthropic = Anthropic(api_key=config['api_key'])
        self.config = config

    def process_zip_structure(self, file_list: List[str]) -> ZipStructure:
        prompt = f"""Given the following list of files from a ZIP archive, parse the structure and return ONLY a JSON object following the ZipStructure format. Do not include any explanatory text before or after the JSON.

            File list:
            {file_list}

            Instructions:
            1. The root folder name is the 'manuscript_id'.
            2. The 'xml' file is in the root folder.
            3. The 'docx' file is in a subfolder named 'doc', 'DOC', or 'Doc'.
            4. The 'pdf' file is in a subfolder named 'pdf'.
            5. The 'appendix' is typically a PDF file in the 'suppl_data' folder containing 'appendix' in its name.
            6. For 'figures':
            - Main figure image files are in the 'graphic' folder.
            - Figure labels can be "Figure X" or "EV X" (for extended view figures).
            - Source data ('sd_files') are in the 'suppl_data' folder.
            - Include ALL related files from 'suppl_data' for each figure.
            - Match source data to figures based on number (e.g., "Figure 1" matches "Figure 1.zip", "Dataset 1.xlsx", "Table 1.docx").
            - For EV figures, match "EV1" with "Dataset EV1.xlsx", "Table EV1.docx", etc.

            The JSON object should include 'manuscript_id', 'xml', 'docx', 'pdf', 'appendix', and 'figures' fields. The 'figures' field should be a list of objects, each with 'figure_label', 'img_files', 'sd_files', 'figure_caption', and 'figure_panels' fields. Use null for any missing values. Ensure the output is valid JSON.

            Example input-output:

            Input:
            EMM-2023-18636
            ├── EMM-2023-18636.xml
            ├── doc
            │   └── manuscript.docx
            ├── graphic
            │   ├── Figure 1.eps
            │   └── EV1.eps
            ├── pdf
            │   └── manuscript.pdf
            └── suppl_data
                ├── Appendix.pdf
                ├── Figure 1.zip
                ├── Dataset 1.xlsx
                ├── Table 1.docx
                ├── Dataset EV1.xlsx
                └── Table EV1.docx

            Output:
            """
        prompt += """
            {
            "manuscript_id": "EMM-2023-18636",
            "xml": "EMM-2023-18636.xml",
            "docx": "doc/manuscript.docx",
            "pdf": "pdf/manuscript.pdf",
            "appendix": ["suppl_data/Appendix.pdf"],
            "figures": [
                {
                "figure_label": "Figure 1",
                "img_files": ["graphic/Figure 1.eps"],
                "sd_files": [
                    "suppl_data/Figure 1.zip",
                    "suppl_data/Dataset 1.xlsx",
                    "suppl_data/Table 1.docx"
                ],
                "figure_caption": "TO BE ADDED IN LATER STEP",
                "figure_panels": []
                },
                {
                "figure_label": "Figure EV1",
                "img_files": ["graphic/EV1.eps"],
                "sd_files": [
                    "suppl_data/Dataset EV1.xlsx",
                    "suppl_data/Table EV1.docx"
                ],
                "figure_caption": "TO BE ADDED IN LATER STEP",
                "figure_panels": []
                }
            ]
            }

            Now, parse the provided file list and return the JSON object:"""
        
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
            # Extract the JSON string from the response
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
