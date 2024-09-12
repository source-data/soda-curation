from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
import json

@dataclass
class Figure:
    figure_label: str
    img_files: List[str]
    sd_files: List[str]
    figure_caption: str = ""
    figure_panels: List[str] = ()

@dataclass
class ZipStructure:
    manuscript_id: str
    xml: str
    docx: str
    pdf: str
    appendix: List[str]
    figures: List[Figure]

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (Figure, ZipStructure)):
            return asdict(obj)
        return super().default(obj)

class StructureZipFile(ABC):
    @abstractmethod
    def process_zip_structure(self, file_list: List[str]) -> ZipStructure:
        pass

    def _json_to_zip_structure(self, json_str: str) -> ZipStructure:
        try:
            data = json.loads(json_str)
            figures = [Figure(**fig) for fig in data.get('figures', [])]
            
            appendix = data.get('appendix', [])
            if isinstance(appendix, str):
                appendix = [appendix]
            
            return ZipStructure(
                manuscript_id=data.get('manuscript_id', ''),
                xml=data.get('xml', ''),
                docx=data.get('docx', ''),
                pdf=data.get('pdf', ''),
                appendix=appendix,
                figures=figures
            )
        except json.JSONDecodeError:
            print("Error: Invalid JSON response from AI")
            return None
        except KeyError as e:
            print(f"Error: Missing key in JSON response: {str(e)}")
            return None
        except Exception as e:
            print(f"Error in parsing AI response: {str(e)}")
            return None
