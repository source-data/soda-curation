from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
import json
import logging

logger = logging.getLogger(__name__)

@dataclass
class Figure:
    """
    Represents a figure in the manuscript.

    Attributes:
        figure_label (str): The label of the figure (e.g., "Figure 1", "EV1").
        img_files (List[str]): List of image file paths associated with the figure.
        sd_files (List[str]): List of source data file paths associated with the figure.
        figure_caption (str): The caption of the figure. Defaults to an empty string.
        figure_panels (List[str]): List of panel descriptions for the figure. Defaults to an empty tuple.
    """
    figure_label: str
    img_files: List[str]
    sd_files: List[str]
    figure_caption: str = ""
    figure_panels: List[str] = ()

@dataclass
class ZipStructure:
    """
    Represents the structure of a ZIP file containing manuscript data from eJP.

    Attributes:
        manuscript_id (str): The identifier of the manuscript.
        xml (str): Path to the XML file.
        docx (str): Path to the DOCX file.
        pdf (str): Path to the PDF file.
        appendix (List[str]): List of paths to appendix files.
        figures (List[Figure]): List of Figure objects representing the figures in the manuscript.
    """
    manuscript_id: str
    xml: str
    docx: str
    pdf: str
    appendix: List[str]
    figures: List[Figure]

class CustomJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for ZipStructure and Figure objects.

    This encoder extends the default JSONEncoder to handle ZipStructure and Figure objects,
    converting them to dictionaries for JSON serialization.
    """
    def default(self, obj):
        """
        Converts ZipStructure and Figure objects to dictionaries.

        Args:
            obj: The object to be serialized.

        Returns:
            dict: A dictionary representation of the object if it's a ZipStructure or Figure instance.
            Any: The default serialization for other types.
        """
        if isinstance(obj, (Figure, ZipStructure)):
            return asdict(obj)
        return super().default(obj)

class StructureZipFile(ABC):
    """
    Abstract base class for processing ZIP file structures.

    This class defines the interface for classes that process ZIP file structures
    and convert them into ZipStructure objects.
    """
    @abstractmethod
    def process_zip_structure(self, file_list: List[str]) -> ZipStructure:
        """
        Process the structure of a ZIP file based on its file list.

        This method should be implemented by subclasses to define how the ZIP structure
        is processed and converted into a ZipStructure object.

        Args:
            file_list (List[str]): A list of file paths in the ZIP archive.

        Returns:
            ZipStructure: A ZipStructure object representing the processed ZIP structure.
        """
        pass

    def _json_to_zip_structure(self, json_str: str) -> ZipStructure:
        """
        Convert a JSON string to a ZipStructure object.

        This method parses a JSON string representation of a ZIP structure and
        creates a corresponding ZipStructure object.

        Args:
            json_str (str): A JSON string representing the ZIP structure.

        Returns:
            ZipStructure: A ZipStructure object created from the JSON data.
            None: If there's an error in parsing or the JSON is invalid.

        Raises:
            json.JSONDecodeError: If the JSON string is invalid.
            KeyError: If required keys are missing in the JSON data.
            Exception: For any other unexpected errors during parsing.
        """
        try:
            data = json.loads(json_str)
            required_fields = ['manuscript_id', 'xml', 'docx', 'pdf', 'appendix', 'figures']
            if not all(field in data for field in required_fields):
                logger.error("Missing required fields in JSON response")
                return None

            figures = []
            for fig in data.get('figures', []):
                try:
                    figures.append(Figure(
                        figure_label=fig['figure_label'],
                        img_files=fig['img_files'],
                        sd_files=fig['sd_files'],
                        figure_caption=fig.get('figure_caption', ''),
                        figure_panels=fig.get('figure_panels', [])
                    ))
                except KeyError as e:
                    logger.error(f"Missing key in figure data: {str(e)}")
                    return None
            
            appendix = data.get('appendix', [])
            if appendix is None:
                appendix = []
            elif isinstance(appendix, str):
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
            logger.error("Invalid JSON response from AI")
            return None
        except KeyError as e:
            logger.error(f"Missing key in JSON response: {str(e)}")
            return None
        except Exception as e:
            logger.exception(f"Error in parsing AI response: {str(e)}")
            return None
