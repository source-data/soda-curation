"""
This module defines the base classes and data structures for representing and processing
the structure of ZIP files containing manuscript data.
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def full_path(extract_dir: str, file_path: str) -> str:
    return os.path.join(extract_dir, file_path)


@dataclass
class Panel:
    """
    Represents a panel within a figure.

    Attributes:
        panel_label (str): The label of the panel (e.g., "A", "B", "C").
        panel_caption (str): The caption specific to this panel.
        panel_bbox (List[float]): Bounding box coordinates of the panel [x1, y1, x2, y2].
        confidence (float): Confidence of the object detection algorithm.
        ai_response (Optional[str]): The raw AI response for this panel.
        sd_files (List[str]): Source data files for the panel.
    """

    panel_label: str
    panel_caption: str
    panel_bbox: List[float] = field(default_factory=list)
    confidence: float = 0.0
    ai_response: Optional[str] = None
    sd_files: List[str] = field(default_factory=list)


@dataclass
class Figure:
    """
    Represents a figure in the manuscript.

    Attributes:
        figure_label (str): The label of the figure (e.g., "Figure 1", "EV1").
        img_files (List[str]): List of image file paths associated with the figure.
        sd_files (List[str]): List of source data file paths associated with the figure.
        figure_caption (str): The caption of the figure.
        panels (List[Panel]): List of Panel objects representing individual panels.
        duplicated_panels (str): Flag indicating if panels are duplicated.
        ai_response_panel_source_assign (Optional[str]): AI response for panel source assignment.
        possible_hallucination (bool): Flag indicating if caption might be hallucinated.
        unassigned_sd_files (List[str]): Source data files not assigned to specific panels.
        _full_img_files (List[str]): Full paths to image files.
        _full_sd_files (List[str]): Full paths to source data files.
    """

    figure_label: str
    img_files: List[str]
    sd_files: List[str]
    panels: List[Panel] = field(default_factory=list)
    unassigned_sd_files: List[str] = field(default_factory=list)
    _full_img_files: List[str] = field(default_factory=list)
    _full_sd_files: List[str] = field(default_factory=list)
    duplicated_panels: str = "false"
    ai_response_panel_source_assign: Optional[str] = None
    figure_caption: str = ""
    caption_title: str = ""  # New field for the figure caption title


@dataclass
class TokenUsage:
    """Represents token usage for a specific AI operation."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0


@dataclass
class ProcessingCost:
    """Tracks token usage and costs across different processing steps."""

    extract_sections: TokenUsage = field(default_factory=TokenUsage)
    extract_individual_captions: TokenUsage = field(default_factory=TokenUsage)
    assign_panel_source: TokenUsage = field(default_factory=TokenUsage)
    match_caption_panel: TokenUsage = field(default_factory=TokenUsage)
    extract_data_sources: TokenUsage = field(default_factory=TokenUsage)
    total: TokenUsage = field(default_factory=TokenUsage)


@dataclass
class ZipStructure:
    """
    Represents the structure of a ZIP file containing manuscript data.

    Attributes:
        manuscript_id (str): The identifier of the manuscript.
        xml (str): Path to the XML file.
        docx (str): Path to the DOCX file.
        pdf (str): Path to the PDF file.
        appendix (List[str]): List of paths to appendix files.
        figures (List[Figure]): List of Figure objects.
        errors (List[str]): List of errors encountered during processing.
        ai_response (Optional[str]): The raw AI response for figure extraction.
        non_associated_sd_files (List[str]): List of non-associated source data files.
        _full_docx (str): Full path to DOCX file.
        _full_pdf (str): Full path to PDF file.
        _full_appendix (List[str]): Full paths to appendix files.
    """

    appendix: List[str] = field(default_factory=list)
    figures: List[Figure] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    non_associated_sd_files: List[str] = field(default_factory=list)
    _full_appendix: List[str] = field(default_factory=list)
    ai_config: Dict[str, Any] = field(default_factory=dict)
    data_availability: Dict = field(default_factory=dict)
    manuscript_id: str = ""
    xml: str = ""
    docx: str = ""
    pdf: str = ""
    ai_response_locate_captions: Optional[str] = None
    ai_response_extract_individual_captions: Optional[str] = ""
    cost: ProcessingCost = field(default_factory=ProcessingCost)
    _full_docx: str = ""
    _full_pdf: str = ""
    ai_provider: str = ""

    def __post_init__(self):
        """Initialize any attributes that might be missing."""
        if not hasattr(self, "_full_appendix"):
            self._full_appendix = []

    def update_total_cost(self):
        """Update total cost by summing all processing steps."""
        total = self.cost.total

        # Reset total values first
        total.prompt_tokens = 0
        total.completion_tokens = 0
        total.total_tokens = 0
        total.cost = 0.0

        # Add up each component
        for component in [
            self.cost.extract_sections,
            self.cost.extract_individual_captions,
            self.cost.assign_panel_source,
            self.cost.match_caption_panel,
            self.cost.extract_data_sources,
        ]:
            total.prompt_tokens += component.prompt_tokens
            total.completion_tokens += component.completion_tokens
            total.total_tokens += component.total_tokens
            total.cost += component.cost

        # Verify total_tokens equals sum of prompt and completion
        total.total_tokens = total.prompt_tokens + total.completion_tokens


class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for ZipStructure and related objects."""

    def default(self, obj):
        """Convert dataclass objects to dictionaries, excluding private fields."""
        if isinstance(obj, (ZipStructure, Figure, Panel, ProcessingCost, TokenUsage)):
            # Get all non-private attributes
            dict_obj = {}
            for k, v in vars(obj).items():
                if k.startswith("_"):
                    continue

                # Handle special cases that might cause circular references
                if k == "figures" and isinstance(obj, ZipStructure):
                    dict_obj[k] = [self.default(fig) for fig in v]
                elif k == "panels" and isinstance(obj, Figure):
                    dict_obj[k] = [self.default(panel) for panel in v]
                elif k == "sd_files" and isinstance(obj, Panel):
                    # Preserve the full paths for source data files
                    dict_obj[k] = [str(f) for f in v] if v else []
                else:
                    dict_obj[k] = v

            # Remove any None values and empty collections
            return {
                k: v
                for k, v in dict_obj.items()
                if v is not None and v != {} and v != []
            }

        return super().default(obj)

    def serialize_dataclass(self, obj):
        """Serialize a dataclass object, excluding private fields."""
        if isinstance(obj, (ZipStructure, Figure)):
            return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
        elif isinstance(obj, (ProcessingCost, TokenUsage)):
            return {k: v for k, v in obj.__dict__.items()}
        return super().default(obj)


class XMLStructureExtractor(ABC):
    """Abstract base class for processing ZIP file structures."""

    @abstractmethod
    def process_zip_structure(self, file_list: List[str]) -> ZipStructure:
        """Process the structure of a ZIP file based on its file list."""
        pass

    def _json_to_zip_structure(self, json_str: str) -> Optional[ZipStructure]:
        """Convert a JSON string to a ZipStructure object."""
        try:
            data = json.loads(json_str)
            required_fields = [
                "manuscript_id",
                "xml",
                "docx",
                "pdf",
                "appendix",
                "figures",
            ]

            if not all(field in data for field in required_fields):
                logger.error("Missing required fields in JSON response")
                return None

            figures = []
            for fig in data.get("figures", []):
                try:
                    figures.append(
                        Figure(
                            figure_label=fig["figure_label"],
                            img_files=fig["img_files"],
                            sd_files=fig["sd_files"],
                            figure_caption=fig.get("figure_caption", ""),
                            panels=fig.get("panels", []),
                            ai_response_locate_captions=fig.get(
                                "ai_response_locate_captions"
                            ),
                            ai_response_extract_captions=data.get(
                                "ai_response_extract_captions"
                            ),
                            rouge_l_score=fig.get("rouge_l_score", 0.0),
                        )
                    )

                except KeyError as e:
                    logger.error(f"Missing key in figure data: {str(e)}")
                    return None

            appendix = data.get("appendix", [])
            if appendix is None:
                appendix = []
            elif isinstance(appendix, str):
                appendix = [appendix]

            return ZipStructure(
                manuscript_id=data.get("manuscript_id", ""),
                xml=data.get("xml", ""),
                docx=data.get("docx", ""),
                pdf=data.get("pdf", ""),
                appendix=appendix,
                figures=figures,
                ai_response_locate_captions=data.get("ai_response_locate_captions"),
                ai_response_extract_captions=data.get("ai_response_extract_captions"),
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
