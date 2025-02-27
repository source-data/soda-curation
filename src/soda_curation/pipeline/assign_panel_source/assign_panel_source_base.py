import logging
import os
import re
import zipfile
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel

from ..manuscript_structure.manuscript_structure import Figure, Panel, ZipStructure
from ..prompt_handler import PromptHandler


class AsignedFiles(BaseModel):
    """Model for a list of panels."""

    panel_label: str
    panel_sd_files: List[str]


class AsignedFilesList(BaseModel):
    """Model for a list of assigned files."""

    assigned_files: List[AsignedFiles]
    not_assigned_files: List[str]


logger = logging.getLogger(__name__)


class PanelSourceAssigner(ABC):
    def __init__(self, config: Dict[str, Any], prompt_handler: PromptHandler):
        self.extraction_dir = config["extraction_dir"]
        self.config = config
        self.prompt_handler = prompt_handler
        self._validate_config()
        logger.info("PanelSourceAssigner initialized successfully")

    @staticmethod
    def normalize_panel_label(label: str) -> str:
        """Normalize panel labels by removing figure numbers and standardizing format."""

        match = re.search(r"[A-Z]\d*$", label.upper())
        if match:
            return match.group(0).rstrip("0123456789")
        return label.upper()

    @abstractmethod
    def _validate_config(self) -> None:
        """
        Validate the configuration specific to each implementation.

        Raises:
            ValueError: If configuration is invalid
        """
        pass

    def assign_panel_source(self, zip_structure: ZipStructure) -> List[Figure]:
        logger.info("Starting panel source assignment process")
        self.zip_structure = zip_structure
        for figure in self.zip_structure.figures:
            logger.info(f"Assigning data source to figure: {figure.figure_label}")
            self._assign_to_figure(figure)
        return self.zip_structure.figures

    def _assign_to_figure(self, figure: Figure) -> None:
        """Process single figure."""
        # First normalize and deduplicate existing panels
        normalized_panels = {}
        for panel in figure.panels:
            norm_label = self.normalize_panel_label(panel.panel_label)
            if norm_label in normalized_panels:
                # If we already have this panel, keep the one with higher confidence
                existing = normalized_panels[norm_label]
                if panel.confidence > existing.confidence:
                    normalized_panels[norm_label] = panel
            else:
                normalized_panels[norm_label] = panel

            # Update the panel label to normalized form
            panel.panel_label = norm_label

        # Update figure with deduplicated panels
        figure.panels = list(normalized_panels.values())

        # Initialize empty sd_files for all panels if not present
        for panel in figure.panels:
            if not hasattr(panel, "sd_files"):
                panel.sd_files = []

        sd_files = [os.path.join(self.extraction_dir, f) for f in figure.sd_files]

        if not sd_files:
            # If there are no source data files, return early with empty assignments
            assigned_files_list = AsignedFilesList(
                assigned_files=[], not_assigned_files=[]
            )
        else:
            # Get file list from zip files
            extracted_files = self._get_zip_contents(sd_files)

            # Get panel labels (now normalized)
            panel_labels = [panel.panel_label for panel in figure.panels]

            # Get the prompt
            prompt = self._get_assign_panel_source_prompt(
                figure.figure_label, panel_labels, extracted_files
            )

            # Call AI service
            assigned_files_list = self.call_ai_service(prompt, extracted_files)

        # Convert assigned files to Panel objects and normalize their labels
        panels = self.parse_assigned_files_to_panels(assigned_files_list)
        for p in panels:
            p.panel_label = self.normalize_panel_label(p.panel_label)

        # Update figure with assignments
        self._update_figure_with_assignments(
            figure, panels, assigned_files_list.not_assigned_files
        )

    @abstractmethod
    def call_ai_service(
        self, prompt: str, allowed_files: List[str]
    ) -> AsignedFilesList:
        """Abstract method to call AI service."""
        pass

    def _get_zip_contents(self, sd_files: List[str]) -> List[str]:
        """Extract file list from zip files and return paths in a specific format."""
        extracted_files = []
        windows_system_files = {"Thumbs.db", "desktop.ini"}

        for file_path in sd_files:
            if file_path.endswith(".zip"):
                with zipfile.ZipFile(file_path, "r") as zip_ref:
                    for file_info in zip_ref.infolist():
                        # Skip directories, macOS metadata, and Windows system files
                        filename = file_info.filename
                        if (
                            filename.endswith("/")
                            or "__MACOSX" in filename
                            or ".DS_Store" in filename
                            or any(wsf in filename for wsf in windows_system_files)
                        ):
                            continue

                        # Normalize Unicode characters
                        try:
                            filename = filename.encode("cp1252").decode("utf-8")
                        except (UnicodeEncodeError, UnicodeDecodeError):
                            # If encoding fails, keep original filename
                            pass

                        # Construct the path format
                        zip_relative_path = os.path.relpath(
                            file_path, self.extraction_dir
                        )
                        extracted_files.append(f"{zip_relative_path}:{filename}")
            else:
                # Non-zip files are added directly
                extracted_files.append(os.path.relpath(file_path, self.extraction_dir))

        return extracted_files

    def _get_assign_panel_source_prompt(
        self, figure_label: str, panel_labels: List[str], file_list: List[str]
    ) -> str:
        """Get the prompt using the prompt handler."""
        variables = {
            "figure_label": figure_label,
            "panel_labels": ", ".join(panel_labels),
            "file_list": "\n".join(file_list),
        }

        prompts = self.prompt_handler.get_prompt("assign_panel_source", variables)
        return prompts["user"]  # We only need the user prompt here

    def parse_assigned_files_to_panels(
        self, assigned_files_list: AsignedFilesList
    ) -> List[Panel]:
        """Parse the assigned files list into Panel objects."""
        panels = []
        for assigned_file in assigned_files_list.assigned_files:
            panel = Panel(
                panel_label=assigned_file.panel_label,
                panel_caption="",  # Assuming caption is not provided in this context
                sd_files=assigned_file.panel_sd_files
                or [],  # Ensure empty list if None
            )
            panels.append(panel)
        return panels

    def _update_figure_with_assignments(
        self, figure: Figure, panels: List[Panel], not_assigned_files: List[str]
    ) -> None:
        """Update figure with source data assignments."""
        try:
            # Create a dictionary to store panels by their normalized labels
            normalized_panels = {}

            # First process existing panels
            for panel in figure.panels:
                norm_label = self.normalize_panel_label(panel.panel_label)
                panel.panel_label = norm_label  # Update the label to normalized form
                if not hasattr(panel, "sd_files"):
                    panel.sd_files = []
                normalized_panels[norm_label] = panel

            # Process new panels and update/add them
            for new_panel in panels:
                norm_label = self.normalize_panel_label(new_panel.panel_label)
                new_panel.panel_label = norm_label

                if norm_label in normalized_panels:
                    # Update existing panel's sd_files
                    normalized_panels[norm_label].sd_files = new_panel.sd_files or []
                else:
                    # Add new panel
                    new_panel.sd_files = new_panel.sd_files or []
                    normalized_panels[norm_label] = new_panel

            # Update figure's panels with deduplicated list
            figure.panels = list(normalized_panels.values())

            # Update unassigned files
            figure.unassigned_sd_files = not_assigned_files

            logger.info(
                f"Updated figure {figure.figure_label} with {len(panels)} panel assignments"
            )

        except Exception as e:
            logger.error(
                f"Error updating assignments for {figure.figure_label}: {str(e)}"
            )
            raise

    @staticmethod
    def filter_files(
        assigned_files: List[AsignedFiles],
        not_assigned_files: List[str],
        allowed_files: List[str],
    ) -> Tuple[List[AsignedFiles], List[str]]:
        """Remove any files that are not in allowed_files."""
        filtered_assigned_files = []

        for af in assigned_files:
            # Filter valid files for this panel
            valid_files = [file for file in af.panel_sd_files if file in allowed_files]
            # Only keep panels that have at least one valid file
            if valid_files:
                filtered_assigned_files.append(
                    AsignedFiles(panel_label=af.panel_label, panel_sd_files=valid_files)
                )

        filtered_not_assigned_files = [
            file for file in not_assigned_files if file in allowed_files
        ]

        return filtered_assigned_files, filtered_not_assigned_files
