import logging
import os
import zipfile
from abc import ABC, abstractmethod
from typing import Any, Dict, List

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
        sd_files = [os.path.join(self.extraction_dir, f) for f in figure.sd_files]
        if not sd_files:
            # If there are no source data files, return early with empty assignments
            assigned_files_list = AsignedFilesList(
                assigned_files=[], not_assigned_files=[]
            )
        else:
            # Get file list from zip files
            extracted_files = self._get_zip_contents(sd_files)

            # Get panel labels
            panel_labels = [panel.panel_label for panel in figure.panels]

            # Get the prompt
            prompt = self._get_assign_panel_source_prompt(
                figure.figure_label, panel_labels, extracted_files
            )

            # Call AI service
            assigned_files_list = self.call_ai_service(prompt)

        # Convert assigned files to Panel objects
        panels = self.parse_assigned_files_to_panels(assigned_files_list)

        # Update figure with assignments
        self._update_figure_with_assignments(
            figure, panels, assigned_files_list.not_assigned_files
        )

    @abstractmethod
    def call_ai_service(self, prompt: str) -> AsignedFilesList:
        """Abstract method to call AI service."""
        pass

    def _get_zip_contents(self, sd_files: List[str]) -> List[str]:
        """Extract file list from zip files and return paths in a specific format."""
        extracted_files = []

        for file_path in sd_files:
            if file_path.endswith(".zip"):
                try:
                    with zipfile.ZipFile(file_path, "r") as zip_ref:
                        for file_info in zip_ref.infolist():
                            # Skip directories (they end with '/') and macOS metadata
                            if (
                                file_info.filename.endswith("/")
                                or "__MACOSX" in file_info.filename
                                or ".DS_Store" in file_info.filename
                            ):
                                continue

                            # Construct the path format: suppl_data/zip_file_path:internal_path
                            zip_relative_path = os.path.relpath(
                                file_path, self.extraction_dir
                            )
                            extracted_files.append(
                                f"{zip_relative_path}:{file_info.filename}"
                            )
                except zipfile.BadZipFile:
                    logger.error(f"Bad zip file: {file_path}")
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
        # Access assigned_files directly instead of using get()
        for assigned_file in assigned_files_list.assigned_files:
            panel = Panel(
                panel_label=assigned_file.panel_label,
                panel_caption="",  # Assuming caption is not provided in this context
                sd_files=assigned_file.panel_sd_files,
            )
            panels.append(panel)
        return panels

    def _update_figure_with_assignments(
        self, figure: Figure, panels: List[Panel], not_assigned_files: List[str]
    ) -> None:
        """Update figure with source data assignments."""
        try:
            # Update panel assignments
            for new_panel in panels:
                # Find matching existing panel
                existing_panel = next(
                    (
                        p
                        for p in figure.panels
                        if p.panel_label == new_panel.panel_label
                    ),
                    None,
                )

                if existing_panel:
                    # Update existing panel's sd_files
                    existing_panel.sd_files = new_panel.sd_files
                else:
                    # Add new panel to figure
                    figure.panels.append(new_panel)

            # Update unassigned files
            figure.unassigned_sd_files = not_assigned_files

            # Remove duplicates from figure's sd_files
            figure.sd_files = list(set(figure.sd_files))

            logger.info(
                f"Updated figure {figure.figure_label} with {len(panels)} panel assignments"
            )

        except Exception as e:
            logger.error(
                f"Error updating assignments for {figure.figure_label}: {str(e)}"
            )
            raise
