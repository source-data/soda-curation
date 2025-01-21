"""Panel source assignment module."""

import json
import logging
import os
import re
import zipfile
from typing import Any, Dict, List, Union

import openai
from openai._types import NotGiven

from ..manuscript_structure.manuscript_structure import Figure, ZipStructure
from .assign_panel_source_prompts import SYSTEM_PROMPT, get_assign_panel_source_prompt

logger = logging.getLogger(__name__)

class PanelSourceAssigner:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = openai.OpenAI(api_key=self.config["openai"]["api_key"])
        self.assistant = self._setup_assistant()
        logger.info("PanelSourceAssigner initialized successfully")

    def _setup_assistant(self):
        assistant_id = self.config["openai"].get("panel_source_data_assistant_id")
        if assistant_id:
            logger.info(f"Updating existing assistant with ID: {assistant_id}")
            return self.client.beta.assistants.update(
                assistant_id,
                instructions=SYSTEM_PROMPT,
                model=self.config["openai"]["model"]
            )
        else:
            logger.error("panel_source_data_assistant_id is not set in the configuration")
            raise ValueError("panel_source_data_assistant_id is not set in the configuration")

    def assign_panel_source(self, input_obj: Union[ZipStructure, Figure]) -> Union[ZipStructure, Figure]:
        logger.info("Starting panel source assignment process")
        
        if isinstance(input_obj, ZipStructure):
            return self._assign_to_zip_structure(input_obj)
        elif isinstance(input_obj, Figure):
            return self._assign_to_figure(input_obj)
        else:
            raise TypeError(f"Expected ZipStructure or Figure, got {type(input_obj)}")
        
    def _assign_to_zip_structure(self, zip_structure: ZipStructure) -> ZipStructure:
        """Process entire ZipStructure object."""
        for figure in zip_structure.figures:
            if figure._full_sd_files:
                figure = self._assign_to_figure(figure)
                
        # Handle EV materials
        self._process_ev_materials(zip_structure)
        return zip_structure

    def _assign_to_figure(self, figure: Figure) -> Figure:
        """Process single figure."""
        if not figure._full_sd_files:
            logger.warning(f"No source data files found for figure: {figure.figure_label}")
            return figure

        # Find ZIP file among source data files
        zip_files = [f for f in figure._full_sd_files if f.endswith('.zip')]
        if not zip_files:
            return figure

        for zip_file in zip_files:
            try:
                file_list = self._get_zip_contents(zip_file)
                if not file_list:
                    continue

                panel_labels = [panel.panel_label for panel in figure.panels if panel.panel_label]
                if not panel_labels:
                    continue

                # Get AI response and assignments
                prompt = get_assign_panel_source_prompt(
                    figure.figure_label, 
                    ", ".join(panel_labels), 
                    "\n".join(file_list)
                )
                ai_response = self._call_openai_api(prompt)
                if not ai_response:
                    continue

                assignments = self._parse_response(ai_response)
                if not assignments:
                    continue

                # Update figure with assignments
                self._update_figure_with_assignments(figure, zip_file, assignments)
                figure.ai_response_panel_source_assign = ai_response

            except Exception as e:
                logger.error(f"Error processing ZIP file {zip_file} for {figure.figure_label}: {str(e)}")

        return figure

    def _get_zip_contents(self, zip_path: str) -> List[str]:
        """Get list of valid files from ZIP archive using exact internal paths."""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Get all files, excluding Mac OS metadata and empty directories
                return [
                    f for f in zip_ref.namelist()
                    if not f.startswith('__MACOSX')
                    and not f.endswith('.DS_Store')
                    and not f.endswith('/')
                ]
        except Exception as e:
            logger.error(f"Error reading ZIP file {zip_path}: {str(e)}")
            return []

    def _update_figure_with_assignments(self, figure: Figure, zip_path: str, assignments: Dict[str, List[str]]):
        """Update figure with source data assignments using exact ZIP paths."""
        try:
            zip_filename = os.path.basename(zip_path)
            assigned_files = set()

            # Clean existing assignments
            figure.sd_files = []
            for panel in figure.panels:
                panel.sd_files = []

            # Process panel assignments
            for panel in figure.panels:
                if panel.panel_label in assignments:
                    panel_files = assignments[panel.panel_label]
                    
                    # Construct proper paths maintaining the full directory structure
                    panel.sd_files = [
                        f"suppl_data/{zip_filename}:{file}"
                        for file in panel_files
                    ]
                    assigned_files.update(panel.sd_files)

            # Add unassigned files
            if 'unassigned' in assignments:
                unassigned = [
                    f"suppl_data/{zip_filename}:{file}"
                    for file in assignments['unassigned']
                ]
                figure.unassigned_sd_files = unassigned
                assigned_files.update(unassigned)

            # Set figure source data files
            figure.sd_files = ["suppl_data/" + zip_filename]

        except Exception as e:
            logger.error(f"Error updating assignments for {figure.figure_label}: {str(e)}")

    def _process_ev_materials(self, zip_structure: ZipStructure):
        """Process and assign EV materials."""
        ev_materials = []
        for figure in zip_structure.figures:
            if figure._full_sd_files:
                for file in figure._full_sd_files:
                    if re.search(r'(Figure|Table|Dataset)\s*EV', os.path.basename(file), re.IGNORECASE):
                        ev_materials.append(os.path.basename(file))

        for material in ev_materials:
            match = re.search(r'(Figure|Table|Dataset)\s*EV(\d+)', material, re.IGNORECASE)
            if match:
                ev_type = match.group(1).capitalize()
                ev_number = match.group(2)
                ev_figure = next(
                    (fig for fig in zip_structure.figures 
                     if fig.figure_label == f"{ev_type} EV{ev_number}"), 
                    None
                )
                if ev_figure:
                    ev_figure.sd_files = ["suppl_data/" + material]
                    logger.info(f"Assigned {material} to {ev_figure.figure_label}")
                else:
                    if not hasattr(zip_structure, 'non_associated_sd_files'):
                        zip_structure.non_associated_sd_files = []
                    zip_structure.non_associated_sd_files.append("suppl_data/" + material)
                    logger.info(f"Added {material} to non_associated_sd_files")

    def _call_openai_api(self, prompt: str) -> str:
        """Call the OpenAI API and get response."""
        try:
            thread = self.client.beta.threads.create()
            
            self.client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=prompt
            )

            run = self.client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=self.assistant.id,
            )

            while run.status != "completed":
                run = self.client.beta.threads.runs.retrieve(
                    thread_id=thread.id,
                    run_id=run.id
                )

            messages = self.client.beta.threads.messages.list(thread_id=thread.id)
            response = messages.data[0].content[0].text.value
            logger.info(f"****************")
            logger.info(f"PANEL SOURCE ASSIGNMENT RESPONSE")
            logger.info(f"****************")
            logger.info(response)

            # Cleanup
            self.client.beta.threads.delete(thread_id=thread.id)
            
            return response

        except Exception as e:
            logger.error(f"Error in OpenAI API call: {str(e)}")
            return ""

    def _parse_response(self, response: str) -> Dict[str, List[str]]:
        """Parse JSON response from OpenAI."""
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            return json.loads(response)
            
        except json.JSONDecodeError:
            logger.error("Failed to parse JSON from OpenAI's response")
            return {}
