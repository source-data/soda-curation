import json
import logging
import os
import re
import zipfile
from typing import Any, Dict, List, Union

import openai
from openai._types import NotGiven

from soda_curation.pipeline.manuscript_structure.manuscript_structure import (
    Figure,
    ZipStructure,
)

from .assign_panel_source_prompts import SYSTEM_PROMPT, get_assign_panel_source_prompt

logger = logging.getLogger(__name__)

class PanelSourceAssigner:
    """
    A class to assign source data files to specific panels within figures using OpenAI's API.

    This class interacts with an OpenAI assistant to intelligently assign source data files
    to individual panels based on file names and structures.

    Attributes:
        config (Dict[str, Any]): Configuration dictionary for OpenAI API and other settings.
        client (openai.OpenAI): OpenAI API client.
        assistant (openai.types.beta.Assistant): OpenAI assistant for panel source assignment.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the PanelSourceAssigner with the given configuration.

        Args:
            config (Dict[str, Any]): Configuration dictionary containing OpenAI API settings.

        Raises:
            ValueError: If required configuration parameters are missing.
        """
        self.config = config
        self.client = openai.OpenAI(api_key=self.config["openai"]["api_key"])
        self.assistant = self._setup_assistant()
        logger.info("PanelSourceAssigner initialized successfully")

    def _setup_assistant(self):
        """
        Set up or retrieve the OpenAI assistant for panel source assignment.

        Returns:
            openai.types.beta.Assistant: The OpenAI assistant object.

        Raises:
            ValueError: If the panel_source_data_assistant_id is not set in the configuration.
        """
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
        """
        Assign source data files to panels for all figures in the ZIP structure.

        Args:
            zip_structure (ZipStructure): The ZIP structure containing figures and their information.

        Returns:
            ZipStructure: Updated ZIP structure with assigned panel source data.
        """
        logger.info("Starting panel source assignment process")
        
        if isinstance(input_obj, ZipStructure):
            return self._assign_to_zip_structure(input_obj)
        elif isinstance(input_obj, Figure):
            return self._assign_to_figure(input_obj)
        else:
            raise TypeError(f"Expected ZipStructure or Figure, got {type(input_obj)}")
        
    def _assign_to_zip_structure(self, zip_structure: ZipStructure) -> ZipStructure:
        """Process entire ZipStructure object"""
        ev_materials = []
        for figure in zip_structure.figures:
            if figure._full_sd_files:
                figure = self._assign_to_figure(figure)
                
        # Handle EV materials at structure level
        self._assign_ev_materials(zip_structure, ev_materials)
        return zip_structure

    def _assign_to_figure(self, figure: Figure) -> Figure:
        """Process single figure"""
        if not figure._full_sd_files:
            logger.warning(f"No source data files found for figure: {figure.figure_label}")
            return figure

        sd_zip_file = None
        figure.sd_files = []  # Reset sd_files
        
        # Find ZIP file among source data files
        for file in figure._full_sd_files:
            if file.endswith('.zip'):
                sd_zip_file = file
                break
        
        if sd_zip_file:
            try:
                with zipfile.ZipFile(sd_zip_file, 'r') as zip_ref:
                    file_list = [
                        f for f in zip_ref.namelist()
                        if not f.startswith('__MACOSX')
                        and not f.endswith('.DS_Store')
                        and not f == ""
                        and not f.endswith('/')
                    ]
                
                if not file_list:
                    logger.warning(f"No valid files found in ZIP for {figure.figure_label}")
                    return figure

                # Get panel labels
                panel_labels = [panel.panel_label for panel in figure.panels if panel.panel_label]
                if not panel_labels:
                    logger.warning(f"No panel labels found for {figure.figure_label}")
                    return figure
                
                panel_labels_str = ", ".join(panel_labels)
                prompt = get_assign_panel_source_prompt(figure.figure_label, panel_labels_str, "\n".join(file_list))
                
                # Get AI response for file assignments
                ai_response = self._call_openai_api(prompt)
                if not ai_response:
                    return figure
                    
                # Parse and update assignments
                assignments = self._parse_response(ai_response)
                if assignments:
                    self._update_figure_with_assignments(figure, assignments)
                    figure.ai_response_panel_source_assign = ai_response
                    
            except Exception as e:
                logger.error(f"Error processing {figure.figure_label}: {str(e)}")
        
        return figure

    def _process_figure_zip(self, figure: Figure, zip_file_path: str, zip_structure: ZipStructure):
        """Process a single figure's ZIP file, assigning source data files to its panels.

        Args:
            figure (Figure): The figure object to process.
            zip_file_path (str): Path to the figure's source data ZIP file.
            zip_structure (ZipStructure): The overall ZIP structure.
        """
        try:
            file_list = []
            # First extract full file list from ZIP
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                file_list = [
                    f for f in zip_ref.namelist() 
                    if not f.startswith('__MACOSX') 
                    and not f.endswith('.DS_Store')
                    and not f == ""  # Skip empty entries
                    and not f.endswith('/')  # Skip directory entries
                ]
            
            if not file_list:
                logger.warning(f"No valid files found in ZIP for {figure.figure_label}")
                return

            logger.debug(f"Files in ZIP for {figure.figure_label}: {file_list}")

            # Create panel list with labels
            panel_labels = []
            for panel in figure.panels:
                if panel.panel_label:  # Only include panels with assigned labels
                    panel_labels.append(panel.panel_label)
            
            if not panel_labels:
                logger.warning(f"No panel labels found for {figure.figure_label}")
                return
                
            panel_labels_str = ", ".join(panel_labels)
            prompt = get_assign_panel_source_prompt(figure.figure_label, panel_labels_str, "\n".join(file_list))
            logger.debug(f"Generated prompt for {figure.figure_label}: {prompt}")

            # Get AI response
            ai_response = self._call_openai_api(prompt)
            if not ai_response:
                logger.warning(f"No AI response for {figure.figure_label}")
                return
                
            logger.debug(f"Raw API response for {figure.figure_label}: {ai_response}")
            
            # Parse response
            try:
                assignments = self._parse_response(ai_response)
                if not assignments:
                    logger.warning(f"No valid assignments parsed for {figure.figure_label}")
                    return
                    
                logger.debug(f"Parsed assignments for {figure.figure_label}: {json.dumps(assignments, indent=2)}")
                
                # Update figure with assignments
                self._update_figure_with_assignments(figure, assignments)
                figure.ai_response_panel_source_assign = ai_response
                
            except Exception as e:
                logger.error(f"Error parsing assignments for {figure.figure_label}: {str(e)}")
                
        except zipfile.BadZipFile:
            logger.error(f"Invalid ZIP file: {zip_file_path}")
        except Exception as e:
            logger.error(f"Error processing {figure.figure_label}: {str(e)}")
            
    def _update_figure_with_assignments(self, figure: Figure, assignments: Dict[str, List[str]]):
        """
        Update the figure with source data file assignments.

        Args:
            figure (Figure): The figure to update
            assignments (Dict[str, List[str]]): The assignments of files to panels
        """
        logger.info(f"Updating {figure.figure_label} with source data assignments")
        assigned_files = set()
        
        # First assign files to panels
        for panel in figure.panels:
            if panel.panel_label in assignments:
                panel_files = assignments[panel.panel_label]
                logger.debug(f"Assigning {len(panel_files)} files to panel {panel.panel_label}")
                panel.sd_files = panel_files
                assigned_files.update(panel_files)
            else:
                logger.debug(f"No files assigned to panel {panel.panel_label}")
                panel.sd_files = []
                
        # Handle unassigned files
        if 'unassigned' in assignments:
            unassigned_files = set(assignments['unassigned']) - assigned_files
            logger.debug(f"Found {len(unassigned_files)} unassigned files")
            if not hasattr(figure, 'unassigned_sd_files'):
                figure.unassigned_sd_files = []
            figure.unassigned_sd_files.extend(unassigned_files)
            
    def _process_figure(self, figure: Figure, figure_files: Dict[str, Any], zip_structure: ZipStructure):
        """
        Process a single figure, assigning source data files to its panels.

        Args:
            figure (Figure): The figure object to process.
            figure_files (Dict[str, Any]): Dictionary representing the file structure for this figure.
            zip_structure (ZipStructure): The overall ZIP structure.
        """
        file_list = self._flatten_file_tree(figure_files)
        logger.info(f"Flattened file list for {figure.figure_label}: {file_list}")
        
        panel_labels = ", ".join([panel.panel_label for panel in figure.panels])
        prompt = get_assign_panel_source_prompt(figure.figure_label, panel_labels, file_list)
        logger.info(f"Generated prompt for {figure.figure_label}: {prompt}")

        try:
            logger.info(f"Calling OpenAI API for figure: {figure.figure_label}")
            response = self._call_openai_api(prompt)
            logger.info(f"Raw API response for {figure.figure_label}: {response}")
            assignments = self._parse_response(response)
            logger.info(f"Parsed assignments for {figure.figure_label}: {json.dumps(assignments, indent=2)}")
            self._update_figure_with_assignments(figure, assignments, zip_structure)
            figure.ai_response_panel_source_assign = response
        except Exception as e:
            logger.error(f"Error processing figure {figure.figure_label}: {str(e)}")
            figure.ai_response_panel_source_assign = f"Error: {str(e)}"

    def _flatten_file_tree(self, file_tree: Dict[str, Any], prefix: str = "") -> str:
        """
        Flatten a nested dictionary representing a file tree into a string.

        Args:
            file_tree (Dict[str, Any]): The nested dictionary representing the file tree.
            prefix (str, optional): The current path prefix. Defaults to "".

        Returns:
            str: A string representation of the flattened file tree.
        """
        file_list = []
        for key, value in file_tree.items():
            path = f"{prefix}/{key}" if prefix else key
            if isinstance(value, dict):
                file_list.extend(self._flatten_file_tree(value, path))
            else:
                file_list.append(path)
        logger.debug(f"Flattened file tree: {file_list}")
        return "\n".join(file_list)

    def _call_openai_api(self, prompt: str) -> str:
        """
        Call the OpenAI API to get panel source assignments.

        Args:
            prompt (str): The prompt to send to the OpenAI API.

        Returns:
            str: The response from the OpenAI API.

        Raises:
            Exception: If there's an error in the API call.
        """
        try:
            logger.info("Creating new thread for OpenAI API call")
            thread = self.client.beta.threads.create()
            logger.debug(f"Created thread with ID: {thread.id}")

            logger.info("Adding message to thread")
            self.client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=prompt
            )

            logger.info("Creating and running assistant")
            run = self.client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=self.assistant.id,
            )
            logger.debug(f"Created run with ID: {run.id}")

            logger.info("Waiting for assistant to complete")
            while run.status != "completed":
                run = self.client.beta.threads.runs.retrieve(
                    thread_id=thread.id,
                    run_id=run.id
                )
                logger.debug(f"Run status: {run.status}")

            logger.info("Retrieving assistant's response")
            messages = self.client.beta.threads.messages.list(thread_id=thread.id)
            return messages.data[0].content[0].text.value
        except Exception as e:
            logger.error(f"Error in OpenAI API call: {str(e)}")
            raise

    def _parse_response(self, response: str) -> Dict[str, List[str]]:
        """
        Parse the JSON response from the OpenAI API.

        Args:
            response (str): The JSON string response from the OpenAI API.

        Returns:
            Dict[str, List[str]]: A dictionary of panel labels and their assigned files.
        """
        try:
            # Extract JSON object from the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed_response = json.loads(json_str)
            else:
                parsed_response = json.loads(response)
            
            logger.info("Successfully parsed API response")
            logger.debug(f"Parsed response: {parsed_response}")
            return parsed_response
        except json.JSONDecodeError:
            logger.error("Failed to parse JSON from OpenAI's response")
            logger.debug(f"Raw response that failed to parse: {response}")
            return {}

    def _assign_ev_materials(self, zip_structure: ZipStructure, ev_materials: List[str]):
        """
        Assign EV datasets to the appropriate figures based on the dataset name.
        
        Args:
            zip_structure (ZipStructure): The overall ZIP structure to update.
            ev_materials (List[str]): List of EV dataset file names.
            
        Returns:
            None
        
        """
        for material in ev_materials:
            material_name = os.path.basename(material)
            match = re.search(r'(Figure|Table|Dataset)\s*EV(\d+)', material_name, re.IGNORECASE)
            if match:
                ev_type = match.group(1).capitalize()
                ev_number = match.group(2)
                ev_figure = next((fig for fig in zip_structure.figures if fig.figure_label == f"{ev_type} EV{ev_number}"), None)
                if ev_figure:
                    ev_figure.sd_files.append(material_name)
                    logger.info(f"Assigned {material_name} to {ev_figure.figure_label}")
                else:
                    zip_structure.non_associated_sd_files.append(material_name)
                    logger.info(f"Added {material_name} to non_associated_sd_files (no matching EV figure found)")
            else:
                zip_structure.non_associated_sd_files.append(material_name)
                logger.info(f"Added {material_name} to non_associated_sd_files (not recognized as EV material)")

