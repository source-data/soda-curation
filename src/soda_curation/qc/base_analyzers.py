"""Base QC test analyzers."""

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel

from .model_api import ModelAPI
from .prompt_registry import registry

logger = logging.getLogger(__name__)


class BaseQCAnalyzer(ABC):
    """Base class for all QC analyzers."""

    def __init__(self, config: Dict):
        """Initialize the analyzer."""
        self.config = config
        self.model_api = ModelAPI(config)
        self.test_name = self.__class__.__name__.replace("Analyzer", "").lower()

        # Get metadata from registry for traceability
        self.metadata = registry.get_prompt_metadata(self.test_name)

        # Get the dynamically generated model
        self.result_model = registry.get_pydantic_model(self.test_name)

    def get_test_config(self) -> Dict:
        """Get the test-specific configuration."""
        # Look in default.pipeline if available
        if "default" in self.config and "pipeline" in self.config["default"]:
            pipeline_config = self.config["default"]["pipeline"]
            if self.test_name in pipeline_config:
                return pipeline_config[self.test_name]

        # Fallback to root level
        return self.config.get(self.test_name, {})

    @abstractmethod
    def analyze(self, *args, **kwargs) -> Tuple[bool, Any]:
        """Run the analysis and return results."""
        pass

    def create_empty_result(self) -> Dict:
        """Create an empty result object."""
        return {"outputs": []}


class PanelQCAnalyzer(BaseQCAnalyzer):
    """Base class for panel-level QC tests."""

    def analyze_figure(
        self, figure_label: str, encoded_image: str, figure_caption: str
    ) -> Tuple[bool, Any]:
        """Analyze a figure and return results."""
        try:
            # Get API config from main config or use defaults
            test_config = self.get_test_config()

            # Make sure we have an openai section
            if "openai" not in test_config:
                # Use default openai config from main config
                if "default" in self.config and "openai" in self.config["default"]:
                    test_config["openai"] = self.config["default"]["openai"].copy()
                else:
                    # Create a minimal openai config
                    test_config["openai"] = {
                        "model": "gpt-4o",
                        "temperature": 0.1,
                        "json_mode": True,
                    }

            # Make sure we have prompts section
            if "prompts" not in test_config["openai"]:
                test_config["openai"]["prompts"] = {}

            # Set system prompt with one from registry
            test_config["openai"]["prompts"]["system"] = registry.get_prompt(
                self.test_name
            )

            # Call the model with the correct parameters
            response = self.model_api.generate_response(
                encoded_image=encoded_image,
                caption=figure_caption,
                prompt_config=test_config["openai"],
                response_type=self.result_model,
            )

            # Process and validate the response
            result = self.process_response(response)

            # Check if the test passed
            passed = self.check_test_passed(result)

            return passed, result

        except Exception as e:
            logger.error(f"Error analyzing figure with {self.test_name}: {str(e)}")
            return False, self.create_empty_result()

    def process_response(self, response: Any) -> Any:
        """Process the response from the model API."""
        # If response is a string, try to parse it
        if isinstance(response, str):
            try:
                response_data = json.loads(response)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse response as JSON: {response}")
                return self.create_empty_result()
        else:
            response_data = response

        # Handle both direct outputs and nested outputs
        if isinstance(response_data, dict):
            if "outputs" in response_data:
                return response_data
            else:
                # Wrap in outputs structure if not already present
                return {"outputs": [response_data]}

        # Return empty result if we can't process the response
        return self.create_empty_result()

    def check_test_passed(self, result: Any) -> bool:
        """Check if the test passed based on the result."""
        # Default implementation
        if not result or not isinstance(result, dict) or "outputs" not in result:
            return False

        # If any panel has issues, the test fails
        for panel in result.get("outputs", []):
            if not self.check_panel_passed(panel):
                return False

        # If we get here, all panels passed
        return True

    def check_panel_passed(self, panel: Any) -> bool:
        """Check if an individual panel passed the test."""
        # Default implementation considers any non-empty panel as passed
        return panel is not None and len(panel) > 0


class FigureQCAnalyzer(BaseQCAnalyzer):
    """Base class for figure-level QC tests."""

    def analyze_figure(
        self, figure_label: str, encoded_image: str, figure_caption: str
    ) -> Tuple[bool, Any]:
        """Analyze a whole figure and return results."""
        try:
            # Get API config from main config or use defaults
            test_config = self.get_test_config()

            # Make sure we have an openai section
            if "openai" not in test_config:
                # Use default openai config from main config
                if "default" in self.config and "openai" in self.config["default"]:
                    test_config["openai"] = self.config["default"]["openai"].copy()
                else:
                    # Create a minimal openai config
                    test_config["openai"] = {
                        "model": "gpt-4o",
                        "temperature": 0.1,
                        "json_mode": True,
                    }

            # Make sure we have prompts section
            if "prompts" not in test_config["openai"]:
                test_config["openai"]["prompts"] = {}

            # Set system prompt with one from registry
            test_config["openai"]["prompts"]["system"] = registry.get_prompt(
                self.test_name
            )

            # Call the model with the correct parameters
            response = self.model_api.generate_response(
                encoded_image=encoded_image,
                caption=figure_caption,
                prompt_config=test_config["openai"],
                response_type=self.result_model,
            )

            # Process and validate the response
            result = self.process_response(response)

            # Check if the test passed
            passed = self.check_test_passed(result)

            return passed, result

        except Exception as e:
            logger.error(f"Error analyzing figure with {self.test_name}: {str(e)}")
            return False, self.create_empty_result()

    def process_response(self, response: Any) -> Any:
        """Process the response from the model API."""
        if not response:
            return self.create_empty_result()

        # If response is a JSON string, parse it
        if isinstance(response, str):
            try:
                import json

                parsed_response = json.loads(response)
                return parsed_response
            except json.JSONDecodeError:
                logger.error(
                    f"Failed to parse JSON response for {self.test_name}: {response}"
                )
                return self.create_empty_result()

        return response

    def check_test_passed(self, result: Any) -> bool:
        """Check if the test passed based on the result."""
        # Figure-level implementation
        return bool(result)


class ManuscriptQCAnalyzer(BaseQCAnalyzer):
    """Base class for manuscript-level QC tests."""

    def analyze_manuscript(
        self, zip_structure: Any, word_file_path: str = None
    ) -> Tuple[bool, Any]:
        """Analyze an entire manuscript and return results."""
        try:
            # Get API config from main config or use defaults
            test_config = self.get_test_config()

            # Make sure we have an openai section
            if "openai" not in test_config:
                # Use default openai config from main config
                if "default" in self.config and "openai" in self.config["default"]:
                    test_config["openai"] = self.config["default"]["openai"].copy()
                else:
                    # Create a minimal openai config
                    test_config["openai"] = {
                        "model": "gpt-4o",
                        "temperature": 0.1,
                        "json_mode": True,
                    }

            # Make sure we have prompts section
            if "prompts" not in test_config["openai"]:
                test_config["openai"]["prompts"] = {}

            # Set system prompt with one from registry
            test_config["openai"]["prompts"]["system"] = registry.get_prompt(
                self.test_name
            )

            # Extract word file content from zip_structure or provided path
            word_file_content = self.extract_word_file_content(
                zip_structure, word_file_path
            )

            # Call the model with the correct parameters
            response = self.model_api.generate_response(
                prompt_config=test_config["openai"],
                response_type=self.result_model,
                word_file_content=word_file_content,
            )

            # Process and validate the response
            result = self.process_response(response)

            # Check if the test passed
            passed = self.check_test_passed(result)

            return passed, result

        except Exception as e:
            logger.error(f"Error analyzing manuscript with {self.test_name}: {str(e)}")
            return False, self.create_empty_result()

    def extract_word_file_content(
        self, zip_structure: Any, word_file_path: str = None
    ) -> str:
        """Extract text content from word file in the manuscript structure."""
        try:
            # Import python-docx for reading Word documents
            from docx import Document

            # Determine the word file path to use
            docx_path = None

            if word_file_path:
                # Use provided word file path
                docx_path = word_file_path
            elif (
                zip_structure
                and hasattr(zip_structure, "_full_docx")
                and zip_structure._full_docx
            ):
                # Use full path from zip structure
                docx_path = zip_structure._full_docx
            elif (
                zip_structure and hasattr(zip_structure, "docx") and zip_structure.docx
            ):
                # Fallback to relative path (might need to be joined with extract directory)
                docx_path = zip_structure.docx

            if not docx_path:
                logger.warning("No word file path found in zip structure")
                return "No word file available for analysis"

            # Check if file exists
            import os

            if not os.path.exists(docx_path):
                logger.warning(f"Word file not found at path: {docx_path}")
                return f"Word file not found at path: {docx_path}"

            # Extract text from the Word document
            doc = Document(docx_path)

            # Extract all text from paragraphs
            text_content = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():  # Only include non-empty paragraphs
                    text_content.append(paragraph.text.strip())

            # Extract text from tables if any
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        if cell.text.strip():
                            text_content.append(cell.text.strip())

            # Join all text with newlines
            full_text = "\n".join(text_content)

            if not full_text.strip():
                logger.warning("No text content found in word file")
                return "No text content found in word file"

            logger.info(
                f"Successfully extracted {len(full_text)} characters from word file"
            )
            return full_text

        except ImportError:
            logger.error("python-docx library not available for reading Word documents")
            return "Error: python-docx library not available"
        except Exception as e:
            logger.error(f"Error extracting word file content: {str(e)}")
            return f"Error extracting word file content: {str(e)}"

    def extract_manuscript_text(self, zip_structure: Any) -> str:
        """Extract relevant text from the manuscript structure. (Legacy method - kept for compatibility)"""
        # Delegate to the new word file extraction method
        return self.extract_word_file_content(zip_structure)

    def process_response(self, response: Any) -> Any:
        """Process the response from the model API."""
        if not response:
            return self.create_empty_result()

        # If response is a JSON string, parse it
        if isinstance(response, str):
            try:
                import json

                parsed_response = json.loads(response)
                return parsed_response
            except json.JSONDecodeError:
                logger.error(
                    f"Failed to parse JSON response for {self.test_name}: {response}"
                )
                return self.create_empty_result()

        return response

    def check_test_passed(self, result: Any) -> bool:
        """Check if the test passed based on the result."""
        # Manuscript-level implementation
        return bool(result)
