"""Factory for creating QC analyzers."""

import importlib
import inspect
import logging
from typing import Any, Dict, Type

from .base_analyzers import (
    BaseQCAnalyzer,
    FigureQCAnalyzer,
    ManuscriptQCAnalyzer,
    PanelQCAnalyzer,
)
from .model_api import ModelAPI
from .prompt_registry import registry

logger = logging.getLogger(__name__)


class AnalyzerFactory:
    """Factory for creating QC analyzers."""

    @classmethod
    def create_analyzer(cls, test_name: str, config: Dict) -> BaseQCAnalyzer:
        """Create an analyzer based on the test name."""
        # Try to import a custom module for this test
        try:
            module_name = f"..qc_tests.{test_name}"
            module = importlib.import_module(module_name, package="soda_curation.qc")

            # Convert snake_case to CamelCase for class name
            class_name = (
                "".join(word.capitalize() for word in test_name.split("_")) + "Analyzer"
            )

            # Get the analyzer class
            if hasattr(module, class_name):
                analyzer_class = getattr(module, class_name)
                return analyzer_class(config)

        except (ModuleNotFoundError, AttributeError) as e:
            logger.debug(f"No custom module found for {test_name}: {str(e)}")

        # If no custom module, try to create a generic analyzer based on test type
        return cls.create_generic_analyzer(test_name, config)

    @classmethod
    def create_generic_analyzer(cls, test_name: str, config: Dict) -> BaseQCAnalyzer:
        """Create a generic analyzer based on the test configuration."""
        # Determine the test type from configuration
        test_type = cls._determine_test_type(test_name, config)

        # Create the appropriate analyzer
        if test_type == "panel":
            return GenericPanelQCAnalyzer(test_name, config)
        elif test_type == "figure":
            return GenericFigureQCAnalyzer(test_name, config)
        elif test_type == "document" or test_type == "manuscript":
            return GenericManuscriptQCAnalyzer(test_name, config)
        else:
            # Default to panel-level if we can't determine
            logger.warning(
                f"Couldn't determine test type for {test_name}, defaulting to panel-level"
            )
            return GenericPanelQCAnalyzer(test_name, config)

    @classmethod
    def _determine_test_type(cls, test_name: str, config: Dict) -> str:
        """Determine the test type based on config structure and test name."""
        # First check if the test is in the qc_test_metadata structure
        if "qc_test_metadata" in config:
            # Check each level
            if (
                "panel" in config["qc_test_metadata"]
                and test_name in config["qc_test_metadata"]["panel"]
            ):
                return "panel"
            if (
                "figure" in config["qc_test_metadata"]
                and test_name in config["qc_test_metadata"]["figure"]
            ):
                return "figure"
            if (
                "document" in config["qc_test_metadata"]
                and test_name in config["qc_test_metadata"]["document"]
            ):
                return "document"

        # If not in metadata structure, check if there's an explicit test_type in the test config
        test_config = config.get("default", {}).get("pipeline", {}).get(test_name, {})
        if "test_type" in test_config:
            return test_config["test_type"]

        # Use naming conventions as fallback
        if test_name.startswith("manuscript_") or test_name.startswith("document_"):
            return "document"
        elif test_name.startswith("figure_"):
            return "figure"
        else:
            return "panel"  # Default to panel-level


class GenericPanelQCAnalyzer(PanelQCAnalyzer):
    """Generic implementation of a panel-level QC analyzer."""

    def __init__(self, test_name: str, config: Dict):
        """Initialize with a specific test name."""
        # Don't call super().__init__ since it uses class name for test_name
        # Instead, initialize required attributes directly
        self.config = config
        self.model_api = ModelAPI(config)
        self.test_name = test_name

        # Get metadata and model for this test
        self.metadata = registry.get_prompt_metadata(test_name)
        self.result_model = registry.get_pydantic_model(test_name)

    def analyze(self, *args, **kwargs) -> tuple:
        """Implement the abstract analyze method."""
        if len(args) >= 3:
            return self.analyze_figure(args[0], args[1], args[2])
        elif all(
            k in kwargs for k in ["figure_label", "encoded_image", "figure_caption"]
        ):
            return self.analyze_figure(
                kwargs["figure_label"],
                kwargs["encoded_image"],
                kwargs["figure_caption"],
            )
        else:
            logger.error(f"Incorrect arguments for analyze method: {args}, {kwargs}")
            return False, self.create_empty_result()


class GenericFigureQCAnalyzer(FigureQCAnalyzer):
    """Generic implementation of a figure-level QC analyzer."""

    def __init__(self, test_name: str, config: Dict):
        """Initialize with a specific test name."""
        self.config = config
        self.model_api = ModelAPI(config)
        self.test_name = test_name

        # Get metadata and model for this test
        self.metadata = registry.get_prompt_metadata(test_name)
        self.result_model = registry.get_pydantic_model(test_name)

    def analyze(self, *args, **kwargs) -> tuple:
        """Implement the abstract analyze method."""
        if len(args) >= 3:
            return self.analyze_figure(args[0], args[1], args[2])
        elif all(
            k in kwargs for k in ["figure_label", "encoded_image", "figure_caption"]
        ):
            return self.analyze_figure(
                kwargs["figure_label"],
                kwargs["encoded_image"],
                kwargs["figure_caption"],
            )
        else:
            logger.error(f"Incorrect arguments for analyze method: {args}, {kwargs}")
            return False, self.create_empty_result()


class GenericManuscriptQCAnalyzer(ManuscriptQCAnalyzer):
    """Generic implementation of a manuscript-level QC analyzer."""

    def __init__(self, test_name: str, config: Dict):
        """Initialize with a specific test name."""
        self.config = config
        self.model_api = ModelAPI(config)
        self.test_name = test_name

        # Get metadata and model for this test
        self.metadata = registry.get_prompt_metadata(test_name)
        self.result_model = registry.get_pydantic_model(test_name)

    def analyze(self, *args, **kwargs) -> tuple:
        """Implement the abstract analyze method."""
        if len(args) >= 1:
            return self.analyze_manuscript(args[0])
        elif "zip_structure" in kwargs:
            return self.analyze_manuscript(kwargs["zip_structure"])
        else:
            logger.error(f"Incorrect arguments for analyze method: {args}, {kwargs}")
            return False, self.create_empty_result()
