"""Factory for creating QC analyzers."""

import importlib
import logging
from typing import Dict, Optional

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
    """Factory for creating QC analyzers based on test type."""

    @classmethod
    def create_analyzer(cls, test_name: str, config: Dict) -> BaseQCAnalyzer:
        """Create an analyzer for the given test name."""
        try:
            # Try to load a custom module for this test
            module_name = f"..qc_tests.{test_name}"
            analyzer_class = None

            # Look for a class named like TestNameAnalyzer
            class_name = (
                "".join(word.capitalize() for word in test_name.split("_")) + "Analyzer"
            )

            try:
                # Try to import the module and get the analyzer class
                module = importlib.import_module(
                    module_name, package="soda_curation.qc"
                )
                analyzer_class = getattr(module, class_name, None)

                if analyzer_class:
                    return analyzer_class(config)

            except (ImportError, AttributeError) as e:
                logger.debug(f"No custom module found for {test_name}: {str(e)}")

            # If no custom analyzer found, create a generic one
            return cls.create_generic_analyzer(test_name, config)

        except Exception as e:
            logger.error(f"Error creating analyzer for {test_name}: {str(e)}")
            return cls.create_generic_analyzer(test_name, config)

    @classmethod
    def create_generic_analyzer(cls, test_name: str, config: Dict) -> BaseQCAnalyzer:
        """Create a generic analyzer based on the test type."""
        test_type = cls._determine_test_type(test_name, config)

        if test_type == "panel":
            return GenericPanelQCAnalyzer(test_name, config)
        elif test_type == "figure":
            return GenericFigureQCAnalyzer(test_name, config)
        elif test_type in ["document", "manuscript"]:
            return GenericManuscriptQCAnalyzer(test_name, config)
        else:
            logger.warning(
                f"Couldn't determine test type for {test_name}, defaulting to panel-level"
            )
            return GenericPanelQCAnalyzer(test_name, config)

    @classmethod
    def _determine_type_from_schema(cls, test_name: str) -> Optional[str]:
        """Determine test type by inspecting the generated Pydantic model structure.

        This is much more reliable than parsing raw JSON schemas because we can
        inspect the actual Python types after model generation.
        """
        try:
            # First check if a real schema exists
            try:
                registry.get_schema(test_name)
                # Schema exists, proceed with model-based detection
                model_class = registry.get_pydantic_model(test_name)
                if not model_class:
                    return None

                # Use the registry's model inspection method
                return registry.determine_test_type_from_model(model_class)
            except FileNotFoundError:
                # No schema found, return None to fall back to config-based detection
                return None

        except Exception as e:
            logger.debug(f"Error determining type from model for {test_name}: {str(e)}")
            return None

    @classmethod
    def _determine_test_type(cls, test_name: str, config: Dict) -> str:
        """Determine the test type based on schema structure, config structure, and test name."""

        # First, try to determine test type from schema structure
        try:
            schema_type = cls._determine_type_from_schema(test_name)
            if schema_type:
                logger.debug(
                    f"Determined {test_name} as {schema_type} from schema structure"
                )
                return schema_type
        except Exception as e:
            logger.debug(
                f"Could not determine type from schema for {test_name}: {str(e)}"
            )

        # Second, check if the test is in the qc_check_metadata structure
        if "qc_check_metadata" in config:
            # Check each level
            panel_tests = config["qc_check_metadata"].get("panel", {})
            if panel_tests and test_name in panel_tests:
                return "panel"
            figure_tests = config["qc_check_metadata"].get("figure", {})
            if figure_tests and test_name in figure_tests:
                return "figure"

            document_tests = config["qc_check_metadata"].get("document", {})
            if document_tests and test_name in document_tests:
                return "document"

        # Third, check if there's an explicit test_type in the test config
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
            # Get optional word_file_path from args or kwargs
            word_file_path = args[1] if len(args) >= 2 else kwargs.get("word_file_path")
            return self.analyze_manuscript(args[0], word_file_path)
        elif "zip_structure" in kwargs:
            word_file_path = kwargs.get("word_file_path")
            return self.analyze_manuscript(kwargs["zip_structure"], word_file_path)
        else:
            logger.error(f"Incorrect arguments for analyze method: {args}, {kwargs}")
            return False, self.create_empty_result()
