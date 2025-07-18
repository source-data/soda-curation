"""Quality control pipeline for manuscript figures."""

import importlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..pipeline.manuscript_structure.manuscript_structure import ZipStructure
from .analyzer_factory import AnalyzerFactory
from .base_analyzers import (
    BaseQCAnalyzer,
    FigureQCAnalyzer,
    ManuscriptQCAnalyzer,
    PanelQCAnalyzer,
)
from .data_models import QCPipelineResult, QCResult

logger = logging.getLogger(__name__)


class QCPipeline:
    """Quality Control Pipeline for SODA curation."""

    def __init__(self, config, extract_dir):
        """
        Initialize QC Pipeline.

        Args:
            config: Configuration for QC pipeline
            extract_dir: Directory containing extracted manuscript files
        """
        self.config = config
        self.extract_dir = extract_dir
        self.tests = self._initialize_tests()
        self.qc_results = {"figures": {}}
        logger.info("Initialized QC Pipeline")

    def _initialize_tests(self) -> Dict:
        """Initialize QC test analyzers based on configuration."""
        tests = {}

        # Look for tests in default.pipeline
        if "default" in self.config and "pipeline" in self.config["default"]:
            pipeline_config = self.config["default"]["pipeline"]

            # Iterate through pipeline items to find test modules
            for test_name, test_config in pipeline_config.items():
                # Skip if this is a metadata entry or not a test
                if test_name.startswith("_") or not isinstance(test_config, dict):
                    continue

                try:
                    # Create analyzer using factory
                    analyzer = AnalyzerFactory.create_analyzer(test_name, self.config)
                    tests[test_name] = analyzer
                    logger.info(f"Loaded test module: {test_name}")
                except Exception as e:
                    logger.error(f"Error loading test module {test_name}: {str(e)}")

        # Also look for tests in qc_test_metadata structure
        if "qc_test_metadata" in self.config:
            # Check panel-level tests
            panel_tests = self.config["qc_test_metadata"].get("panel", {})
            if isinstance(panel_tests, dict):
                for test_name in panel_tests:
                    if test_name not in tests:  # Only if not already loaded
                        try:
                            analyzer = AnalyzerFactory.create_analyzer(
                                test_name, self.config
                            )
                            tests[test_name] = analyzer
                            logger.info(f"Loaded panel-level test module: {test_name}")
                        except Exception as e:
                            logger.error(
                                f"Error loading panel-level test module {test_name}: {str(e)}"
                            )

            # Check figure-level tests
            figure_tests = self.config["qc_test_metadata"].get("figure", {})
            if isinstance(figure_tests, dict):
                for test_name in figure_tests:
                    if test_name not in tests:  # Only if not already loaded
                        try:
                            analyzer = AnalyzerFactory.create_analyzer(
                                test_name, self.config
                            )
                            tests[test_name] = analyzer
                            logger.info(f"Loaded figure-level test module: {test_name}")
                        except Exception as e:
                            logger.error(
                                f"Error loading figure-level test module {test_name}: {str(e)}"
                            )

            # Check document-level tests
            doc_tests = self.config["qc_test_metadata"].get("document", {})
            if isinstance(doc_tests, dict):
                for test_name in doc_tests:
                    if test_name not in tests:  # Only if not already loaded
                        try:
                            analyzer = AnalyzerFactory.create_analyzer(
                                test_name, self.config
                            )
                            tests[test_name] = analyzer
                            logger.info(
                                f"Loaded document-level test module: {test_name}"
                            )
                        except Exception as e:
                            logger.error(
                                f"Error loading document-level test module {test_name}: {str(e)}"
                            )

        return tests

    def run(
        self,
        zip_structure: ZipStructure,
        figure_data: Optional[List[Tuple[str, str, str]]] = None,
        unified_output: bool = True,
    ) -> dict:
        """
        Run QC pipeline on the given zip structure.

        Args:
            zip_structure: ZipStructure object
            figure_data: List of tuples (figure_label, encoded_image, figure_caption)
            unified_output: Whether to output a unified JSON

        Returns:
            dict: QC results
        """
        logger.info("*** QC PIPELINE STARTED ***")

        # Get figures from zip structure or figure_data
        figures = []
        if figure_data:
            figures = figure_data
        elif hasattr(zip_structure, "figures"):
            # Transform zip_structure.figures to expected format
            for fig in zip_structure.figures:
                if hasattr(fig, "figure_label") and hasattr(fig, "figure_caption"):
                    encoded_image = (
                        ""  # Would need to be populated from the actual image
                    )
                    figures.append(
                        (fig.figure_label, encoded_image, fig.figure_caption)
                    )

        logger.info(f"QC pipeline processing {len(figures)} figures")

        # Reset results
        self.qc_results = {"figures": {}}

        # Track status
        num_processed = 0
        pipeline_status = "success"

        # Run tests on all figures
        for figure_label, encoded_image, figure_caption in figures:
            logger.info(f"Processing {figure_label}")
            figure_id = figure_label.replace(" ", "_").lower()

            # Run each test on the figure
            for test_name, test_analyzer in self.tests.items():
                try:
                    # Call the appropriate analyze method based on the test type
                    if isinstance(test_analyzer, (PanelQCAnalyzer, FigureQCAnalyzer)):
                        # Panel and figure tests need figure data
                        passed, result = test_analyzer.analyze_figure(
                            figure_label, encoded_image, figure_caption
                        )

                        # Add results to output
                        self.add_qc_result(figure_id, test_name, passed, result)
                    elif isinstance(test_analyzer, ManuscriptQCAnalyzer):
                        # Document tests need manuscript structure
                        # Only run once for the first figure to avoid duplicates
                        if figure_id == figures[0][0].replace(" ", "_").lower():
                            passed, result = test_analyzer.analyze_manuscript(
                                zip_structure
                            )
                            # Add manuscript results to a special section
                            if "manuscript" not in self.qc_results:
                                self.qc_results["manuscript"] = {}
                            self.qc_results["manuscript"][test_name] = {
                                "passed": passed,
                                "result": result,
                            }
                    else:
                        # Fallback for other analyzer types
                        logger.warning(
                            f"Unknown analyzer type for {test_name}, skipping"
                        )

                except Exception as e:
                    logger.error(
                        f"Error running test {test_name} on {figure_label}: {str(e)}"
                    )
                    pipeline_status = "error"

            num_processed += 1

        # Build the output structure
        if unified_output:
            output = {
                "qc_version": self.config.get(
                    "qc_version", "1.0.0"
                ),  # Version of the QC pipeline
                "figures": self.qc_results["figures"],
                "qc_test_metadata": {},
                "status": pipeline_status,
            }

            # Add manuscript results if any
            if "manuscript" in self.qc_results:
                output["manuscript"] = self.qc_results["manuscript"]

            # Add metadata for each test
            for test_name, test_analyzer in self.tests.items():
                output["qc_test_metadata"][test_name] = {
                    "name": test_name,
                    "description": "",
                    "permalink": "",
                }

                # Add metadata from the test analyzer if available
                if hasattr(test_analyzer, "metadata") and test_analyzer.metadata:
                    # Handle metadata as dictionary
                    if isinstance(test_analyzer.metadata, dict):
                        # Flatten and add specific fields
                        for key, value in test_analyzer.metadata.items():
                            output["qc_test_metadata"][test_name][key] = value
                    else:
                        # Add specific fields if they exist
                        if hasattr(test_analyzer.metadata, "name"):
                            output["qc_test_metadata"][test_name][
                                "name"
                            ] = test_analyzer.metadata.name
                        if hasattr(test_analyzer.metadata, "description"):
                            output["qc_test_metadata"][test_name][
                                "description"
                            ] = test_analyzer.metadata.description
                        if hasattr(test_analyzer.metadata, "permalink"):
                            output["qc_test_metadata"][test_name][
                                "permalink"
                            ] = test_analyzer.metadata.permalink
                        if hasattr(test_analyzer.metadata, "version"):
                            output["qc_test_metadata"][test_name][
                                "version"
                            ] = test_analyzer.metadata.version
        else:
            output = self.qc_results
            output["status"] = pipeline_status

        # Report pipeline status
        if num_processed == 0:
            output["status"] = "unknown"
        logger.info(f"QC pipeline completed with status: {output['status']}")
        logger.info(f"Processed {num_processed} figures")

        return output

    def add_qc_result(self, figure_id, test_name, passed, result):
        """Add a QC result to the figure."""
        # Initialize figure entry if it doesn't exist
        if figure_id not in self.qc_results["figures"]:
            self.qc_results["figures"][figure_id] = {"panels": []}

        # Extract outputs from result
        outputs = []
        if hasattr(result, "outputs"):
            outputs = result.outputs
        elif isinstance(result, dict):
            if "result" in result and hasattr(result["result"], "outputs"):
                outputs = result["result"].outputs
            else:
                outputs = result.get("outputs", [])

        # Process each panel in the outputs
        for panel in outputs:
            # Find or create panel entry
            panel_label = None
            if hasattr(panel, "panel_label"):
                panel_label = panel.panel_label
            elif isinstance(panel, dict) and "panel_label" in panel:
                panel_label = panel["panel_label"]

            if not panel_label:
                continue

            # Look for existing panel
            panel_entry = None
            for p in self.qc_results["figures"][figure_id]["panels"]:
                if p.get("panel_label") == panel_label:
                    panel_entry = p
                    break

            # Create new panel if not found
            if not panel_entry:
                panel_entry = {"panel_label": panel_label, "qc_tests": []}
                self.qc_results["figures"][figure_id]["panels"].append(panel_entry)

            # Create test entry with only essential fields
            test_obj = {
                "test_name": test_name,
                "passed": passed,
            }

            # Add model output - convert to dict if needed
            if hasattr(panel, "model_dump"):
                # For Pydantic v2+
                panel_dict = panel.model_dump()
                # Convert enum values to strings
                for key, value in panel_dict.items():
                    if hasattr(value, "value"):  # Check if it's an enum
                        panel_dict[key] = value.value
                test_obj["model_output"] = panel_dict
            elif hasattr(panel, "dict"):
                # For Pydantic v1
                panel_dict = panel.dict()
                # Convert enum values to strings
                for key, value in panel_dict.items():
                    if hasattr(value, "value"):  # Check if it's an enum
                        panel_dict[key] = value.value
                test_obj["model_output"] = panel_dict
            elif isinstance(panel, dict):
                # Already a dict
                test_obj["model_output"] = panel

            # Add test to panel
            panel_entry["qc_tests"].append(test_obj)
