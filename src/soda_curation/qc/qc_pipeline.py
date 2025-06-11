"""QC Pipeline for SODA curation."""

import importlib
import logging
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

from ..pipeline.manuscript_structure.manuscript_structure import ZipStructure
from .data_types import QCPipelineResult, QCResult

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
        logger.info("Initialized QC Pipeline")

    def _initialize_tests(self) -> Dict:
        """Initialize QC test analyzers based on configuration."""
        tests = {}

        # Only process if pipeline key exists in config
        if "pipeline" not in self.config:
            logger.warning("No 'pipeline' section found in config")
            return tests

        # Iterate through pipeline items to find test modules
        for test_name, test_config in self.config["pipeline"].items():
            # Try to import the corresponding module from qc_tests
            try:
                # Convert snake_case to CamelCase for class name
                class_name = (
                    "".join(word.capitalize() for word in test_name.split("_"))
                    + "Analyzer"
                )
                module_name = f".qc_tests.{test_name}"

                # Check if module exists
                try:
                    module = importlib.import_module(
                        module_name, package="soda_curation.qc"
                    )

                    # Get the analyzer class
                    if hasattr(module, class_name):
                        analyzer_class = getattr(module, class_name)
                        tests[test_name] = analyzer_class(self.config)
                        logger.info(f"Loaded test module: {test_name}")
                    else:
                        logger.warning(
                            f"Module {test_name} found but analyzer class {class_name} not found"
                        )
                except ModuleNotFoundError:
                    logger.warning(
                        f"Test module {test_name} specified in config but not found in qc_tests"
                    )

            except Exception as e:
                logger.error(f"Error loading test module {test_name}: {str(e)}")

        return tests

    def run(
        self,
        zip_structure: ZipStructure,
        figure_data: Optional[List[Tuple[str, str, str]]] = None,
    ) -> dict:
        """
        Run QC pipeline on the given zip structure.

        Args:
            zip_structure: Structure containing manuscript and figure data
            figure_data: List of tuples containing (figure_label, base64_encoded_image, figure_caption)

        Returns:
            QC results data
        """
        logger.info("*** QC PIPELINE STARTED ***")

        # Log information about the figures we received
        figures_count = (
            len(zip_structure.figures) if hasattr(zip_structure, "figures") else 0
        )
        figure_data_count = len(figure_data) if figure_data else 0

        logger.info(
            f"QC pipeline processing {figures_count} figures from zip_structure"
        )
        logger.info(
            f"QC pipeline received {figure_data_count} figure images for analysis"
        )

        # Process each figure if figure_data is provided
        figure_qc_results = []
        if figure_data:
            for figure_label, encoded_image, figure_caption in figure_data:
                # Create a result object for this figure
                figure_result = QCResult(
                    figure_label=figure_label,
                    qc_checks={},
                )

                # Run configured tests on this figure
                all_tests_passed = True

                # Run each configured test
                for test_name, test_analyzer in self.tests.items():
                    try:
                        # Check if the analyzer has the analyze_figure method
                        if hasattr(test_analyzer, "analyze_figure"):
                            passed, result = test_analyzer.analyze_figure(
                                figure_label, encoded_image, figure_caption
                            )
                            figure_result.qc_checks[test_name] = {
                                "passed": passed,
                                "result": asdict(result)
                                if hasattr(result, "__dataclass_fields__")
                                else result,
                            }
                            if not passed:
                                all_tests_passed = False
                        else:
                            logger.warning(
                                f"Test {test_name} does not have analyze_figure method"
                            )
                    except Exception as e:
                        logger.error(
                            f"Error in {test_name} for figure {figure_label}: {str(e)}"
                        )
                        figure_result.qc_checks[test_name] = {
                            "passed": False,
                            "error": str(e),
                        }
                        all_tests_passed = False

                # Set overall figure status
                figure_result.qc_status = "passed" if all_tests_passed else "failed"
                figure_qc_results.append(figure_result)
                logger.info(
                    f"Completed QC for figure {figure_label}: {figure_result.qc_status}"
                )

        # Create overall QC results
        qc_results = QCPipelineResult(
            qc_version="0.1.0",
            qc_status="passed"
            if all(r.qc_status == "passed" for r in figure_qc_results)
            else "failed",
            figures_processed=figure_data_count,
            figure_results=figure_qc_results,
        )

        logger.info("*** QC PIPELINE COMPLETED ***")
        # Convert all dataclasses to dictionaries before returning
        return asdict(qc_results)
