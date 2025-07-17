"""QC Pipeline for SODA curation."""

import importlib
import logging
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

from ..pipeline.manuscript_structure.manuscript_structure import ZipStructure
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
        unified_output: bool = True,
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
        unified_figures = []
        if figure_data:
            for figure_label, encoded_image, figure_caption in figure_data:
                # Create a result object for this figure
                figure_result = QCResult(
                    figure_label=figure_label,
                    qc_checks={},
                )

                # For unified output, collect per-panel results from all tests
                panel_results = {}  # panel_label -> {panel info}

                all_tests_passed = True

                # Prepare to collect figure-level QC test results
                figure_level_tests = []

                def to_bool(val):
                    if isinstance(val, bool):
                        return val
                    if isinstance(val, str):
                        return val.strip().lower() in [
                            "yes",
                            "true",
                            "1",
                        ]
                    if isinstance(val, (int, float)):
                        return bool(val)
                    return False

                for test_name, test_analyzer in self.tests.items():
                    # Determine test level from config
                    test_level = (
                        self.config["pipeline"].get(test_name, {}).get("level", "panel")
                    )
                    try:
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

                            outputs = None
                            if hasattr(result, "outputs"):
                                outputs = result.outputs
                            elif isinstance(result, dict):
                                outputs = result.get("outputs", [])
                            logger.debug(
                                f"[QC PIPELINE] {test_name} outputs for figure {figure_label}: {outputs}"
                            )
                            if outputs is None:
                                logger.warning(
                                    "No outputs found for test %s on figure %s",
                                    test_name,
                                    figure_label,
                                )
                                continue

                            if test_level == "figure":
                                # Only one output expected for figure-level tests
                                for output in outputs:
                                    # Map test output to unified test result (same fields as panel)
                                    get_attr = (
                                        lambda obj, key: getattr(obj, key, None)
                                        if hasattr(obj, key)
                                        else obj.get(key)
                                        if isinstance(obj, dict)
                                        else None
                                    )
                                    test_obj = {
                                        "test_name": test_name,
                                        "passed": None,
                                        "comments": None,
                                        "model_output": None,
                                        "is_a_plot": False,
                                        "test_needed": False,
                                    }
                                    if test_name == "replicates_defined":
                                        involves_replicates = get_attr(
                                            output, "involves_replicates"
                                        )
                                        number_of_replicates = (
                                            get_attr(output, "number_of_replicates")
                                            or []
                                        )
                                        type_of_replicates = (
                                            get_attr(output, "type_of_replicates") or []
                                        )
                                        test_needed = involves_replicates == "yes"
                                        test_passed = (
                                            test_needed
                                            and number_of_replicates
                                            and type_of_replicates
                                            and all(
                                                x != "unknown"
                                                for x in number_of_replicates
                                            )
                                            and all(
                                                x != "unknown"
                                                for x in type_of_replicates
                                            )
                                        )
                                        test_obj["passed"] = (
                                            bool(test_passed) if test_needed else False
                                        )
                                        test_obj["comments"] = ""
                                        test_obj["model_output"] = str(output)
                                        test_obj["is_a_plot"] = False
                                        test_obj["test_needed"] = bool(test_needed)
                                    # Add more figure-level test logic here as needed
                                    figure_level_tests.append(test_obj)
                            else:
                                # Panel-level test (default)
                                for panel in outputs:
                                    logger.debug(f"[QC PIPELINE] Panel object: {panel}")
                                    panel_label = None
                                    if hasattr(panel, "panel_label"):
                                        panel_label = getattr(
                                            panel, "panel_label", None
                                        )
                                        get_attr = lambda obj, key: getattr(
                                            obj, key, None
                                        )
                                    elif isinstance(panel, dict):
                                        panel_label = panel.get("panel_label")
                                        get_attr = lambda obj, key: obj.get(key)
                                    else:
                                        logger.warning(
                                            f"Panel is not a recognized type: {type(panel)}"
                                        )
                                        continue
                                    if not panel_label:
                                        logger.warning(
                                            f"Panel missing label in test {test_name} for figure {figure_label}: {panel}"
                                        )
                                        continue
                                    if panel_label not in panel_results:
                                        panel_results[panel_label] = {
                                            "panel_label": panel_label,
                                            "qc_tests": [],
                                        }
                                    # Map test output to unified test result
                                    test_obj = {
                                        "test_name": test_name,
                                        "passed": None,
                                        "comments": None,
                                        "model_output": None,
                                        "is_a_plot": False,
                                        "test_needed": False,
                                    }

                                    if test_name == "stats_test":
                                        test_passed = (
                                            get_attr(
                                                panel, "statistical_test_mentioned"
                                            )
                                            == "yes"
                                        )
                                        comments = (
                                            get_attr(panel, "from_the_caption") or ""
                                        )
                                        test_needed = to_bool(
                                            get_attr(panel, "statistical_test_needed")
                                        )
                                        if not test_passed:
                                            comments = (
                                                get_attr(
                                                    panel, "justify_why_test_is_missing"
                                                )
                                                or comments
                                            )
                                        # Only pass if test is needed and passed
                                        test_obj["passed"] = (
                                            bool(test_passed) if test_needed else False
                                        )
                                        test_obj["comments"] = comments or ""
                                        test_obj["model_output"] = str(panel)
                                        test_obj["is_a_plot"] = to_bool(
                                            get_attr(panel, "is_a_plot")
                                        )
                                        test_obj["test_needed"] = bool(test_needed)
                                    elif test_name == "stats_significance_level":
                                        symbols = (
                                            get_attr(
                                                panel,
                                                "significance_level_symbols_on_image",
                                            )
                                            or []
                                        )
                                        defined = (
                                            get_attr(
                                                panel,
                                                "significance_level_symbols_defined_in_caption",
                                            )
                                            or []
                                        )
                                        test_passed = (
                                            all(s == "yes" for s in defined)
                                            if symbols
                                            else True
                                        )
                                        comments = get_attr(panel, "from_the_caption")
                                        if isinstance(comments, list):
                                            comments = ", ".join(comments)
                                        test_needed = (
                                            bool(symbols) if symbols else False
                                        )
                                        # Only pass if test is needed and passed
                                        test_obj["passed"] = (
                                            bool(test_passed) if test_needed else False
                                        )
                                        test_obj["comments"] = comments or ""
                                        test_obj["model_output"] = str(panel)
                                        test_obj["is_a_plot"] = to_bool(
                                            get_attr(panel, "is_a_plot")
                                        )
                                        test_obj["test_needed"] = bool(test_needed)
                                    elif test_name == "plot_axis_units":
                                        # For plot_axis_units, pass if all axes are yes or not needed
                                        units = get_attr(panel, "units_provided") or []
                                        justifications = (
                                            get_attr(
                                                panel, "justify_why_units_are_missing"
                                            )
                                            or []
                                        )
                                        # Determine pass/fail
                                        test_passed = (
                                            all(
                                                getattr(u, "answer", None)
                                                in ("yes", "not needed")
                                                for u in units
                                            )
                                            if units
                                            else True
                                        )
                                        # Comments: join all justifications if any
                                        comments = (
                                            "; ".join(
                                                getattr(j, "justification", "")
                                                for j in justifications
                                            )
                                            if justifications
                                            else ""
                                        )
                                        test_needed = any(
                                            getattr(u, "answer", None) == "no"
                                            for u in units
                                        )
                                        # Only pass if test is needed and passed
                                        test_obj["passed"] = (
                                            bool(test_passed) if test_needed else False
                                        )
                                        test_obj["comments"] = comments
                                        test_obj["model_output"] = str(panel)
                                        test_obj["is_a_plot"] = to_bool(
                                            get_attr(panel, "is_a_plot")
                                        )
                                        test_obj["test_needed"] = bool(test_needed)
                                    elif test_name == "replicates_defined":
                                        involves_replicates = get_attr(
                                            panel, "involves_replicates"
                                        )
                                        number_of_replicates = (
                                            get_attr(panel, "number_of_replicates")
                                            or []
                                        )
                                        type_of_replicates = (
                                            get_attr(panel, "type_of_replicates") or []
                                        )
                                        # Test is needed if involves_replicates is "yes"
                                        test_needed = involves_replicates == "yes"
                                        # Pass if number and type are both known and not unknown
                                        test_passed = (
                                            test_needed
                                            and number_of_replicates
                                            and type_of_replicates
                                            and all(
                                                x != "unknown"
                                                for x in number_of_replicates
                                            )
                                            and all(
                                                x != "unknown"
                                                for x in type_of_replicates
                                            )
                                        )
                                        test_obj["passed"] = (
                                            bool(test_passed) if test_needed else False
                                        )
                                        test_obj["comments"] = ""
                                        test_obj["model_output"] = str(panel)
                                        test_obj["is_a_plot"] = False
                                        test_obj["test_needed"] = bool(test_needed)
                                    # Add more tests here as needed
                                    panel_results[panel_label]["qc_tests"].append(
                                        test_obj
                                    )
                        else:
                            logger.warning(
                                "Test %s does not have analyze_figure method", test_name
                            )
                    except Exception as e:
                        logger.error(
                            "Error in %s for figure %s: %s",
                            test_name,
                            figure_label,
                            str(e),
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

                # --- Unified output: build figure dict ---
                if unified_output:
                    figure_dict = {
                        "figure_label": figure_label,
                        "figure_caption": figure_caption,
                        "panels": list(panel_results.values()),
                    }
                    if figure_level_tests:
                        figure_dict["figure_qc_tests"] = figure_level_tests
                    unified_figures.append(figure_dict)

        qc_version = self.config.get("qc_version", "0.3.0")
        if unified_output:
            output = {"qc_version": qc_version, "figures": unified_figures}
            # Add qc_test_metadata from config if present
            if "qc_test_metadata" in self.config:
                output["qc_test_metadata"] = self.config["qc_test_metadata"]
            return output

        # Legacy output
        qc_results = QCPipelineResult(
            qc_version=qc_version,
            qc_status="passed"
            if all(r.qc_status == "passed" for r in figure_qc_results)
            else "failed",
            figures_processed=figure_data_count,
            figure_results=figure_qc_results,
        )
        result_dict = asdict(qc_results)
        # Add qc_test_metadata from config if present
        if "qc_test_metadata" in self.config:
            result_dict["qc_test_metadata"] = self.config["qc_test_metadata"]
        logger.info("*** QC PIPELINE COMPLETED ***")
        return result_dict
