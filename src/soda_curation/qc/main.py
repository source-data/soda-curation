#!/usr/bin/env python3
"""
Main module for running the QC pipeline.
This module loads configuration and runs the QC tests specified in the config.
"""

import argparse
import importlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

from ..config import ConfigurationLoader
from ..data_storage import load_figure_data, load_zip_structure
from .qc_pipeline import QCPipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG to see debug logs
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_test_modules(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dynamically load QC test modules based on configuration.

    Args:
        config: Configuration dictionary with QC test settings

    Returns:
        Dictionary of initialized test modules
    """
    test_modules = {}

    # Only process if pipeline key exists in config
    if "pipeline" not in config:
        logger.warning("No 'pipeline' section found in config")
        return test_modules

    # Iterate through pipeline items to find test modules
    for test_name, test_config in config["pipeline"].items():
        # Try to import the corresponding module from qc_tests
        try:
            # Convert snake_case to CamelCase for class name
            class_name = (
                "".join(word.capitalize() for word in test_name.split("_")) + "Analyzer"
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
                    test_modules[test_name] = analyzer_class(config)
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

    return test_modules


def main():
    """Run QC pipeline with specified configuration."""
    parser = argparse.ArgumentParser(description="Run QC pipeline")
    parser.add_argument(
        "--config", default="config.qc.yaml", help="Path to QC configuration file"
    )
    parser.add_argument(
        "--figure-data",
        default="data/development/figure_data.json",
        help="Path to saved figure data",
    )
    parser.add_argument(
        "--zip-structure",
        default="data/development/zip_structure.pickle",
        help="Path to saved zip structure",
    )
    parser.add_argument(
        "--extract-dir", default="data/extract", help="Path to extraction directory"
    )
    parser.add_argument(
        "--output",
        default="data/output/qc_results.json",
        help="Path to save QC results",
    )
    args = parser.parse_args()

    # Get absolute paths
    cwd = Path.cwd()
    figure_data_path = Path(args.figure_data)
    zip_structure_path = Path(args.zip_structure)
    extract_dir = Path(args.extract_dir)
    config_path = Path(args.config)
    output_path = Path(args.output)

    # Make absolute if not already
    if not figure_data_path.is_absolute():
        figure_data_path = cwd / figure_data_path
    if not zip_structure_path.is_absolute():
        zip_structure_path = cwd / zip_structure_path
    if not extract_dir.is_absolute():
        extract_dir = cwd / extract_dir
    if not config_path.is_absolute():
        config_path = cwd / config_path
    if not output_path.is_absolute():
        output_path = cwd / output_path

    logger.info(f"Using figure data from: {figure_data_path}")
    logger.info(f"Using zip structure from: {zip_structure_path}")
    logger.info(f"Using config from: {config_path}")

    # Make sure output directory exists
    os.makedirs(output_path.parent, exist_ok=True)

    # Load configuration
    config_loader = ConfigurationLoader(str(config_path))
    config = config_loader.config

    # Load saved data
    figure_data = load_figure_data(str(figure_data_path))
    zip_structure = load_zip_structure(str(zip_structure_path))

    if not figure_data:
        logger.error(f"Failed to load figure data from {figure_data_path}")
        return
    if not zip_structure:
        logger.error(f"Failed to load zip structure from {zip_structure_path}")
        return

    logger.info(f"Loaded {len(figure_data)} figures from saved data")

    # Run QC pipeline
    logger.info("Starting QC pipeline")
    qc_pipeline = QCPipeline(config, extract_dir)

    logger.info("Running QC pipeline")
    qc_results = qc_pipeline.run(zip_structure, figure_data, unified_output=True)

    # Save results to JSON file
    with open(output_path, "w") as f:
        json.dump(qc_results, f, indent=2)
    logger.info(f"QC results saved to {output_path}")

    # Print summary
    logger.info(
        f"QC pipeline completed with status: {qc_results.get('qc_status', 'unknown')}"
    )
    logger.info(f"Processed {qc_results.get('figures_processed', 0)} figures")

    # Show some details of the first few figure results
    if qc_results.get("figure_results"):
        figure_results = qc_results["figure_results"]
        for i, figure_result in enumerate(figure_results):  # Show first 3
            figure_label = figure_result["figure_label"]
            qc_status = figure_result["qc_status"]
            logger.info(f"Figure {figure_label} QC status: {qc_status}")

            # Log summary of each QC check for this figure
            qc_checks = figure_result.get("qc_checks", {})
            for check_name, check_result in qc_checks.items():
                passed = check_result.get("passed", False)
                logger.info(f"  - {check_name}: {'Passed' if passed else 'Failed'}")


if __name__ == "__main__":
    main()
