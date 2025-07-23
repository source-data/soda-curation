#!/usr/bin/env python3
"""
Main module for running the QC pipeline.
This module loads configuration and runs the QC tests specified in the config.
"""

import argparse
import enum
import importlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

from ..config import ConfigurationLoader
from ..data_storage import load_figure_data, load_zip_structure
from .prompt_registry import registry
from .qc_pipeline import QCPipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG to see debug logs
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


class EnumAwareJSONEncoder(json.JSONEncoder):
    """JSON Encoder that can handle enum values."""

    def default(self, obj):
        if isinstance(obj, enum.Enum):
            return obj.value
        return super().default(obj)


def main():
    """Run the main function for the QC pipeline."""
    parser = argparse.ArgumentParser(description="Run the QC pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="config.qc.yaml",
        help="Path to config file (default: config.qc.yaml)",
    )
    parser.add_argument(
        "--extract-dir",
        type=str,
        default="data/extract",
        help="Path to extract directory (default: data/extract)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/qc_results.json",
        help="Path to output file (default: data/output/qc_results.json)",
    )
    parser.add_argument(
        "--word-document",
        type=str,
        help="Path to Word document for manuscript analysis",
    )
    parser.add_argument(
        "--figure-data",
        type=str,
        help="Path to figure data JSON file (optional, will be generated if not provided)",
    )
    parser.add_argument(
        "--zip-structure",
        type=str,
        help="Path to zip structure pickle file (optional, will be generated if not provided)",
    )
    args = parser.parse_args()

    # Log the paths
    logger.info("Using figure data from: %s", args.figure_data)
    logger.info("Using zip structure from: %s", args.zip_structure)
    logger.info("Using config from: %s", args.config)

    # Load configuration
    try:
        # Try loading with ConfigurationLoader if it has the right method
        config = ConfigurationLoader.load_from_file(args.config)
    except (AttributeError, ImportError):
        # Fall back to direct YAML loading
        import yaml

        with open(args.config, "r") as config_file:
            config = yaml.safe_load(config_file)

    # Load figure data and zip structure
    figure_data = None
    zip_structure = None

    if args.figure_data:
        figure_data = load_figure_data(args.figure_data)

    if args.zip_structure:
        zip_structure = load_zip_structure(args.zip_structure)

    # Check if both figure_data and zip_structure are loaded
    if not figure_data or not zip_structure:
        logger.error("Both figure_data and zip_structure are required")
        return

    # Log the number of figures
    logger.info("Loaded %d figures from saved data", len(figure_data))

    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure Word document path is set for manuscript analysis
    word_file_path = (
        args.word_document
        or "data/output/240429 EMM  DUSP6 targeting abrogates HER3 R2.docx"
    )
    if zip_structure and Path(word_file_path).exists():
        zip_structure._full_docx = word_file_path
        logger.info("Word document path set to: %s", word_file_path)
    elif word_file_path:
        logger.warning("Word document not found at: %s", word_file_path)

    # Run the QC pipeline
    logger.info("Starting QC pipeline")
    qc_pipeline = QCPipeline(config, args.extract_dir)

    logger.info("Running QC pipeline")
    qc_results = qc_pipeline.run(zip_structure, figure_data)

    # Log the results structure
    logger.info("QC results structure: %s", qc_results.keys())

    # Add permalinks from the prompt registry to the output
    if "qc_check_metadata" in qc_results:
        for test_name, metadata in qc_results["qc_check_metadata"].items():
            try:
                registry_metadata = registry.get_prompt_metadata(test_name)
                if registry_metadata and hasattr(registry_metadata, "permalink"):
                    metadata["permalink"] = registry_metadata.permalink
            except Exception as e:
                logger.warning(f"Failed to get metadata for {test_name}: {e}")

    # Log the number of figures and panels
    if "figures" in qc_results:
        logger.info("Number of figures: %d", len(qc_results["figures"]))
        for figure_id, figure in qc_results["figures"].items():
            logger.info(
                "Figure %s has %d panels", figure_id, len(figure.get("panels", []))
            )

    # Save the results to a JSON file
    with open(args.output, "w") as f:
        json.dump(qc_results, f, indent=2, cls=EnumAwareJSONEncoder)

    logger.info("QC results saved to %s", args.output)

    # Report final status
    if "status" in qc_results:
        logger.info("QC pipeline completed with status: %s", qc_results["status"])
    else:
        logger.info("QC pipeline completed with status: unknown")

    if "figures" in qc_results:
        logger.info("Processed %d figures", len(qc_results["figures"]))
    else:
        logger.info("Processed 0 figures")


if __name__ == "__main__":
    main()
