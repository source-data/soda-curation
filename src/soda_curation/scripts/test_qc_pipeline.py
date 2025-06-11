#!/usr/bin/env python3
"""
Test script for the QC pipeline.
This script loads saved figure data and zip structure, then runs the QC pipeline.
"""

import argparse
import json
import logging
import os
from dataclasses import asdict
from pathlib import Path

from ..config import ConfigurationLoader
from ..data_storage import load_figure_data, load_zip_structure
from ..qc.qc_pipeline import QCPipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Custom JSON encoder to handle dataclasses
class DataclassJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "__dataclass_fields__"):
            return asdict(obj)
        return super().default(obj)


def main():
    """Run QC pipeline tests with saved data."""
    parser = argparse.ArgumentParser(description="Test QC pipeline with saved data")
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
        default="data/development/qc_results.json",
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
    logger.info("Starting QC pipeline development run")
    qc_pipeline = QCPipeline(config, extract_dir)

    logger.info("Running QC pipeline")
    qc_results = qc_pipeline.run(zip_structure, figure_data)

    # Save results - use custom encoder for dataclasses
    with open(output_path, "w") as f:
        json.dump(qc_results, f, indent=2, cls=DataclassJSONEncoder)
    logger.info(f"QC results saved to {output_path}")

    # Print summary
    logger.info(
        f"QC pipeline completed with status: {qc_results.get('qc_status', 'unknown')}"
    )
    logger.info(f"Processed {qc_results.get('figures_processed', 0)} figures")

    # Show some details of the first figure result
    if qc_results.get("figure_results"):
        # Access figure_results as list of dictionaries
        figure_results = qc_results["figure_results"]
        for i, figure_result in enumerate(figure_results[:3]):  # Show first 3
            # Access properties directly from dictionary
            figure_label = figure_result["figure_label"]
            qc_status = figure_result["qc_status"]
            logger.info(f"Figure {figure_label} QC status: {qc_status}")

            # Check if the stats test was run for this figure
            qc_checks = figure_result.get("qc_checks", {})
            if "stats_test" in qc_checks:
                stats_test = qc_checks["stats_test"]
                logger.info(f"  - Stats test passed: {stats_test.get('passed')}")

                # Show details of panels if available
                if "result" in stats_test and "outputs" in stats_test["result"]:
                    outputs = stats_test["result"]["outputs"]
                    logger.info(f"  - Analyzed {len(outputs)} panels")

                    # Show details of first 2 panels (if any)
                    for j, panel in enumerate(outputs[:2] if outputs else []):
                        logger.info(
                            f"    - Panel {panel['panel_label']}: "
                            f"is_a_plot={panel['is_a_plot']}, "
                            f"test_needed={panel['statistical_test_needed']}, "
                            f"test_mentioned={panel['statistical_test_mentioned']}"
                        )

    logger.info("QC development run completed")


if __name__ == "__main__":
    main()
