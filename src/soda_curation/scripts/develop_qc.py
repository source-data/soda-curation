"""
Development script for QC pipeline that uses saved data.
This allows quick iteration without running the full pipeline.
"""

import argparse
import logging
from pathlib import Path

from ..config import ConfigurationLoader
from ..data_storage import load_figure_data, load_zip_structure
from ..qc.qc_pipeline import QCPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Develop QC pipeline with saved data")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument(
        "--figure-data", required=True, help="Path to saved figure data"
    )
    parser.add_argument(
        "--zip-structure", required=True, help="Path to saved zip structure"
    )
    parser.add_argument(
        "--extract-dir", required=True, help="Path to extraction directory"
    )
    args = parser.parse_args()

    # Load configuration
    config_loader = ConfigurationLoader(args.config)
    qc_config = config_loader.config.get("qc", {})

    # Load saved data
    figure_data = load_figure_data(args.figure_data)
    zip_structure = load_zip_structure(args.zip_structure)

    if not figure_data or not zip_structure:
        logger.error("Failed to load required data")
        return

    # Run QC pipeline
    logger.info("Starting QC pipeline development run")
    qc_pipeline = QCPipeline(qc_config, Path(args.extract_dir))
    qc_results = qc_pipeline.run(zip_structure, figure_data)

    # Display results
    logger.info(
        f"QC pipeline completed with status: {qc_results.get('qc_status', 'unknown')}"
    )
    logger.info(f"Processed {len(qc_results.get('figure_results', []))} figures")

    # Example to print specific results
    for i, figure_result in enumerate(
        qc_results.get("figure_results", [])[:3]
    ):  # Show first 3
        logger.info(
            f"Figure {figure_result['figure_label']} QC status: {figure_result['qc_status']}"
        )

    logger.info("QC development run completed")


if __name__ == "__main__":
    main()
