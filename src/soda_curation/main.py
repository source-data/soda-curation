"""Main entry point for SODA curation pipeline."""

import json
import logging
from typing import Optional

from ._main_utils import (
    cleanup_extract_dir,
    setup_extract_dir,
    validate_paths,
    write_output,
)
from .config import ConfigurationLoader
from .logging_config import setup_logging
from .pipeline.manuscript_structure.manuscript_structure import CustomJSONEncoder
from .pipeline.manuscript_structure.manuscript_xml_parser import XMLStructureExtractor

logger = logging.getLogger(__name__)


def main(zip_path: str, config_path: str, output_path: Optional[str] = None) -> str:
    """
    Main entry point for SODA curation pipeline.

    Args:
        zip_path: Path to input ZIP file
        config_path: Path to configuration file
        output_path: Optional path to output JSON file

    Returns:
        JSON string containing processing results

    Raises:
        Various exceptions for validation and processing errors
    """
    # Validate inputs
    validate_paths(zip_path, config_path, output_path)

    # Load configuration
    config_loader = ConfigurationLoader(config_path)

    # Setup logging based on environment
    setup_logging(config_loader.config)
    logger.info("Starting SODA curation pipeline")

    try:
        # Setup extraction directory
        extract_dir = setup_extract_dir(zip_path)
        extract_dir.mkdir(exist_ok=True)

        try:
            # Extract manuscript structure (first pipeline step)
            # manuscript_config = config_loader.get_pipeline_config(
            #     PipelineStep.MANUSCRIPT_STRUCTURE
            # )
            extractor = XMLStructureExtractor(zip_path, str(extract_dir))
            zip_structure = extractor.extract_structure()
            # Update total costs before returning results
            # zip_structure.update_total_cost()

            # Convert to JSON
            output_json = json.dumps(
                zip_structure, cls=CustomJSONEncoder, ensure_ascii=False, indent=2
            )

            # Write to file if output path provided
            if output_path:
                write_output(output_json, output_path)

            return output_json

        finally:
            cleanup_extract_dir(extract_dir)

    except Exception as e:
        logger.exception(f"Pipeline failed: {str(e)}")
        return json.dumps({"error": str(e)})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process a ZIP file using SODA curation"
    )
    parser.add_argument("--zip", required=True, help="Path to the input ZIP file")
    parser.add_argument(
        "--config", required=True, help="Path to the configuration file"
    )
    parser.add_argument("--output", help="Path to the output JSON file")

    args = parser.parse_args()

    output_json = main(args.zip, args.config, args.output)
    if not args.output:
        print(output_json)
