"""Main entry point for SODA curation pipeline."""

import json
import logging
from typing import Optional

from .config import load_config
from .logging_config import setup_logging
from ._main_utils import (
    validate_paths,
    setup_extract_dir,
    write_output,
    cleanup_extract_dir
)
from .pipeline.manuscript_structure.manuscript_xml_parser import XMLStructureExtractor
from .pipeline.manuscript_structure.manuscript_structure import CustomJSONEncoder

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
    
    # Load configuration and setup logging
    config = load_config(config_path)
    setup_logging(config)
    logger.info("Starting SODA curation pipeline")
    
    try:
        # Setup extraction directory
        extract_dir = setup_extract_dir(zip_path)
        extract_dir.mkdir(exist_ok=True)
        
        try:
            # Extract manuscript structure
            extractor = XMLStructureExtractor(zip_path, str(extract_dir))
            structure = extractor.extract_structure()
            
            # Add AI configuration
            structure.ai_provider = config["ai"]
            structure.ai_config = {
                "provider": config["ai"],
                "model": config[config["ai"]]["model"],
                "temperature": config[config["ai"]].get("temperature", 0.5),
                "top_p": config[config["ai"]].get("top_p", 1.0)
            }
            
            # Add provider-specific parameters
            if config["ai"] == "openai":
                structure.ai_config.update({
                    "max_tokens": config["openai"].get("max_tokens", 0)
                })
            elif config["ai"] == "anthropic":
                structure.ai_config.update({
                    "max_tokens_to_sample": config["anthropic"].get("max_tokens_to_sample", 0),
                    "top_k": config["anthropic"].get("top_k", 0)
                })
            
            # Convert to JSON
            output_json = json.dumps(
                structure,
                cls=CustomJSONEncoder,
                ensure_ascii=False,
                indent=2
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
    
    parser = argparse.ArgumentParser(description="Process a ZIP file using SODA curation")
    parser.add_argument("--zip", required=True, help="Path to the input ZIP file")
    parser.add_argument("--config", required=True, help="Path to the configuration file")
    parser.add_argument("--output", help="Path to the output JSON file")
    
    args = parser.parse_args()
    
    output_json = main(args.zip, args.config, args.output)
    if not args.output:
        print(output_json)