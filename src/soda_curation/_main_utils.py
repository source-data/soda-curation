"""Utility functions supporting the main entry point."""

import logging
import os
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

def validate_paths(zip_path: str, config_path: str, output_path: Optional[str] = None) -> None:
    """
    Validate input paths and ensure output directory exists.
    
    Args:
        zip_path: Path to input ZIP file
        config_path: Path to configuration file
        output_path: Optional path to output JSON file
        
    Raises:
        ValueError: If required paths are not provided
        FileNotFoundError: If input files don't exist
    """
    if not zip_path:
        raise ValueError("ZIP path must be provided")
    if not config_path:
        raise ValueError("config path must be provided")
        
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"ZIP file {zip_path} does not exist")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} does not exist")
        
    if output_path:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Ensured output directory exists: {output_dir}")

def setup_extract_dir(zip_path: str) -> Path:
    """
    Create and return extraction directory path.
    
    Args:
        zip_path: Path to ZIP file
        
    Returns:
        Path to extraction directory
    """
    zip_file = Path(zip_path)
    return zip_file.parent / zip_file.stem

def write_output(output_json: str, output_path: str) -> None:
    """
    Write JSON output to file.
    
    Args:
        output_json: JSON string to write
        output_path: Path to output file
    """
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output_json)
        logger.info(f"Output written to {output_path}")
    except Exception as e:
        logger.error(f"Failed to write output: {str(e)}")
        raise

def cleanup_extract_dir(extract_dir: Path) -> None:
    """
    Clean up extraction directory.
    
    Args:
        extract_dir: Path to extraction directory
    """
    if extract_dir.exists():
        try:
            shutil.rmtree(extract_dir)
            logger.info(f"Cleaned up extracted files in {extract_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up extraction directory: {str(e)}")