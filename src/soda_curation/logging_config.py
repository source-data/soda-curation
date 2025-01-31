"""Logging configuration with environment support."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def setup_logging(config: Dict[str, Any]) -> None:
    """
    Set up logging with environment-specific configuration.

    Args:
        config: Configuration dictionary containing logging settings
    """
    # Get logging configuration with defaults
    log_config = config.get("logging", {})
    log_level = getattr(logging, log_config.get("level", "INFO"))
    log_format = log_config.get(
        "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    date_format = log_config.get("date_format", "%Y-%m-%d %H:%M:%S")

    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / f"{timestamp}.log"

    # Configure logging
    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    logging.info(f"Logging initialized. Log file: {log_file}")
