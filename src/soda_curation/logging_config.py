import logging
import os
from typing import Dict, Any

def setup_logging(config: Dict[str, Any]) -> None:
    """
    Set up logging configuration based on the provided config.

    Args:
        config (Dict[str, Any]): The configuration dictionary containing logging settings.
    """
    log_config = config.get('logging', {})
    log_level = log_config.get('level', 'INFO').upper()
    log_file = log_config.get('file', 'soda_curation.log')
    log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    date_format = log_config.get('date_format', '%Y-%m-%d %H:%M:%S')

    # Ensure the log directory exists
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Set up basic configuration
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        datefmt=date_format,
        filename=log_file,
        filemode='a'
    )

    # Add console handler to print logs to console as well
    console = logging.StreamHandler()
    console.setLevel(getattr(logging, log_level))
    formatter = logging.Formatter(log_format)
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    logging.info(f"Logging initialized with level: {log_level}")
