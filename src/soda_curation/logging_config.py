import logging
import os
from datetime import datetime


def setup_logging(config) -> None:
    # Create absolute path for logs directory
    log_dir = os.path.abspath("logs")
    os.makedirs(log_dir, exist_ok=True)

    # Configure file logging
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(log_dir, f"{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Logging initialized. Log file: {log_file}")

