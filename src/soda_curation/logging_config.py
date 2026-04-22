"""Logging configuration with environment support."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

_RESERVED_LOG_RECORD_FIELDS = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
    "message",
    "asctime",
}


class ContextAwareFormatter(logging.Formatter):
    """Formatter that appends structured `extra` context to each log line."""

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _RESERVED_LOG_RECORD_FIELDS and not key.startswith("_")
        }
        if not extras:
            return base

        def _stringify(value: Any) -> str:
            if isinstance(value, (dict, list, tuple, set)):
                try:
                    return json.dumps(value, ensure_ascii=False, default=str)
                except Exception:
                    return str(value)
            return str(value)

        extras_str = " ".join(
            f"{key}={_stringify(value)}" for key, value in sorted(extras.items())
        )
        return f"{base} | {extras_str}"


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
    formatter = ContextAwareFormatter(log_format, datefmt=date_format)
    file_handler = logging.FileHandler(log_file)
    stream_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logging.basicConfig(
        level=log_level,
        handlers=[file_handler, stream_handler],
        force=True,
    )

    logging.info(f"Logging initialized. Log file: {log_file}")
