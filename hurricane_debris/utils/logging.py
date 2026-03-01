"""
Structured logging utilities.
Replaces all print() calls with proper logging.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


_CONFIGURED = False


def setup_logger(
    name: str = "hurricane_debris",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Configure and return the root package logger.

    Args:
        name: Logger name (usually the package name).
        log_file: Optional file path for log output.
        level: Logging level.

    Returns:
        Configured logger instance.
    """
    global _CONFIGURED

    logger = logging.getLogger(name)

    if _CONFIGURED:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _CONFIGURED = True
    return logger


def get_logger(module_name: str) -> logging.Logger:
    """Return a child logger for a specific module."""
    return logging.getLogger(f"hurricane_debris.{module_name}")
