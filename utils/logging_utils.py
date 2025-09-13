"""
logging_utils.py
----------------
Utility module for consistent logging across the project.
Supports console logging, file logging, and configurable log levels.
"""

import logging
import os
from datetime import datetime

# --------------------------------------
# 1. SETUP LOGGER
# --------------------------------------
def get_logger(name: str, log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    Creates a logger instance with console and optional file output.

    Args:
        name (str): Name of the logger
        log_file (str, optional): Path to log file
        level (int): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Prevent duplicate handlers
    if not logger.handlers:
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(level)
        formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File handler
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setLevel(level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
    
    return logger


# --------------------------------------
# 2. HELPER FUNCTIONS
# --------------------------------------
def log_start(logger: logging.Logger, message: str):
    """Log the start of a process."""
    logger.info(f"--- START: {message} ---")

def log_end(logger: logging.Logger, message: str):
    """Log the end of a process."""
    logger.info(f"--- END: {message} ---")

def log_exception(logger: logging.Logger, message: str, exc: Exception):
    """Log an exception with details."""
    logger.error(f"{message} | Exception: {exc}", exc_info=True)