"""
Logging utilities for the N.O.S academic tool.

This module provides centralized logging configuration and utilities
for the NEAT-based optimization system and Streamlit application.

Author: N.O.S Team
Date: 2024
"""

import logging
import os
from datetime import datetime
from pathlib import Path


def setup_logging(log_level=logging.INFO, log_file=None):
    """
    Set up logging configuration for the application.
    
    Args:
        log_level: Logging level (default: INFO)
        log_file: Optional log file path
    """
    # Create logs directory if it doesn't exist
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Console handler
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )


def get_logger(name):
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Default logger for the application
logger = get_logger(__name__) 