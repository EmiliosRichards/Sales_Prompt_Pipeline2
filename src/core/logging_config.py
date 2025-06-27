"""
Logging configuration module for the Intelligent Prospect Analyzer.

This module provides a centralized function `setup_logging` to configure
application-wide logging. It supports both console output and logging to a
rotating file, with configurable log levels for each.
"""
import logging
import sys
from typing import Optional
from logging.handlers import RotatingFileHandler

def setup_logging(
    file_log_level: int = logging.INFO,
    console_log_level: int = logging.WARNING,
    log_file_path: Optional[str] = None
) -> None:
    """
    Set up logging configuration for both console and optional file output.

    This function configures the root logger. It adds a console handler
    and, if a `log_file_path` is provided, a `RotatingFileHandler`.
    The rotating file handler manages log file sizes and backups.

    Args:
        file_log_level (int): The logging level for the file handler
            (e.g., `logging.DEBUG`, `logging.INFO`).
            Defaults to `logging.INFO`.
        console_log_level (int): The logging level for the console handler.
            Defaults to `logging.WARNING`.
        log_file_path (Optional[str]): Path to the log file. If None,
            file logging is disabled. Defaults to None.
    """
    # Get the root logger
    root_logger = logging.getLogger()
    # Set root logger level to the lowest of the handlers to allow all messages through to handlers
    effective_root_level = min(file_log_level, console_log_level)
    root_logger.setLevel(effective_root_level)

    # Clear existing handlers to avoid duplicate logs if this function is called multiple times
    # (e.g., in interactive sessions or tests)
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Create a standard formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File Handler (if path is provided)
    if log_file_path:
        try:
            # Use RotatingFileHandler for log rotation
            max_bytes = 10 * 1024 * 1024  # 10 MB per log file
            backup_count = 5  # Keep up to 5 backup log files
            
            file_handler = RotatingFileHandler(
                log_file_path,
                maxBytes=max_bytes,
                backupCount=backup_count,
                mode='a',        # Append mode
                encoding='utf-8' # Explicitly set encoding for consistency
            )
            file_handler.setLevel(file_log_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            # Log initial setup message to the file logger itself if it's at INFO or lower
            if file_log_level <= logging.INFO:
                 # Use a temporary logger to ensure this message goes to the file if root is higher
                file_setup_logger = logging.getLogger(__name__ + ".file_setup")
                file_setup_logger.addHandler(file_handler) # Temporarily add handler
                file_setup_logger.setLevel(file_log_level)
                file_setup_logger.info(f"File logging enabled. Log file: {log_file_path}, Level: {logging.getLevelName(file_log_level)}")
                file_setup_logger.removeHandler(file_handler) # Clean up temporary handler
            
        except Exception as e:
            # Log error to console if file logging setup fails
            logging.error(f"Failed to set up file logging to {log_file_path}: {e}", exc_info=True)
            # Execution continues with console logging enabled.