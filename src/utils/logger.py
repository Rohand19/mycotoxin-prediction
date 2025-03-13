"""Logging configuration for DON concentration prediction."""

import logging
import logging.handlers
import os
from datetime import datetime


def setup_logging(log_dir="logs", log_level=logging.INFO):
    """Set up logging configuration.

    Args:
        log_dir (str): Directory to store log files
        log_level: Logging level (default: logging.INFO)
    """
    try:
        # Create logs directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)

        # Generate log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"don_predictor_{timestamp}.log")

        # Create formatters
        file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        # Create and configure file handler
        file_handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=10485760, backupCount=5)  # 10MB
        file_handler.setLevel(log_level)
        file_handler.setFormatter(file_formatter)

        # Create and configure console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(console_formatter)

        # Get root logger and configure it
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)

        # Remove existing handlers to avoid duplicates
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Add handlers to root logger
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)

        # Log initial message
        root_logger.info(f"Logging configured. Log file: {log_file}")

    except Exception as e:
        print(f"Error setting up logging: {str(e)}")
        raise


class LoggerManager:
    """Context manager for temporarily changing log level."""

    def __init__(self, logger_name=None, level=logging.DEBUG):
        """Initialize logger manager.

        Args:
            logger_name (str, optional): Logger name (None for root logger)
            level: Temporary logging level
        """
        self.logger = logging.getLogger(logger_name)
        self.level = level
        self.original_level = self.logger.level

    def __enter__(self):
        """Set temporary log level."""
        self.logger.setLevel(self.level)
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original log level."""
        self.logger.setLevel(self.original_level)


def get_memory_handler(threshold=90):
    """Create a handler that logs when memory usage exceeds threshold.

    Args:
        threshold (int): Memory usage threshold percentage

    Returns:
        MemoryHandler: Custom logging handler for memory monitoring
    """

    class MemoryHandler(logging.Handler):
        def __init__(self, threshold):
            super().__init__()
            self.threshold = threshold
            self.formatter = logging.Formatter("%(asctime)s - MEMORY WARNING - %(message)s")

        def emit(self, record):
            try:
                import psutil

                memory_percent = psutil.Process().memory_percent()

                if memory_percent > self.threshold:
                    record.msg = f"Memory usage ({memory_percent:.1f}%) " f"exceeded threshold ({self.threshold}%)"
                    print(self.formatter.format(record))
            except Exception:
                pass

    return MemoryHandler(threshold)


def log_exception(logger):
    """Decorator to log exceptions in functions.

    Args:
        logger: Logger instance to use for logging

    Returns:
        Decorated function
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.exception(f"Exception in {func.__name__}: {str(e)}")
                raise

        return wrapper

    return decorator
