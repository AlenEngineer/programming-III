"""
Centralized logging setup for the Academic Data Analysis System.
Provides consistent logging configuration across all modules.
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional, Dict, Any

from ..config import logging_config, app_config


class ContextFilter(logging.Filter):
    """Custom filter to add context information to log records."""
    
    def __init__(self, context: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.context = context or {}
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add context information to the log record."""
        for key, value in self.context.items():
            setattr(record, key, value)
        return True


def setup_logging(
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None
) -> None:
    """
    Set up logging configuration for the application.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        context: Additional context to include in logs
    """
    # Use configuration defaults if not provided
    level = level or logging_config.default_level
    log_file = log_file or logging_config.log_file
    
    # Create logs directory if it doesn't exist
    log_path = app_config.logs_dir / log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        logging_config.log_format,
        logging_config.date_format
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, logging_config.console_level.upper()))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=logging_config.max_file_size,
        backupCount=logging_config.backup_count
    )
    file_handler.setLevel(getattr(logging, logging_config.file_level.upper()))
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Add context filter if provided
    if context:
        context_filter = ContextFilter(context)
        for handler in root_logger.handlers:
            handler.addFilter(context_filter)
    
    # Set up specific loggers
    for logger_name in logging_config.loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, level.upper()))
    
    # Log setup completion
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured - Level: {level}, File: {log_path}")


def get_logger(name: str, context: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """
    Get a logger instance with optional context.
    
    Args:
        name: Logger name (usually __name__)
        context: Additional context to include in logs
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Add context filter if provided
    if context:
        context_filter = ContextFilter(context)
        for handler in logger.handlers:
            handler.addFilter(context_filter)
    
    return logger


def log_function_call(func_name: str, args: Optional[Dict[str, Any]] = None, level: str = "DEBUG") -> None:
    """
    Log a function call with arguments.
    
    Args:
        func_name: Name of the function being called
        args: Function arguments
        level: Log level
    """
    logger = get_logger(__name__)
    log_level = getattr(logging, level.upper())
    
    message = f"Calling function: {func_name}"
    if args:
        message += f" with args: {args}"
    
    logger.log(log_level, message)


def log_analysis_step(step_name: str, details: Optional[str] = None, level: str = "INFO") -> None:
    """
    Log an analysis step with consistent formatting.
    
    Args:
        step_name: Name of the analysis step
        details: Additional details
        level: Log level
    """
    logger = get_logger(__name__)
    log_level = getattr(logging, level.upper())
    
    message = f"📊 {step_name}"
    if details:
        message += f" - {details}"
    
    logger.log(log_level, message)


def log_performance(func_name: str, duration: float, level: str = "INFO") -> None:
    """
    Log performance information for a function.
    
    Args:
        func_name: Name of the function
        duration: Execution duration in seconds
        level: Log level
    """
    logger = get_logger(__name__)
    log_level = getattr(logging, level.upper())
    
    logger.log(log_level, f"⏱️ {func_name} completed in {duration:.2f} seconds")


def log_data_info(data_info: Dict[str, Any], level: str = "INFO") -> None:
    """
    Log data information in a structured format.
    
    Args:
        data_info: Dictionary containing data information
        level: Log level
    """
    logger = get_logger(__name__)
    log_level = getattr(logging, level.upper())
    
    message = "📈 Data Info:"
    for key, value in data_info.items():
        message += f" {key}={value}"
    
    logger.log(log_level, message)


# Decorator for automatic function logging
def log_function(level: str = "DEBUG"):
    """
    Decorator to automatically log function calls.
    
    Args:
        level: Log level for function calls
        
    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            logger = get_logger(func.__module__)
            log_level = getattr(logging, level.upper())
            
            logger.log(log_level, f"→ Entering {func_name}")
            
            try:
                result = func(*args, **kwargs)
                logger.log(log_level, f"← Exiting {func_name}")
                return result
            except Exception as e:
                logger.log(logging.ERROR, f"❌ Error in {func_name}: {str(e)}")
                raise
        
        return wrapper
    return decorator