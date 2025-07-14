"""
Core module for the Academic Data Analysis System.
Contains base classes, exceptions, and core utilities.
"""

from .exceptions import (
    AcademicAnalysisError,
    DataLoadError,
    DataValidationError,
    DataCleaningError,
    AnalysisError,
    VisualizationError,
    ReportGenerationError,
    ConfigurationError
)

from .logging_setup import setup_logging, get_logger
from .validators import validate_dataframe, validate_config

__all__ = [
    'AcademicAnalysisError',
    'DataLoadError', 
    'DataValidationError',
    'DataCleaningError',
    'AnalysisError',
    'VisualizationError',
    'ReportGenerationError',
    'ConfigurationError',
    'setup_logging',
    'get_logger',
    'validate_dataframe',
    'validate_config'
]