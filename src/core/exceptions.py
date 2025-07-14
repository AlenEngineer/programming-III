"""
Custom exceptions for the Academic Data Analysis System.
Provides specific error types for better error handling and debugging.
"""

from typing import Optional, Any, Dict


class AcademicAnalysisError(Exception):
    """Base exception for Academic Data Analysis System."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the exception.
        
        Args:
            message: Human-readable error message
            error_code: Machine-readable error code
            details: Additional error details
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'error_code': self.error_code,
            'details': self.details
        }


class DataLoadError(AcademicAnalysisError):
    """Raised when data loading fails."""
    
    def __init__(self, message: str, file_path: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="DATA_LOAD_ERROR", **kwargs)
        self.file_path = file_path
        if file_path:
            self.details['file_path'] = file_path


class DataValidationError(AcademicAnalysisError):
    """Raised when data validation fails."""
    
    def __init__(self, message: str, validation_type: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="DATA_VALIDATION_ERROR", **kwargs)
        self.validation_type = validation_type
        if validation_type:
            self.details['validation_type'] = validation_type


class DataCleaningError(AcademicAnalysisError):
    """Raised when data cleaning fails."""
    
    def __init__(self, message: str, cleaning_step: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="DATA_CLEANING_ERROR", **kwargs)
        self.cleaning_step = cleaning_step
        if cleaning_step:
            self.details['cleaning_step'] = cleaning_step


class AnalysisError(AcademicAnalysisError):
    """Raised when statistical analysis fails."""
    
    def __init__(self, message: str, analysis_type: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="ANALYSIS_ERROR", **kwargs)
        self.analysis_type = analysis_type
        if analysis_type:
            self.details['analysis_type'] = analysis_type


class VisualizationError(AcademicAnalysisError):
    """Raised when visualization generation fails."""
    
    def __init__(self, message: str, chart_type: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="VISUALIZATION_ERROR", **kwargs)
        self.chart_type = chart_type
        if chart_type:
            self.details['chart_type'] = chart_type


class ReportGenerationError(AcademicAnalysisError):
    """Raised when report generation fails."""
    
    def __init__(self, message: str, report_section: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="REPORT_GENERATION_ERROR", **kwargs)
        self.report_section = report_section
        if report_section:
            self.details['report_section'] = report_section


class ConfigurationError(AcademicAnalysisError):
    """Raised when configuration is invalid."""
    
    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="CONFIGURATION_ERROR", **kwargs)
        self.config_key = config_key
        if config_key:
            self.details['config_key'] = config_key


# Utility functions for exception handling
def log_exception(logger, exception: AcademicAnalysisError, context: Optional[str] = None) -> None:
    """
    Log an exception with structured information.
    
    Args:
        logger: Logger instance
        exception: The exception to log
        context: Additional context information
    """
    error_dict = exception.to_dict()
    if context:
        error_dict['context'] = context
    
    logger.error(
        f"{exception.__class__.__name__}: {exception.message}",
        extra=error_dict
    )


def handle_exception(func):
    """
    Decorator to handle exceptions and convert them to AcademicAnalysisError.
    
    Args:
        func: Function to wrap
        
    Returns:
        Wrapped function
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except AcademicAnalysisError:
            # Re-raise custom exceptions
            raise
        except Exception as e:
            # Convert other exceptions to AcademicAnalysisError
            raise AcademicAnalysisError(
                f"Unexpected error in {func.__name__}: {str(e)}",
                error_code="UNEXPECTED_ERROR",
                details={'function': func.__name__, 'original_error': str(e)}
            ) from e
    
    return wrapper