"""
Validation utilities for the Academic Data Analysis System.
Provides data validation and configuration validation functions.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from .exceptions import DataValidationError, ConfigurationError
from ..config import data_config, analysis_config


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    min_rows: Optional[int] = None,
    max_missing_percentage: Optional[float] = None
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate a DataFrame against specified criteria.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        min_rows: Minimum number of rows required
        max_missing_percentage: Maximum percentage of missing values allowed
        
    Returns:
        Tuple of (is_valid, validation_results)
        
    Raises:
        DataValidationError: If validation fails
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'info': {}
    }
    
    try:
        # Check if DataFrame is empty
        if df.empty:
            validation_results['is_valid'] = False
            validation_results['errors'].append("DataFrame is empty")
            return False, validation_results
        
        # Check minimum rows
        min_rows = min_rows or data_config.min_rows
        if len(df) < min_rows:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"DataFrame has {len(df)} rows, minimum required: {min_rows}")
        
        # Check required columns
        if required_columns:
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                validation_results['is_valid'] = False
                validation_results['errors'].append(f"Missing required columns: {missing_columns}")
        
        # Check missing values percentage
        max_missing_percentage = max_missing_percentage or data_config.max_missing_percentage
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isnull().sum().sum()
        missing_percentage = missing_cells / total_cells
        
        if missing_percentage > max_missing_percentage:
            validation_results['is_valid'] = False
            validation_results['errors'].append(
                f"Missing values percentage ({missing_percentage:.2%}) exceeds limit ({max_missing_percentage:.2%})"
            )
        
        # Check data types
        numeric_cols = [col for col in data_config.numeric_columns if col in df.columns]
        categorical_cols = [col for col in data_config.categorical_columns if col in df.columns]
        
        # Validate numeric columns
        for col in numeric_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                validation_results['warnings'].append(f"Column '{col}' should be numeric but is {df[col].dtype}")
        
        # Check for duplicates
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            validation_results['warnings'].append(f"Found {duplicates} duplicate rows")
        
        # Add info
        validation_results['info'] = {
            'shape': df.shape,
            'missing_percentage': missing_percentage,
            'duplicates': duplicates,
            'numeric_columns': numeric_cols,
            'categorical_columns': categorical_cols
        }
        
        return validation_results['is_valid'], validation_results
        
    except Exception as e:
        raise DataValidationError(f"Error during DataFrame validation: {str(e)}")


def validate_numeric_column(df: pd.DataFrame, column: str, min_value: Optional[float] = None, max_value: Optional[float] = None) -> bool:
    """
    Validate a numeric column in a DataFrame.
    
    Args:
        df: DataFrame containing the column
        column: Name of the column to validate
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        
    Returns:
        True if valid, False otherwise
        
    Raises:
        DataValidationError: If validation fails
    """
    try:
        if column not in df.columns:
            raise DataValidationError(f"Column '{column}' not found in DataFrame")
        
        # Check if column is numeric
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise DataValidationError(f"Column '{column}' is not numeric")
        
        # Check for infinite values
        if df[column].isin([np.inf, -np.inf]).any():
            raise DataValidationError(f"Column '{column}' contains infinite values")
        
        # Check value ranges
        if min_value is not None and df[column].min() < min_value:
            raise DataValidationError(f"Column '{column}' contains values below minimum ({min_value})")
        
        if max_value is not None and df[column].max() > max_value:
            raise DataValidationError(f"Column '{column}' contains values above maximum ({max_value})")
        
        return True
        
    except Exception as e:
        if isinstance(e, DataValidationError):
            raise
        raise DataValidationError(f"Error validating numeric column '{column}': {str(e)}")


def validate_categorical_column(df: pd.DataFrame, column: str, allowed_values: Optional[List[str]] = None) -> bool:
    """
    Validate a categorical column in a DataFrame.
    
    Args:
        df: DataFrame containing the column
        column: Name of the column to validate
        allowed_values: List of allowed values
        
    Returns:
        True if valid, False otherwise
        
    Raises:
        DataValidationError: If validation fails
    """
    try:
        if column not in df.columns:
            raise DataValidationError(f"Column '{column}' not found in DataFrame")
        
        # Check allowed values
        if allowed_values is not None:
            unique_values = df[column].dropna().unique()
            invalid_values = set(unique_values) - set(allowed_values)
            if invalid_values:
                raise DataValidationError(f"Column '{column}' contains invalid values: {invalid_values}")
        
        return True
        
    except Exception as e:
        if isinstance(e, DataValidationError):
            raise
        raise DataValidationError(f"Error validating categorical column '{column}': {str(e)}")


def validate_file_path(file_path: Path, must_exist: bool = True, allowed_extensions: Optional[List[str]] = None) -> bool:
    """
    Validate a file path.
    
    Args:
        file_path: Path to validate
        must_exist: Whether the file must exist
        allowed_extensions: List of allowed file extensions
        
    Returns:
        True if valid, False otherwise
        
    Raises:
        DataValidationError: If validation fails
    """
    try:
        if must_exist and not file_path.exists():
            raise DataValidationError(f"File does not exist: {file_path}")
        
        if allowed_extensions:
            if file_path.suffix.lower() not in [ext.lower() for ext in allowed_extensions]:
                raise DataValidationError(f"File extension {file_path.suffix} not in allowed extensions: {allowed_extensions}")
        
        return True
        
    except Exception as e:
        if isinstance(e, DataValidationError):
            raise
        raise DataValidationError(f"Error validating file path '{file_path}': {str(e)}")


def validate_config(config: Any) -> bool:
    """
    Validate configuration objects.
    
    Args:
        config: Configuration object to validate
        
    Returns:
        True if valid, False otherwise
        
    Raises:
        ConfigurationError: If validation fails
    """
    try:
        from ..config.settings import (
            AppConfig, DataConfig, AnalysisConfig, 
            VisualizationConfig, ReportConfig, LoggingConfig
        )
        
        if isinstance(config, AppConfig):
            if not config.base_dir.exists():
                raise ConfigurationError(f"Base directory does not exist: {config.base_dir}")
            return True
            
        elif isinstance(config, DataConfig):
            if not config.numeric_columns:
                raise ConfigurationError("No numeric columns defined")
            if not config.categorical_columns:
                raise ConfigurationError("No categorical columns defined")
            return True
            
        elif isinstance(config, AnalysisConfig):
            if config.minimum_passing_grade >= config.excellent_grade:
                raise ConfigurationError("Minimum passing grade must be less than excellent grade")
            if config.confidence_level <= 0 or config.confidence_level >= 1:
                raise ConfigurationError("Confidence level must be between 0 and 1")
            return True
            
        elif isinstance(config, VisualizationConfig):
            if config.dpi <= 0:
                raise ConfigurationError("DPI must be positive")
            if config.figure_size[0] <= 0 or config.figure_size[1] <= 0:
                raise ConfigurationError("Figure size dimensions must be positive")
            return True
            
        elif isinstance(config, ReportConfig):
            if not config.title:
                raise ConfigurationError("Report title cannot be empty")
            if not config.author:
                raise ConfigurationError("Report author cannot be empty")
            return True
            
        elif isinstance(config, LoggingConfig):
            valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
            if config.default_level not in valid_levels:
                raise ConfigurationError(f"Invalid log level: {config.default_level}")
            return True
        
        return True
        
    except Exception as e:
        if isinstance(e, ConfigurationError):
            raise
        raise ConfigurationError(f"Error validating configuration: {str(e)}")


def validate_analysis_parameters(params: Dict[str, Any]) -> bool:
    """
    Validate analysis parameters.
    
    Args:
        params: Dictionary of analysis parameters
        
    Returns:
        True if valid, False otherwise
        
    Raises:
        DataValidationError: If validation fails
    """
    try:
        # Validate statistical parameters
        if 'confidence_level' in params:
            confidence_level = params['confidence_level']
            if not 0 < confidence_level < 1:
                raise DataValidationError(f"Confidence level must be between 0 and 1, got: {confidence_level}")
        
        if 'alpha' in params:
            alpha = params['alpha']
            if not 0 < alpha < 1:
                raise DataValidationError(f"Alpha must be between 0 and 1, got: {alpha}")
        
        # Validate threshold parameters
        if 'risk_threshold' in params:
            risk_threshold = params['risk_threshold']
            if risk_threshold < 0:
                raise DataValidationError(f"Risk threshold must be non-negative, got: {risk_threshold}")
        
        return True
        
    except Exception as e:
        if isinstance(e, DataValidationError):
            raise
        raise DataValidationError(f"Error validating analysis parameters: {str(e)}")


def get_validation_summary(validation_results: Dict[str, Any]) -> str:
    """
    Generate a human-readable summary of validation results.
    
    Args:
        validation_results: Results from validation functions
        
    Returns:
        Formatted summary string
    """
    summary = []
    
    if validation_results['is_valid']:
        summary.append("✅ Validation passed")
    else:
        summary.append("❌ Validation failed")
    
    if validation_results['errors']:
        summary.append(f"Errors ({len(validation_results['errors'])}):")
        for error in validation_results['errors']:
            summary.append(f"  • {error}")
    
    if validation_results['warnings']:
        summary.append(f"Warnings ({len(validation_results['warnings'])}):")
        for warning in validation_results['warnings']:
            summary.append(f"  • {warning}")
    
    if validation_results['info']:
        summary.append("Information:")
        for key, value in validation_results['info'].items():
            summary.append(f"  • {key}: {value}")
    
    return "\n".join(summary)