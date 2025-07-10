"""
Utility functions for the Academic Data Analysis System
Common helper functions used across different modules.
"""

import os
import logging
from pathlib import Path
from typing import Union, Optional
import pandas as pd
import numpy as np

def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format a decimal value as a percentage string.
    
    Args:
        value: Decimal value to convert to percentage
        decimals: Number of decimal places to show
        
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"

def safe_divide(numerator: Union[int, float], denominator: Union[int, float]) -> float:
    """
    Safely divide two numbers, returning 0 if denominator is 0.
    
    Args:
        numerator: The dividend
        denominator: The divisor
        
    Returns:
        Division result or 0 if denominator is 0
    """
    if denominator == 0:
        return 0.0
    return numerator / denominator

def create_output_directory(directory_path: Union[str, Path]) -> bool:
    """
    Create a directory if it doesn't exist.
    
    Args:
        directory_path: Path to the directory to create
        
    Returns:
        True if directory was created or already exists, False otherwise
    """
    try:
        Path(directory_path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logging.error(f"Failed to create directory {directory_path}: {e}")
        return False

def setup_logging(log_level: str = "INFO") -> None:
    """
    Set up logging configuration for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('analysis.log'),
            logging.StreamHandler()
        ]
    )

def log_analysis_step(step_name: str, details: Optional[str] = None) -> None:
    """
    Log an analysis step with optional details.
    
    Args:
        step_name: Name of the analysis step
        details: Additional details about the step
    """
    logger = logging.getLogger(__name__)
    message = f"Analysis Step: {step_name}"
    if details:
        message += f" - {details}"
    logger.info(message)

def validate_dataframe_columns(df: pd.DataFrame, required_columns: list) -> bool:
    """
    Validate that a DataFrame contains all required columns.
    
    Args:
        df: DataFrame to validate
        required_columns: List of column names that must be present
        
    Returns:
        True if all required columns are present, False otherwise
    """
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        logging.error(f"Missing required columns: {missing_columns}")
        return False
    return True

def get_numeric_summary(series: pd.Series) -> dict:
    """
    Get a comprehensive numeric summary of a pandas Series.
    
    Args:
        series: Pandas Series with numeric data
        
    Returns:
        Dictionary with statistical summary
    """
    if not pd.api.types.is_numeric_dtype(series):
        raise ValueError("Series must contain numeric data")
    
    return {
        'count': len(series),
        'mean': float(series.mean()),
        'median': float(series.median()),
        'std': float(series.std()),
        'min': float(series.min()),
        'max': float(series.max()),
        'q25': float(series.quantile(0.25)),
        'q75': float(series.quantile(0.75))
    }

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize column names in a DataFrame.
    
    Args:
        df: DataFrame with potentially messy column names
        
    Returns:
        DataFrame with cleaned column names
    """
    df_copy = df.copy()
    df_copy.columns = df_copy.columns.str.strip().str.lower().str.replace(' ', '_')
    return df_copy

def export_to_csv(df: pd.DataFrame, filename: str, output_dir: Union[str, Path]) -> str:
    """
    Export a DataFrame to CSV file in the specified directory.
    
    Args:
        df: DataFrame to export
        filename: Name of the output file (without extension)
        output_dir: Directory to save the file
        
    Returns:
        Full path to the exported file
    """
    output_path = Path(output_dir) / f"{filename}.csv"
    create_output_directory(output_dir)
    df.to_csv(output_path, index=False)
    logging.info(f"Data exported to: {output_path}")
    return str(output_path)
