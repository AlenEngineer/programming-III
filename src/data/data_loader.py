import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Union, Optional, Dict, Any
import sys
import os

# Add the parent directory to the path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import DATA_FILE_PATH, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS
from src.utils.helpers import validate_dataframe_columns, log_analysis_step, setup_logging

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)

def load_csv_data(file_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Load data from a CSV file with proper error handling and validation.
    
    Args:
        file_path: Path to the CSV file. If None, uses default from config.
        
    Returns:
        Loaded DataFrame
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file is empty or has invalid format
    """
    if file_path is None:
        file_path = DATA_FILE_PATH
    
    file_path = Path(file_path)
    
    try:
        log_analysis_step("Loading CSV data", f"File: {file_path}")
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Load the CSV file
        df = pd.read_csv(file_path)
        
        if df.empty:
            raise ValueError("The CSV file is empty")
        
        logger.info(f"Successfully loaded CSV data: {len(df)} rows, {len(df.columns)} columns")
        return df
        
    except Exception as e:
        logger.error(f"Error loading CSV file {file_path}: {e}")
        raise

def load_excel_data(file_path: Union[str, Path], sheet_name: Union[str, int] = 0) -> pd.DataFrame:
    """
    Load data from an Excel file with proper error handling.
    
    Args:
        file_path: Path to the Excel file
        sheet_name: Name or index of the sheet to load
        
    Returns:
        Loaded DataFrame
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file is empty or has invalid format
    """
    file_path = Path(file_path)
    
    try:
        log_analysis_step("Loading Excel data", f"File: {file_path}, Sheet: {sheet_name}")
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Load the Excel file
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        
        if df.empty:
            raise ValueError("The Excel file is empty")
        
        logger.info(f"Successfully loaded Excel data: {len(df)} rows, {len(df.columns)} columns")
        return df
        
    except Exception as e:
        logger.error(f"Error loading Excel file {file_path}: {e}")
        raise

def validate_data_structure(df: pd.DataFrame) -> bool:
    """
    Validate that the loaded data has the expected structure for academic analysis.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        True if data structure is valid, False otherwise
    """
    try:
        log_analysis_step("Validating data structure")
        
        # Check for required columns
        all_required_columns = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS
        if not validate_dataframe_columns(df, all_required_columns):
            return False
        
        # Check data types
        for col in NUMERIC_COLUMNS:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                logger.warning(f"Column {col} is not numeric, will attempt conversion")
        
        # Check for completely empty columns
        empty_columns = df.columns[df.isnull().all()].tolist()
        if empty_columns:
            logger.warning(f"Found completely empty columns: {empty_columns}")
        
        # Check minimum number of rows
        if len(df) < 10:
            logger.warning("Dataset has very few rows, analysis may not be meaningful")
        
        logger.info("Data structure validation completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error during data validation: {e}")
        return False

def get_data_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get comprehensive information about the loaded dataset.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary containing dataset information
    """
    try:
        log_analysis_step("Generating data information summary")
        
        info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'data_types': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'categorical_columns': [col for col in df.columns if col in CATEGORICAL_COLUMNS],
            'numeric_columns': [col for col in df.columns if col in NUMERIC_COLUMNS],
        }
        
        # Add unique value counts for categorical columns
        unique_counts = {}
        for col in info['categorical_columns']:
            if col in df.columns:
                unique_counts[col] = df[col].nunique()
        info['unique_value_counts'] = unique_counts
        
        # Add basic statistics for numeric columns
        numeric_stats = {}
        for col in info['numeric_columns']:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                numeric_stats[col] = {
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std())
                }
        info['numeric_statistics'] = numeric_stats
        
        logger.info("Data information summary generated successfully")
        return info
        
    except Exception as e:
        logger.error(f"Error generating data info: {e}")
        return {}

def load_and_validate_data(file_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Complete data loading pipeline with validation.
    
    Args:
        file_path: Path to the data file (CSV or Excel)
        
    Returns:
        Validated DataFrame ready for analysis
        
    Raises:
        ValueError: If data validation fails
    """
    try:
        # Determine file type and load accordingly
        if file_path is None:
            file_path = DATA_FILE_PATH
        
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.csv':
            df = load_csv_data(file_path)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            df = load_excel_data(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        # Validate the loaded data
        if not validate_data_structure(df):
            raise ValueError("Data validation failed")
        
        # Get and log data information
        data_info = get_data_info(df)
        logger.info(f"Loaded dataset: {data_info['shape'][0]} rows, {data_info['shape'][1]} columns")
        
        return df
        
    except Exception as e:
        logger.error(f"Error in data loading pipeline: {e}")
        raise

if __name__ == "__main__":
    # Test the data loader
    try:
        df = load_and_validate_data()
        print("Data loaded successfully!")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("\nFirst few rows:")
        print(df.head())
    except Exception as e:
        print(f"Error: {e}")
