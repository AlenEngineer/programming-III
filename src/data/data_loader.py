import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Union, Optional, Dict, Any

from ..config import data_config, app_config
from ..core import get_logger, DataLoadError, DataValidationError, validate_dataframe, log_function
from ..utils.helpers import validate_dataframe_columns

# Set up logging
logger = get_logger(__name__)

@log_function()
def load_csv_data(file_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Load data from a CSV file with proper error handling and validation.
    
    Args:
        file_path: Path to the CSV file. If None, uses default from config.
        
    Returns:
        Loaded DataFrame
        
    Raises:
        DataLoadError: If the file doesn't exist or loading fails
    """
    if file_path is None:
        file_path = data_config.data_file_path
    
    file_path = Path(file_path)
    
    try:
        logger.info(f"Loading CSV data from: {file_path}")
        
        if not file_path.exists():
            raise DataLoadError(f"File not found: {file_path}", file_path=str(file_path))
        
        # Load the CSV file
        df = pd.read_csv(file_path)
        
        if df.empty:
            raise DataLoadError("The CSV file is empty", file_path=str(file_path))
        
        logger.info(f"Successfully loaded CSV data: {len(df)} rows, {len(df.columns)} columns")
        return df
        
    except DataLoadError:
        raise
    except Exception as e:
        raise DataLoadError(f"Error loading CSV file {file_path}: {str(e)}", file_path=str(file_path)) from e

@log_function()
def load_excel_data(file_path: Union[str, Path], sheet_name: Union[str, int] = 0) -> pd.DataFrame:
    """
    Load data from an Excel file with proper error handling.
    
    Args:
        file_path: Path to the Excel file
        sheet_name: Name or index of the sheet to load
        
    Returns:
        Loaded DataFrame
        
    Raises:
        DataLoadError: If the file doesn't exist or loading fails
    """
    file_path = Path(file_path)
    
    try:
        logger.info(f"Loading Excel data from: {file_path}, Sheet: {sheet_name}")
        
        if not file_path.exists():
            raise DataLoadError(f"File not found: {file_path}", file_path=str(file_path))
        
        # Load the Excel file
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        
        if df.empty:
            raise DataLoadError("The Excel file is empty", file_path=str(file_path))
        
        logger.info(f"Successfully loaded Excel data: {len(df)} rows, {len(df.columns)} columns")
        return df
        
    except DataLoadError:
        raise
    except Exception as e:
        raise DataLoadError(f"Error loading Excel file {file_path}: {str(e)}", file_path=str(file_path)) from e

@log_function()
def validate_data_structure(df: pd.DataFrame) -> bool:
    """
    Validate that the loaded data has the expected structure for academic analysis.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        True if data structure is valid, False otherwise
        
    Raises:
        DataValidationError: If validation fails
    """
    try:
        logger.info("Validating data structure")
        
        # Use the new validation function
        is_valid, validation_results = validate_dataframe(
            df,
            required_columns=data_config.numeric_columns + data_config.categorical_columns,
            min_rows=data_config.min_rows,
            max_missing_percentage=data_config.max_missing_percentage
        )
        
        if not is_valid:
            error_messages = "; ".join(validation_results['errors'])
            raise DataValidationError(f"Data structure validation failed: {error_messages}")
        
        # Log warnings if any
        if validation_results['warnings']:
            for warning in validation_results['warnings']:
                logger.warning(warning)
        
        logger.info("Data structure validation passed")
        return True
        
    except DataValidationError:
        raise
    except Exception as e:
        raise DataValidationError(f"Error during data structure validation: {str(e)}") from e

@log_function()
def get_data_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get comprehensive information about the loaded dataset.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary containing dataset information
    """
    try:
        logger.info("Generating data information summary")
        
        info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'data_types': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'categorical_columns': [col for col in df.columns if col in data_config.categorical_columns],
            'numeric_columns': [col for col in df.columns if col in data_config.numeric_columns],
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

@log_function()
def load_and_validate_data(file_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Complete data loading pipeline with validation.
    
    Args:
        file_path: Path to the data file (CSV or Excel)
        
    Returns:
        Validated DataFrame ready for analysis
        
    Raises:
        DataLoadError: If data loading fails
        DataValidationError: If data validation fails
    """
    try:
        # Determine file type and load accordingly
        if file_path is None:
            file_path = data_config.data_file_path
        
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.csv':
            df = load_csv_data(file_path)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            df = load_excel_data(file_path)
        else:
            raise DataLoadError(f"Unsupported file format: {file_path.suffix}", file_path=str(file_path))
        
        # Validate the loaded data
        validate_data_structure(df)
        
        # Get and log data information
        data_info = get_data_info(df)
        logger.info(f"Loaded dataset: {data_info['shape'][0]} rows, {data_info['shape'][1]} columns")
        
        return df
        
    except (DataLoadError, DataValidationError):
        raise
    except Exception as e:
        raise DataLoadError(f"Error in data loading pipeline: {str(e)}", file_path=str(file_path)) from e

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
