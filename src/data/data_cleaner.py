"""
Data cleaning and preprocessing module for the Academic Data Analysis System.
Handles data cleaning, type conversion, and preprocessing tasks.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional

from ..config import data_config, analysis_config
from ..core import get_logger, DataCleaningError, log_function
from ..utils.helpers import get_numeric_summary

logger = get_logger(__name__)

@log_function()
def clean_student_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main data cleaning function that orchestrates all cleaning operations.
    
    Args:
        df: Raw DataFrame from data loader
        
    Returns:
        Cleaned DataFrame ready for analysis
        
    Raises:
        DataCleaningError: If data cleaning fails
    """
    try:
        logger.info("Starting data cleaning process")
        
        df_clean = df.copy()
        
        # Clean column names
        df_clean = standardize_columns(df_clean)
        
        # Handle missing values
        df_clean = handle_missing_values(df_clean)
        
        # Convert data types
        df_clean = convert_data_types(df_clean)
        
        # Clean specific columns
        df_clean = clean_categorical_data(df_clean)
        
        # Add derived columns
        df_clean = add_derived_features(df_clean)
        
        logger.info(f"Data cleaning completed. Final shape: {df_clean.shape}")
        return df_clean
        
    except DataCleaningError:
        raise
    except Exception as e:
        raise DataCleaningError(f"Error during data cleaning: {str(e)}", cleaning_step="overall_cleaning") from e

@log_function()
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataset using appropriate strategies.
    
    Args:
        df: DataFrame with potential missing values
        
    Returns:
        DataFrame with missing values handled
        
    Raises:
        DataCleaningError: If missing value handling fails
    """
    try:
        logger.info("Handling missing values")
        
        df_clean = df.copy()
        
        # Log missing value counts
        missing_counts = df_clean.isnull().sum()
        missing_cols = missing_counts[missing_counts > 0]
        
        if len(missing_cols) > 0:
            logger.info(f"Found missing values in columns: {missing_cols.to_dict()}")
        
        # Handle numeric columns - fill with median
        for col in data_config.numeric_columns:
            if col in df_clean.columns and df_clean[col].isnull().sum() > 0:
                median_value = df_clean[col].median()
                df_clean[col].fillna(median_value, inplace=True)
                logger.info(f"Filled {col} missing values with median: {median_value}")
        
        # Handle categorical columns - fill with mode or 'Unknown'
        for col in data_config.categorical_columns:
            if col in df_clean.columns and df_clean[col].isnull().sum() > 0:
                if df_clean[col].mode().empty:
                    fill_value = 'Unknown'
                else:
                    fill_value = df_clean[col].mode()[0]
                df_clean[col].fillna(fill_value, inplace=True)
                logger.info(f"Filled {col} missing values with: {fill_value}")
        
        logger.info("Missing value handling completed")
        return df_clean
        
    except Exception as e:
        raise DataCleaningError(f"Error handling missing values: {str(e)}", cleaning_step="missing_values") from e

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names and structure.
    
    Args:
        df: DataFrame with potentially inconsistent column names
        
    Returns:
        DataFrame with standardized columns
    """
    try:
        log_analysis_step("Standardizing column names")
        
        df_clean = df.copy()
        
        # Remove leading/trailing whitespace from column names
        df_clean.columns = df_clean.columns.str.strip()
        
        # Log original column names for reference
        logger.info(f"Original columns: {list(df.columns)}")
        
        # Standardize specific column names if needed
        column_mapping = {
            'NationalITy': 'Nationality',
            'PlaceofBirth': 'PlaceOfBirth',
            'VisITedResources': 'VisitedResources',
            'raisedhands': 'RaisedHands'
        }
        
        df_clean.rename(columns=column_mapping, inplace=True)
        logger.info(f"Standardized columns: {list(df_clean.columns)}")
        
        return df_clean
        
    except Exception as e:
        logger.error(f"Error standardizing columns: {e}")
        raise

def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert columns to appropriate data types for analysis.
    
    Args:
        df: DataFrame with potentially incorrect data types
        
    Returns:
        DataFrame with corrected data types
    """
    try:
        log_analysis_step("Converting data types")
        
        df_clean = df.copy()
        
        # Convert numeric columns
        numeric_columns_to_convert = ['RaisedHands', 'VisitedResources', 'AnnouncementsView', 'Discussion']
        for col in numeric_columns_to_convert:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                logger.info(f"Converted {col} to numeric type")
        
        # Convert categorical columns to category type for memory efficiency
        categorical_columns_to_convert = ['gender', 'Nationality', 'StageID', 'GradeID', 'SectionID', 
                                        'Topic', 'Semester', 'Relation', 'Class']
        for col in categorical_columns_to_convert:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype('category')
                logger.info(f"Converted {col} to category type")
        
        logger.info("Data type conversion completed")
        return df_clean
        
    except Exception as e:
        logger.error(f"Error converting data types: {e}")
        raise

@log_function()
def clean_categorical_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize categorical data values.
    
    Args:
        df: DataFrame with categorical columns to clean
        
    Returns:
        DataFrame with cleaned categorical data
    """
    try:
        logger.info("Cleaning categorical data")
        
        df_clean = df.copy()
        
        # Standardize gender values
        if 'gender' in df_clean.columns:
            df_clean['gender'] = df_clean['gender'].str.upper()
            logger.info(f"Gender values: {df_clean['gender'].unique()}")
        
        # Standardize semester values
        if 'Semester' in df_clean.columns:
            df_clean['Semester'] = df_clean['Semester'].str.upper()
            logger.info(f"Semester values: {df_clean['Semester'].unique()}")
        
        # Clean absence days - ensure consistent format
        if 'StudentAbsenceDays' in df_clean.columns:
            # Standardize absence day categories
            absence_mapping = {
                'under-7': 'Under-7',
                'above-7': 'Above-7',
                'Under-7': 'Under-7',
                'Above-7': 'Above-7'
            }
            df_clean['StudentAbsenceDays'] = df_clean['StudentAbsenceDays'].map(absence_mapping).fillna(df_clean['StudentAbsenceDays'])
            logger.info(f"Absence days values: {df_clean['StudentAbsenceDays'].unique()}")
        
        # Standardize performance class
        if 'Class' in df_clean.columns:
            df_clean['Class'] = df_clean['Class'].str.upper()
            logger.info(f"Performance classes: {df_clean['Class'].unique()}")
        
        logger.info("Categorical data cleaning completed")
        return df_clean
        
    except Exception as e:
        logger.error(f"Error cleaning categorical data: {e}")
        raise

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features that will be useful for analysis.
    
    Args:
        df: Cleaned DataFrame
        
    Returns:
        DataFrame with additional derived features
    """
    try:
        log_analysis_step("Adding derived features")
        
        df_enhanced = df.copy()
        
        # Add total engagement score
        engagement_cols = ['RaisedHands', 'VisitedResources', 'AnnouncementsView', 'Discussion']
        available_engagement_cols = [col for col in engagement_cols if col in df_enhanced.columns]
        
        if available_engagement_cols:
            df_enhanced['TotalEngagement'] = df_enhanced[available_engagement_cols].sum(axis=1)
            df_enhanced['AvgEngagement'] = df_enhanced[available_engagement_cols].mean(axis=1)
            logger.info("Added engagement metrics")
        
        # Add performance category mapping
        if 'Class' in df_enhanced.columns:
            df_enhanced['PerformanceCategory'] = df_enhanced['Class'].map(PERFORMANCE_MAPPING)
            logger.info("Added performance category mapping")
        
        # Add risk indicator based on absence days
        if 'StudentAbsenceDays' in df_enhanced.columns:
            df_enhanced['HighAbsenceRisk'] = (df_enhanced['StudentAbsenceDays'] == 'Above-7').astype(int)
            logger.info("Added high absence risk indicator")
        
        # Add parent satisfaction indicator
        if 'ParentschoolSatisfaction' in df_enhanced.columns:
            df_enhanced['ParentSatisfied'] = (df_enhanced['ParentschoolSatisfaction'] == 'Good').astype(int)
            logger.info("Added parent satisfaction indicator")
        
        logger.info(f"Derived features added. New shape: {df_enhanced.shape}")
        return df_enhanced
        
    except Exception as e:
        logger.error(f"Error adding derived features: {e}")
        raise

def get_data_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a comprehensive data quality report.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary containing data quality metrics
    """
    try:
        log_analysis_step("Generating data quality report")
        
        report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        # Add categorical column analysis
        categorical_analysis = {}
        for col in df.select_dtypes(include=['category', 'object']).columns:
            categorical_analysis[col] = {
                'unique_values': df[col].nunique(),
                'most_common': df[col].mode().iloc[0] if not df[col].mode().empty else None,
                'unique_value_list': df[col].unique().tolist()[:10]  # First 10 unique values
            }
        report['categorical_analysis'] = categorical_analysis
        
        # Add numeric column analysis
        numeric_analysis = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            numeric_analysis[col] = get_numeric_summary(df[col])
        report['numeric_analysis'] = numeric_analysis
        
        logger.info("Data quality report generated successfully")
        return report
        
    except Exception as e:
        logger.error(f"Error generating data quality report: {e}")
        return {}

if __name__ == "__main__":
    # Test the data cleaner with sample data
    try:
        from data_loader import load_and_validate_data
        
        # Load data
        df = load_and_validate_data()
        print("Original data loaded successfully!")
        print(f"Original shape: {df.shape}")
        
        # Clean data
        df_clean = clean_student_data(df)
        print(f"Cleaned data shape: {df_clean.shape}")
        print(f"Cleaned columns: {list(df_clean.columns)}")
        
        # Generate quality report
        quality_report = get_data_quality_report(df_clean)
        print(f"Quality report generated: {len(quality_report)} metrics")
        
    except Exception as e:
        print(f"Error testing data cleaner: {e}")
