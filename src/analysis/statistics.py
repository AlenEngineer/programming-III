"""
Statistical analysis module using NumPy for the Academic Data Analysis System.
Handles all statistical calculations including averages, standard deviations, and distributions.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Union
import sys
import os

# Add the parent directory to the path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import NUMERIC_COLUMNS, MINIMUM_PASSING_GRADE, EXCELLENT_GRADE
from src.utils.helpers import log_analysis_step, safe_divide, format_percentage

logger = logging.getLogger(__name__)

def calculate_overall_average(df: pd.DataFrame, column: str = 'TotalEngagement') -> float:
    """
    Calculate the overall average for a specified numeric column.
    
    Args:
        df: DataFrame containing the data
        column: Column name to calculate average for
        
    Returns:
        Overall average value
    """
    try:
        log_analysis_step(f"Calculating overall average for {column}")
        
        if column not in df.columns:
            logger.warning(f"Column {column} not found in DataFrame")
            return 0.0
        
        # Use NumPy for calculation
        values = df[column].dropna().to_numpy()
        if len(values) == 0:
            logger.warning(f"No valid values found in column {column}")
            return 0.0
        
        average = np.mean(values)
        logger.info(f"Overall average for {column}: {average:.2f}")
        return float(average)
        
    except Exception as e:
        logger.error(f"Error calculating overall average for {column}: {e}")
        return 0.0

def calculate_standard_deviation(df: pd.DataFrame, column: str = 'TotalEngagement') -> float:
    """
    Calculate the standard deviation for a specified numeric column.
    
    Args:
        df: DataFrame containing the data
        column: Column name to calculate standard deviation for
        
    Returns:
        Standard deviation value
    """
    try:
        log_analysis_step(f"Calculating standard deviation for {column}")
        
        if column not in df.columns:
            logger.warning(f"Column {column} not found in DataFrame")
            return 0.0
        
        # Use NumPy for calculation
        values = df[column].dropna().to_numpy()
        if len(values) < 2:
            logger.warning(f"Insufficient values for standard deviation calculation in {column}")
            return 0.0
        
        std_dev = np.std(values, ddof=1)  # Sample standard deviation
        logger.info(f"Standard deviation for {column}: {std_dev:.2f}")
        return float(std_dev)
        
    except Exception as e:
        logger.error(f"Error calculating standard deviation for {column}: {e}")
        return 0.0

def get_subject_averages(df: pd.DataFrame, engagement_column: str = 'TotalEngagement') -> Dict[str, float]:
    """
    Calculate average engagement/performance by subject/topic.
    
    Args:
        df: DataFrame containing the data
        engagement_column: Column to calculate averages for
        
    Returns:
        Dictionary with subjects as keys and averages as values
    """
    try:
        log_analysis_step("Calculating subject averages")
        
        if 'Topic' not in df.columns or engagement_column not in df.columns:
            logger.warning("Required columns not found for subject averages")
            return {}
        
        subject_averages = {}
        for subject in df['Topic'].unique():
            if pd.isna(subject):
                continue
                
            subject_data = df[df['Topic'] == subject][engagement_column].dropna()
            if len(subject_data) > 0:
                avg = np.mean(subject_data.to_numpy())
                subject_averages[subject] = float(avg)
                logger.info(f"Subject {subject} average: {avg:.2f}")
        
        return subject_averages
        
    except Exception as e:
        logger.error(f"Error calculating subject averages: {e}")
        return {}

def get_semester_averages(df: pd.DataFrame, engagement_column: str = 'TotalEngagement') -> Dict[str, float]:
    """
    Calculate average engagement/performance by semester.
    
    Args:
        df: DataFrame containing the data
        engagement_column: Column to calculate averages for
        
    Returns:
        Dictionary with semesters as keys and averages as values
    """
    try:
        log_analysis_step("Calculating semester averages")
        
        if 'Semester' not in df.columns or engagement_column not in df.columns:
            logger.warning("Required columns not found for semester averages")
            return {}
        
        semester_averages = {}
        for semester in df['Semester'].unique():
            if pd.isna(semester):
                continue
                
            semester_data = df[df['Semester'] == semester][engagement_column].dropna()
            if len(semester_data) > 0:
                avg = np.mean(semester_data.to_numpy())
                semester_averages[semester] = float(avg)
                logger.info(f"Semester {semester} average: {avg:.2f}")
        
        return semester_averages
        
    except Exception as e:
        logger.error(f"Error calculating semester averages: {e}")
        return {}

def get_student_averages(df: pd.DataFrame, engagement_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Calculate individual student averages across engagement metrics.
    
    Args:
        df: DataFrame containing the data
        engagement_columns: List of columns to include in student averages
        
    Returns:
        DataFrame with student indices and their average scores
    """
    try:
        log_analysis_step("Calculating individual student averages")
        
        if engagement_columns is None:
            engagement_columns = ['RaisedHands', 'VisitedResources', 'AnnouncementsView', 'Discussion']
        
        # Filter to existing columns
        available_columns = [col for col in engagement_columns if col in df.columns]
        
        if not available_columns:
            logger.warning("No engagement columns found for student averages")
            return pd.DataFrame()
        
        # Calculate student averages
        student_data = df[available_columns].copy()
        student_averages = np.mean(student_data.to_numpy(), axis=1)
        
        result_df = pd.DataFrame({
            'StudentIndex': df.index,
            'AvgEngagement': student_averages,
            'Class': df['Class'] if 'Class' in df.columns else 'Unknown'
        })
        
        logger.info(f"Calculated averages for {len(result_df)} students")
        return result_df
        
    except Exception as e:
        logger.error(f"Error calculating student averages: {e}")
        return pd.DataFrame()

def calculate_performance_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate comprehensive performance statistics for the dataset.
    
    Args:
        df: DataFrame containing the data
        
    Returns:
        Dictionary containing various performance statistics
    """
    try:
        log_analysis_step("Calculating comprehensive performance statistics")
        
        stats = {}
        
        # Overall class distribution
        if 'Class' in df.columns:
            class_counts = df['Class'].value_counts().to_dict()
            total_students = len(df)
            class_percentages = {k: format_percentage(v/total_students) for k, v in class_counts.items()}
            
            stats['class_distribution'] = {
                'counts': class_counts,
                'percentages': class_percentages
            }
        
        # Engagement statistics
        engagement_cols = ['RaisedHands', 'VisitedResources', 'AnnouncementsView', 'Discussion']
        available_engagement = [col for col in engagement_cols if col in df.columns]
        
        if available_engagement:
            engagement_stats = {}
            for col in available_engagement:
                values = df[col].dropna().to_numpy()
                if len(values) > 0:
                    engagement_stats[col] = {
                        'mean': float(np.mean(values)),
                        'median': float(np.median(values)),
                        'std': float(np.std(values, ddof=1)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'q25': float(np.percentile(values, 25)),
                        'q75': float(np.percentile(values, 75))
                    }
            stats['engagement_statistics'] = engagement_stats
        
        # Gender-based analysis
        if 'gender' in df.columns:
            gender_stats = {}
            for gender in df['gender'].unique():
                if pd.isna(gender):
                    continue
                gender_data = df[df['gender'] == gender]
                
                if 'TotalEngagement' in gender_data.columns:
                    values = gender_data['TotalEngagement'].dropna().to_numpy()
                    if len(values) > 0:
                        gender_stats[gender] = {
                            'count': len(gender_data),
                            'avg_engagement': float(np.mean(values)),
                            'performance_distribution': gender_data['Class'].value_counts().to_dict() if 'Class' in gender_data.columns else {}
                        }
            stats['gender_analysis'] = gender_stats
        
        # Attendance analysis
        if 'StudentAbsenceDays' in df.columns:
            attendance_stats = df['StudentAbsenceDays'].value_counts().to_dict()
            total = len(df)
            attendance_percentages = {k: format_percentage(v/total) for k, v in attendance_stats.items()}
            
            stats['attendance_analysis'] = {
                'distribution': attendance_stats,
                'percentages': attendance_percentages
            }
        
        logger.info("Comprehensive performance statistics calculated successfully")
        return stats
        
    except Exception as e:
        logger.error(f"Error calculating performance statistics: {e}")
        return {}

def calculate_correlation_matrix(df: pd.DataFrame, columns: Optional[List[str]] = None) -> np.ndarray:
    """
    Calculate correlation matrix for numeric columns using NumPy.
    
    Args:
        df: DataFrame containing the data
        columns: List of columns to include in correlation analysis
        
    Returns:
        Correlation matrix as NumPy array
    """
    try:
        log_analysis_step("Calculating correlation matrix")
        
        if columns is None:
            columns = ['RaisedHands', 'VisitedResources', 'AnnouncementsView', 'Discussion']
        
        # Filter to existing numeric columns
        available_columns = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        
        if len(available_columns) < 2:
            logger.warning("Insufficient numeric columns for correlation analysis")
            return np.array([])
        
        # Create correlation matrix using NumPy
        data_matrix = df[available_columns].dropna().to_numpy()
        correlation_matrix = np.corrcoef(data_matrix.T)
        
        logger.info(f"Correlation matrix calculated for {len(available_columns)} variables")
        return correlation_matrix
        
    except Exception as e:
        logger.error(f"Error calculating correlation matrix: {e}")
        return np.array([])

def calculate_z_scores(df: pd.DataFrame, column: str) -> np.ndarray:
    """
    Calculate z-scores for a numeric column to identify outliers.
    
    Args:
        df: DataFrame containing the data
        column: Column name to calculate z-scores for
        
    Returns:
        Array of z-scores
    """
    try:
        log_analysis_step(f"Calculating z-scores for {column}")
        
        if column not in df.columns:
            logger.warning(f"Column {column} not found")
            return np.array([])
        
        values = df[column].dropna().to_numpy()
        if len(values) < 2:
            logger.warning(f"Insufficient data for z-score calculation in {column}")
            return np.array([])
        
        # Calculate z-scores using NumPy
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)
        
        if std_val == 0:
            logger.warning(f"Zero standard deviation in {column}, cannot calculate z-scores")
            return np.zeros(len(values))
        
        z_scores = (values - mean_val) / std_val
        
        # Log outliers (|z| > 3)
        outliers = np.abs(z_scores) > 3
        outlier_count = np.sum(outliers)
        
        logger.info(f"Z-scores calculated for {column}. Found {outlier_count} outliers")
        return z_scores
        
    except Exception as e:
        logger.error(f"Error calculating z-scores for {column}: {e}")
        return np.array([])

def calculate_all_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate all statistical measures for the dataset.
    
    Args:
        df: Cleaned DataFrame ready for analysis
        
    Returns:
        Comprehensive dictionary of all statistical measures
    """
    try:
        log_analysis_step("Calculating all statistical measures")
        
        all_stats = {}
        
        # Basic statistics
        if 'TotalEngagement' in df.columns:
            all_stats['overall_average'] = calculate_overall_average(df, 'TotalEngagement')
            all_stats['overall_std'] = calculate_standard_deviation(df, 'TotalEngagement')
        
        # Group-based averages
        all_stats['subject_averages'] = get_subject_averages(df)
        all_stats['semester_averages'] = get_semester_averages(df)
        
        # Performance statistics
        all_stats['performance_stats'] = calculate_performance_statistics(df)
        
        # Student-level statistics
        student_stats = get_student_averages(df)
        if not student_stats.empty:
            all_stats['student_statistics'] = {
                'total_students': len(student_stats),
                'avg_student_engagement': float(np.mean(student_stats['AvgEngagement'].to_numpy())),
                'std_student_engagement': float(np.std(student_stats['AvgEngagement'].to_numpy(), ddof=1))
            }
        
        # Correlation analysis
        correlation_matrix = calculate_correlation_matrix(df)
        if correlation_matrix.size > 0:
            all_stats['correlation_matrix'] = correlation_matrix.tolist()
        
        logger.info("All statistical measures calculated successfully")
        return all_stats
        
    except Exception as e:
        logger.error(f"Error calculating all statistics: {e}")
        return {}

if __name__ == "__main__":
    # Test the statistics module
    try:
        from src.data.data_loader import load_and_validate_data
        from src.data.data_cleaner import clean_student_data
        
        # Load and clean data
        df = load_and_validate_data()
        df_clean = clean_student_data(df)
        
        print("Data loaded and cleaned successfully!")
        
        # Calculate statistics
        all_stats = calculate_all_statistics(df_clean)
        print(f"Statistics calculated: {list(all_stats.keys())}")
        
        # Display some key statistics
        if 'overall_average' in all_stats:
            print(f"Overall average engagement: {all_stats['overall_average']:.2f}")
        
        if 'subject_averages' in all_stats:
            print(f"Subject averages: {all_stats['subject_averages']}")
        
    except Exception as e:
        print(f"Error testing statistics module: {e}")
