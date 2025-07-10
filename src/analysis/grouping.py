"""
Grouping and segmentation module using Pandas for the Academic Data Analysis System.
Handles all data grouping operations by different categories and criteria.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional
import sys
import os

# Add the parent directory to the path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import PERFORMANCE_MAPPING, RISK_ABSENCE_THRESHOLD
from src.utils.helpers import log_analysis_step, format_percentage

logger = logging.getLogger(__name__)

def group_by_career(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group students by career/topic and calculate performance metrics.
    
    Args:
        df: DataFrame containing student data
        
    Returns:
        DataFrame with career-based groupings and statistics
    """
    try:
        log_analysis_step("Grouping data by career/topic")
        
        if 'Topic' not in df.columns:
            logger.warning("Topic column not found for career grouping")
            return pd.DataFrame()
        
        # Group by Topic/Career
        career_groups = df.groupby('Topic').agg({
            'Class': ['count', lambda x: (x == 'H').sum(), lambda x: (x == 'M').sum(), lambda x: (x == 'L').sum()],
            'TotalEngagement': ['mean', 'std', 'min', 'max'] if 'TotalEngagement' in df.columns else 'count',
            'StudentAbsenceDays': lambda x: (x == 'Above-7').sum() if 'StudentAbsenceDays' in df.columns else 'count'
        }).round(2)
        
        # Flatten column names
        career_groups.columns = ['_'.join(col).strip() if col[1] else col[0] for col in career_groups.columns]
        
        # Rename columns for clarity
        new_names = {
            'Class_count': 'Total_Students',
            'Class_<lambda_0>': 'High_Performance',
            'Class_<lambda_1>': 'Medium_Performance', 
            'Class_<lambda_2>': 'Low_Performance'
        }
        
        # Handle engagement columns if they exist
        if 'TotalEngagement' in df.columns:
            new_names.update({
                'TotalEngagement_mean': 'Avg_Engagement',
                'TotalEngagement_std': 'Std_Engagement',
                'TotalEngagement_min': 'Min_Engagement',
                'TotalEngagement_max': 'Max_Engagement'
            })
        
        # Handle absence columns if they exist
        if 'StudentAbsenceDays' in df.columns:
            new_names.update({
                'StudentAbsenceDays_<lambda>': 'High_Absence_Count'
            })
        
        # Apply renaming for existing columns
        existing_renames = {k: v for k, v in new_names.items() if k in career_groups.columns}
        career_groups.rename(columns=existing_renames, inplace=True)
        
        # Calculate percentages
        if 'Total_Students' in career_groups.columns:
            for perf_level in ['High_Performance', 'Medium_Performance', 'Low_Performance']:
                if perf_level in career_groups.columns:
                    perf_pct = perf_level.replace('_Performance', '_Performance_Pct')
                    career_groups[perf_pct] = (career_groups[perf_level] / career_groups['Total_Students'] * 100).round(2)
        
        logger.info(f"Career grouping completed for {len(career_groups)} careers")
        return career_groups.reset_index()
        
    except Exception as e:
        logger.error(f"Error grouping by career: {e}")
        return pd.DataFrame()

def group_by_semester(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group students by semester and calculate performance metrics.
    
    Args:
        df: DataFrame containing student data
        
    Returns:
        DataFrame with semester-based groupings and statistics
    """
    try:
        log_analysis_step("Grouping data by semester")
        
        if 'Semester' not in df.columns:
            logger.warning("Semester column not found for semester grouping")
            return pd.DataFrame()
        
        # Group by Semester
        semester_groups = df.groupby('Semester').agg({
            'Class': ['count', lambda x: (x == 'H').sum(), lambda x: (x == 'M').sum(), lambda x: (x == 'L').sum()],
            'TotalEngagement': ['mean', 'std'] if 'TotalEngagement' in df.columns else 'count',
            'RaisedHands': 'mean' if 'RaisedHands' in df.columns else 'count',
            'Discussion': 'mean' if 'Discussion' in df.columns else 'count',
            'VisitedResources': 'mean' if 'VisitedResources' in df.columns else 'count'
        }).round(2)
        
        # Flatten column names
        semester_groups.columns = ['_'.join(col).strip() if col[1] else col[0] for col in semester_groups.columns]
        
        # Rename columns for clarity
        rename_mapping = {
            'Class_count': 'Total_Students',
            'Class_<lambda_0>': 'High_Performance_Count',
            'Class_<lambda_1>': 'Medium_Performance_Count',
            'Class_<lambda_2>': 'Low_Performance_Count'
        }
        
        if 'TotalEngagement' in df.columns:
            rename_mapping.update({
                'TotalEngagement_mean': 'Avg_Total_Engagement',
                'TotalEngagement_std': 'Std_Total_Engagement'
            })
        
        for col in ['RaisedHands', 'Discussion', 'VisitedResources']:
            if f'{col}_mean' in semester_groups.columns:
                rename_mapping[f'{col}_mean'] = f'Avg_{col}'
        
        existing_renames = {k: v for k, v in rename_mapping.items() if k in semester_groups.columns}
        semester_groups.rename(columns=existing_renames, inplace=True)
        
        logger.info(f"Semester grouping completed for {len(semester_groups)} semesters")
        return semester_groups.reset_index()
        
    except Exception as e:
        logger.error(f"Error grouping by semester: {e}")
        return pd.DataFrame()

def group_by_teacher(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group students by teacher/relation and calculate performance metrics.
    
    Args:
        df: DataFrame containing student data
        
    Returns:
        DataFrame with teacher-based groupings and statistics
    """
    try:
        log_analysis_step("Grouping data by teacher/relation")
        
        if 'Relation' not in df.columns:
            logger.warning("Relation column not found for teacher grouping")
            return pd.DataFrame()
        
        # Group by Relation (assuming this represents teacher relationship)
        teacher_groups = df.groupby('Relation').agg({
            'Class': ['count', lambda x: (x == 'H').sum()],
            'ParentschoolSatisfaction': lambda x: (x == 'Good').sum() if 'ParentschoolSatisfaction' in df.columns else 'count',
            'ParentAnsweringSurvey': lambda x: (x == 'Yes').sum() if 'ParentAnsweringSurvey' in df.columns else 'count',
            'TotalEngagement': 'mean' if 'TotalEngagement' in df.columns else 'count'
        }).round(2)
        
        # Flatten column names
        teacher_groups.columns = ['_'.join(col).strip() if col[1] else col[0] for col in teacher_groups.columns]
        
        # Rename columns
        rename_mapping = {
            'Class_count': 'Total_Students',
            'Class_<lambda>': 'High_Performance_Count'
        }
        
        if 'ParentschoolSatisfaction' in df.columns:
            rename_mapping['ParentschoolSatisfaction_<lambda>'] = 'Satisfied_Parents_Count'
        
        if 'ParentAnsweringSurvey' in df.columns:
            rename_mapping['ParentAnsweringSurvey_<lambda>'] = 'Survey_Responses_Count'
        
        if 'TotalEngagement' in df.columns:
            rename_mapping['TotalEngagement_mean'] = 'Avg_Engagement'
        
        existing_renames = {k: v for k, v in rename_mapping.items() if k in teacher_groups.columns}
        teacher_groups.rename(columns=existing_renames, inplace=True)
        
        # Calculate satisfaction rates
        if all(col in teacher_groups.columns for col in ['Total_Students', 'Satisfied_Parents_Count']):
            teacher_groups['Parent_Satisfaction_Rate'] = (
                teacher_groups['Satisfied_Parents_Count'] / teacher_groups['Total_Students'] * 100
            ).round(2)
        
        logger.info(f"Teacher grouping completed for {len(teacher_groups)} relation types")
        return teacher_groups.reset_index()
        
    except Exception as e:
        logger.error(f"Error grouping by teacher: {e}")
        return pd.DataFrame()

def group_by_subject(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group students by subject with detailed analysis.
    
    Args:
        df: DataFrame containing student data
        
    Returns:
        DataFrame with subject-based detailed groupings
    """
    try:
        log_analysis_step("Grouping data by subject with detailed analysis")
        
        if 'Topic' not in df.columns:
            logger.warning("Topic column not found for subject grouping")
            return pd.DataFrame()
        
        # More detailed subject analysis
        subject_groups = df.groupby('Topic').agg({
            'Class': ['count', 'nunique'],
            'gender': lambda x: x.value_counts().to_dict() if 'gender' in df.columns else 'count',
            'StageID': lambda x: x.value_counts().to_dict() if 'StageID' in df.columns else 'count',
            'TotalEngagement': ['mean', 'median', 'std'] if 'TotalEngagement' in df.columns else 'count',
            'StudentAbsenceDays': lambda x: (x == 'Above-7').sum() if 'StudentAbsenceDays' in df.columns else 'count'
        })
        
        # Create a more detailed analysis per subject
        detailed_results = []
        
        for subject in df['Topic'].unique():
            if pd.isna(subject):
                continue
                
            subject_data = df[df['Topic'] == subject]
            
            result = {
                'Subject': subject,
                'Total_Students': len(subject_data),
                'Performance_Distribution': subject_data['Class'].value_counts().to_dict() if 'Class' in subject_data.columns else {}
            }
            
            # Add gender distribution if available
            if 'gender' in subject_data.columns:
                result['Gender_Distribution'] = subject_data['gender'].value_counts().to_dict()
            
            # Add engagement statistics if available
            if 'TotalEngagement' in subject_data.columns:
                engagement_data = subject_data['TotalEngagement'].dropna()
                if len(engagement_data) > 0:
                    result.update({
                        'Avg_Engagement': float(engagement_data.mean()),
                        'Median_Engagement': float(engagement_data.median()),
                        'Std_Engagement': float(engagement_data.std())
                    })
            
            # Add absence analysis if available
            if 'StudentAbsenceDays' in subject_data.columns:
                high_absence = (subject_data['StudentAbsenceDays'] == 'Above-7').sum()
                result['High_Absence_Count'] = high_absence
                result['High_Absence_Rate'] = format_percentage(high_absence / len(subject_data))
            
            detailed_results.append(result)
        
        logger.info(f"Detailed subject grouping completed for {len(detailed_results)} subjects")
        return pd.DataFrame(detailed_results)
        
    except Exception as e:
        logger.error(f"Error in detailed subject grouping: {e}")
        return pd.DataFrame()

def segment_by_performance(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Segment students by performance level and analyze each segment.
    
    Args:
        df: DataFrame containing student data
        
    Returns:
        Dictionary with performance levels as keys and DataFrames as values
    """
    try:
        log_analysis_step("Segmenting students by performance level")
        
        if 'Class' not in df.columns:
            logger.warning("Class column not found for performance segmentation")
            return {}
        
        performance_segments = {}
        
        for performance_level in df['Class'].unique():
            if pd.isna(performance_level):
                continue
                
            segment_data = df[df['Class'] == performance_level].copy()
            
            # Add segment analysis
            segment_analysis = {
                'segment_size': len(segment_data),
                'percentage_of_total': format_percentage(len(segment_data) / len(df))
            }
            
            # Add engagement analysis if available
            engagement_cols = ['RaisedHands', 'VisitedResources', 'AnnouncementsView', 'Discussion']
            available_engagement = [col for col in engagement_cols if col in segment_data.columns]
            
            if available_engagement:
                for col in available_engagement:
                    values = segment_data[col].dropna()
                    if len(values) > 0:
                        segment_analysis[f'avg_{col.lower()}'] = float(values.mean())
            
            # Add demographic analysis
            if 'gender' in segment_data.columns:
                segment_analysis['gender_distribution'] = segment_data['gender'].value_counts().to_dict()
            
            if 'Nationality' in segment_data.columns:
                segment_analysis['nationality_distribution'] = segment_data['Nationality'].value_counts().head().to_dict()
            
            # Store the segment with its analysis
            segment_data.attrs = segment_analysis
            performance_segments[performance_level] = segment_data
            
            logger.info(f"Performance segment '{performance_level}': {len(segment_data)} students")
        
        return performance_segments
        
    except Exception as e:
        logger.error(f"Error segmenting by performance: {e}")
        return {}

def group_by_demographics(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Group students by demographic characteristics.
    
    Args:
        df: DataFrame containing student data
        
    Returns:
        Dictionary with demographic group analyses
    """
    try:
        log_analysis_step("Grouping by demographic characteristics")
        
        demographic_groups = {}
        
        # Gender-based grouping
        if 'gender' in df.columns:
            gender_analysis = df.groupby('gender').agg({
                'Class': lambda x: x.value_counts().to_dict(),
                'TotalEngagement': 'mean' if 'TotalEngagement' in df.columns else 'count',
                'StudentAbsenceDays': lambda x: (x == 'Above-7').sum() if 'StudentAbsenceDays' in df.columns else 'count'
            })
            demographic_groups['gender'] = gender_analysis
        
        # Nationality-based grouping
        if 'Nationality' in df.columns:
            nationality_analysis = df.groupby('Nationality').agg({
                'Class': ['count', lambda x: (x == 'H').sum()],
                'TotalEngagement': 'mean' if 'TotalEngagement' in df.columns else 'count'
            })
            demographic_groups['nationality'] = nationality_analysis
        
        # Stage-based grouping
        if 'StageID' in df.columns:
            stage_analysis = df.groupby('StageID').agg({
                'Class': ['count', lambda x: x.value_counts().to_dict()],
                'TotalEngagement': ['mean', 'std'] if 'TotalEngagement' in df.columns else 'count'
            })
            demographic_groups['stage'] = stage_analysis
        
        logger.info(f"Demographic grouping completed for {len(demographic_groups)} categories")
        return demographic_groups
        
    except Exception as e:
        logger.error(f"Error in demographic grouping: {e}")
        return {}

def perform_all_groupings(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform all grouping operations and return comprehensive results.
    
    Args:
        df: Cleaned DataFrame ready for analysis
        
    Returns:
        Dictionary containing all grouping results
    """
    try:
        log_analysis_step("Performing all grouping operations")
        
        all_groupings = {}
        
        # Perform each type of grouping
        all_groupings['by_career'] = group_by_career(df)
        all_groupings['by_semester'] = group_by_semester(df)
        all_groupings['by_teacher'] = group_by_teacher(df)
        all_groupings['by_subject_detailed'] = group_by_subject(df)
        all_groupings['by_performance'] = segment_by_performance(df)
        all_groupings['by_demographics'] = group_by_demographics(df)
        
        # Generate summary statistics
        grouping_summary = {
            'total_students': len(df),
            'unique_subjects': df['Topic'].nunique() if 'Topic' in df.columns else 0,
            'unique_semesters': df['Semester'].nunique() if 'Semester' in df.columns else 0,
            'performance_distribution': df['Class'].value_counts().to_dict() if 'Class' in df.columns else {}
        }
        
        all_groupings['summary'] = grouping_summary
        
        logger.info("All grouping operations completed successfully")
        return all_groupings
        
    except Exception as e:
        logger.error(f"Error performing all groupings: {e}")
        return {}

if __name__ == "__main__":
    # Test the grouping module
    try:
        from src.data.data_loader import load_and_validate_data
        from src.data.data_cleaner import clean_student_data
        
        # Load and clean data
        df = load_and_validate_data()
        df_clean = clean_student_data(df)
        
        print("Data loaded and cleaned successfully!")
        
        # Perform groupings
        all_groupings = perform_all_groupings(df_clean)
        print(f"Groupings completed: {list(all_groupings.keys())}")
        
        # Display some results
        if 'summary' in all_groupings:
            print(f"Summary: {all_groupings['summary']}")
        
    except Exception as e:
        print(f"Error testing grouping module: {e}")
