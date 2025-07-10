"""
Risk analysis module for the Academic Data Analysis System.
Identifies students at risk based on various academic and behavioral indicators.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
import sys
import os

# Add the parent directory to the path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import MINIMUM_PASSING_GRADE, RISK_ABSENCE_THRESHOLD, LOW_PARTICIPATION_THRESHOLD
from src.utils.helpers import log_analysis_step, format_percentage

logger = logging.getLogger(__name__)

def identify_at_risk_students(df: pd.DataFrame, min_engagement: Optional[float] = None) -> pd.DataFrame:
    """
    Identify students at risk based on multiple criteria.
    
    Args:
        df: DataFrame containing student data
        min_engagement: Minimum engagement threshold (uses config default if None)
        
    Returns:
        DataFrame containing at-risk students with risk factors
    """
    try:
        log_analysis_step("Identifying at-risk students")
        
        if min_engagement is None:
            min_engagement = LOW_PARTICIPATION_THRESHOLD
        
        # Initialize risk factors tracking
        at_risk_students = df.copy()
        at_risk_students['RiskFactors'] = ''
        at_risk_students['RiskScore'] = 0
        at_risk_students['IsAtRisk'] = False
        
        # Risk Factor 1: Low performance (Class = 'L')
        if 'Class' in df.columns:
            low_performance = at_risk_students['Class'] == 'L'
            at_risk_students.loc[low_performance, 'RiskFactors'] += 'Low Performance; '
            at_risk_students.loc[low_performance, 'RiskScore'] += 3
            logger.info(f"Found {low_performance.sum()} students with low performance")
        
        # Risk Factor 2: High absence days
        if 'StudentAbsenceDays' in df.columns:
            high_absence = at_risk_students['StudentAbsenceDays'] == 'Above-7'
            at_risk_students.loc[high_absence, 'RiskFactors'] += 'High Absences; '
            at_risk_students.loc[high_absence, 'RiskScore'] += 2
            logger.info(f"Found {high_absence.sum()} students with high absences")
        
        # Risk Factor 3: Low total engagement
        if 'TotalEngagement' in df.columns:
            low_engagement = at_risk_students['TotalEngagement'] < min_engagement
            at_risk_students.loc[low_engagement, 'RiskFactors'] += 'Low Engagement; '
            at_risk_students.loc[low_engagement, 'RiskScore'] += 2
            logger.info(f"Found {low_engagement.sum()} students with low engagement (< {min_engagement})")
        
        # Risk Factor 4: Low individual participation metrics
        participation_cols = ['RaisedHands', 'Discussion']
        for col in participation_cols:
            if col in df.columns:
                low_participation = at_risk_students[col] < (min_engagement / 4)  # Quarter of total threshold
                at_risk_students.loc[low_participation, 'RiskFactors'] += f'Low {col}; '
                at_risk_students.loc[low_participation, 'RiskScore'] += 1
                logger.info(f"Found {low_participation.sum()} students with low {col}")
        
        # Risk Factor 5: Poor parent satisfaction
        if 'ParentschoolSatisfaction' in df.columns:
            poor_satisfaction = at_risk_students['ParentschoolSatisfaction'] == 'Bad'
            at_risk_students.loc[poor_satisfaction, 'RiskFactors'] += 'Poor Parent Satisfaction; '
            at_risk_students.loc[poor_satisfaction, 'RiskScore'] += 1
            logger.info(f"Found {poor_satisfaction.sum()} students with poor parent satisfaction")
        
        # Risk Factor 6: No parent survey response
        if 'ParentAnsweringSurvey' in df.columns:
            no_survey = at_risk_students['ParentAnsweringSurvey'] == 'No'
            at_risk_students.loc[no_survey, 'RiskFactors'] += 'No Parent Survey; '
            at_risk_students.loc[no_survey, 'RiskScore'] += 1
            logger.info(f"Found {no_survey.sum()} students with no parent survey response")
        
        # Determine overall risk status (risk score >= 3 considered at risk)
        at_risk_students['IsAtRisk'] = at_risk_students['RiskScore'] >= 3
        
        # Clean up risk factors string
        at_risk_students['RiskFactors'] = at_risk_students['RiskFactors'].str.rstrip('; ')
        
        # Filter to only at-risk students
        at_risk_only = at_risk_students[at_risk_students['IsAtRisk']].copy()
        
        logger.info(f"Identified {len(at_risk_only)} students at risk out of {len(df)} total students")
        return at_risk_only.sort_values('RiskScore', ascending=False)
        
    except Exception as e:
        logger.error(f"Error identifying at-risk students: {e}")
        return pd.DataFrame()

def analyze_attendance_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze attendance patterns and identify concerning trends.
    
    Args:
        df: DataFrame containing student data
        
    Returns:
        DataFrame with attendance analysis results
    """
    try:
        log_analysis_step("Analyzing attendance patterns")
        
        if 'StudentAbsenceDays' not in df.columns:
            logger.warning("StudentAbsenceDays column not found for attendance analysis")
            return pd.DataFrame()
        
        # Create attendance analysis
        attendance_analysis = []
        
        # Overall attendance distribution
        attendance_dist = df['StudentAbsenceDays'].value_counts()
        total_students = len(df)
        
        for absence_category, count in attendance_dist.items():
            percentage = (count / total_students) * 100
            attendance_analysis.append({
                'Category': f'Students with {absence_category} absence days',
                'Count': count,
                'Percentage': f'{percentage:.1f}%',
                'Analysis_Type': 'Overall Distribution'
            })
        
        # Attendance by subject
        if 'Topic' in df.columns:
            for subject in df['Topic'].unique():
                if pd.isna(subject):
                    continue
                    
                subject_data = df[df['Topic'] == subject]
                high_absence_count = (subject_data['StudentAbsenceDays'] == 'Above-7').sum()
                subject_total = len(subject_data)
                
                if subject_total > 0:
                    high_absence_rate = (high_absence_count / subject_total) * 100
                    attendance_analysis.append({
                        'Category': f'{subject} - High Absence Rate',
                        'Count': high_absence_count,
                        'Percentage': f'{high_absence_rate:.1f}%',
                        'Analysis_Type': 'By Subject'
                    })
        
        # Attendance by performance
        if 'Class' in df.columns:
            for performance in df['Class'].unique():
                if pd.isna(performance):
                    continue
                    
                perf_data = df[df['Class'] == performance]
                high_absence_count = (perf_data['StudentAbsenceDays'] == 'Above-7').sum()
                perf_total = len(perf_data)
                
                if perf_total > 0:
                    high_absence_rate = (high_absence_count / perf_total) * 100
                    attendance_analysis.append({
                        'Category': f'Performance {performance} - High Absence Rate',
                        'Count': high_absence_count,
                        'Percentage': f'{high_absence_rate:.1f}%',
                        'Analysis_Type': 'By Performance'
                    })
        
        # Attendance by gender
        if 'gender' in df.columns:
            for gender in df['gender'].unique():
                if pd.isna(gender):
                    continue
                    
                gender_data = df[df['gender'] == gender]
                high_absence_count = (gender_data['StudentAbsenceDays'] == 'Above-7').sum()
                gender_total = len(gender_data)
                
                if gender_total > 0:
                    high_absence_rate = (high_absence_count / gender_total) * 100
                    attendance_analysis.append({
                        'Category': f'Gender {gender} - High Absence Rate',
                        'Count': high_absence_count,
                        'Percentage': f'{high_absence_rate:.1f}%',
                        'Analysis_Type': 'By Gender'
                    })
        
        result_df = pd.DataFrame(attendance_analysis)
        logger.info(f"Attendance analysis completed with {len(result_df)} insights")
        return result_df
        
    except Exception as e:
        logger.error(f"Error analyzing attendance patterns: {e}")
        return pd.DataFrame()

def get_low_participation_students(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify students with consistently low participation across metrics.
    
    Args:
        df: DataFrame containing student data
        
    Returns:
        DataFrame containing students with low participation
    """
    try:
        log_analysis_step("Identifying low participation students")
        
        participation_cols = ['RaisedHands', 'VisitedResources', 'AnnouncementsView', 'Discussion']
        available_cols = [col for col in participation_cols if col in df.columns]
        
        if len(available_cols) < 2:
            logger.warning("Insufficient participation columns for analysis")
            return pd.DataFrame()
        
        # Calculate participation thresholds (bottom 25th percentile)
        thresholds = {}
        for col in available_cols:
            thresholds[col] = df[col].quantile(0.25)
        
        # Identify low participation students
        low_participation = df.copy()
        low_participation['LowParticipationCount'] = 0
        low_participation['LowParticipationAreas'] = ''
        
        for col in available_cols:
            is_low = low_participation[col] <= thresholds[col]
            
            # Use iloc for safer assignment
            for idx in low_participation.index[is_low]:
                low_participation.at[idx, 'LowParticipationCount'] += 1
                low_participation.at[idx, 'LowParticipationAreas'] += f'{col}; '
        
        # Consider students with low participation in 50% or more areas as concerning
        threshold_count = max(1, len(available_cols) // 2)
        concerning_students = low_participation[low_participation['LowParticipationCount'] >= threshold_count].copy()
        
        # Clean up areas string
        concerning_students['LowParticipationAreas'] = concerning_students['LowParticipationAreas'].str.rstrip('; ')
        
        # Add overall participation score
        if available_cols:
            concerning_students['OverallParticipationScore'] = concerning_students[available_cols].mean(axis=1)
        
        logger.info(f"Found {len(concerning_students)} students with low participation")
        return concerning_students.sort_values('LowParticipationCount', ascending=False)
        
    except Exception as e:
        logger.error(f"Error identifying low participation students: {e}")
        return pd.DataFrame()

def generate_risk_report(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a comprehensive risk assessment report.
    
    Args:
        df: DataFrame containing student data
        
    Returns:
        Dictionary containing comprehensive risk analysis
    """
    try:
        log_analysis_step("Generating comprehensive risk report")
        
        report = {
            'summary': {},
            'at_risk_students': {},
            'attendance_analysis': {},
            'participation_analysis': {},
            'recommendations': []
        }
        
        total_students = len(df)
        report['summary']['total_students'] = total_students
        
        # Get at-risk students
        at_risk_df = identify_at_risk_students(df)
        report['at_risk_students']['count'] = len(at_risk_df)
        report['at_risk_students']['percentage'] = format_percentage(len(at_risk_df) / total_students)
        
        if not at_risk_df.empty:
            # Risk score distribution
            risk_score_dist = at_risk_df['RiskScore'].value_counts().sort_index()
            report['at_risk_students']['risk_score_distribution'] = risk_score_dist.to_dict()
            
            # Most common risk factors
            all_factors = ' '.join(at_risk_df['RiskFactors'].fillna(''))
            factor_words = [factor.strip() for factor in all_factors.split(';') if factor.strip()]
            from collections import Counter
            factor_counts = Counter(factor_words)
            report['at_risk_students']['common_risk_factors'] = dict(factor_counts.most_common(5))
            
            # Subject-wise at-risk distribution
            if 'Topic' in at_risk_df.columns:
                subject_risk = at_risk_df['Topic'].value_counts().head(5).to_dict()
                report['at_risk_students']['subjects_with_most_at_risk'] = subject_risk
        
        # Attendance analysis
        attendance_df = analyze_attendance_patterns(df)
        if not attendance_df.empty:
            high_absence_overall = attendance_df[
                attendance_df['Category'].str.contains('Above-7')
            ].iloc[0] if len(attendance_df) > 0 else None
            
            if high_absence_overall is not None:
                report['attendance_analysis']['high_absence_rate'] = high_absence_overall['Percentage']
        
        # Participation analysis
        low_participation_df = get_low_participation_students(df)
        report['participation_analysis']['low_participation_count'] = len(low_participation_df)
        report['participation_analysis']['low_participation_percentage'] = format_percentage(
            len(low_participation_df) / total_students
        )
        
        # Performance correlation with risk factors
        if 'Class' in df.columns:
            performance_risk = {}
            for performance in df['Class'].unique():
                if pd.isna(performance):
                    continue
                perf_students = df[df['Class'] == performance]
                at_risk_in_perf = identify_at_risk_students(perf_students)
                risk_rate = len(at_risk_in_perf) / len(perf_students) if len(perf_students) > 0 else 0
                performance_risk[performance] = format_percentage(risk_rate)
            
            report['summary']['risk_by_performance'] = performance_risk
        
        # Generate recommendations
        recommendations = []
        
        if len(at_risk_df) > 0:
            recommendations.append(f"Immediate attention needed for {len(at_risk_df)} students identified as at-risk")
        
        if not attendance_df.empty:
            high_absence_rate = attendance_df[attendance_df['Category'].str.contains('Above-7')]
            if not high_absence_rate.empty:
                rate = high_absence_rate.iloc[0]['Percentage']
                recommendations.append(f"Address attendance issues - {rate} of students have high absence rates")
        
        if len(low_participation_df) > 0:
            recommendations.append(f"Implement engagement strategies for {len(low_participation_df)} students with low participation")
        
        # Subject-specific recommendations
        if 'Topic' in df.columns and len(at_risk_df) > 0:
            subject_risks = at_risk_df['Topic'].value_counts()
            if len(subject_risks) > 0:
                highest_risk_subject = subject_risks.index[0]
                recommendations.append(f"Focus intervention efforts on {highest_risk_subject} subject with highest at-risk student count")
        
        recommendations.append("Monitor parent engagement and satisfaction levels regularly")
        recommendations.append("Implement early warning system for students showing multiple risk factors")
        
        report['recommendations'] = recommendations
        
        logger.info("Comprehensive risk report generated successfully")
        return report
        
    except Exception as e:
        logger.error(f"Error generating risk report: {e}")
        return {}

def identify_intervention_priorities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify and prioritize students who need immediate intervention.
    
    Args:
        df: DataFrame containing student data
        
    Returns:
        DataFrame with prioritized intervention list
    """
    try:
        log_analysis_step("Identifying intervention priorities")
        
        # Get at-risk students
        at_risk_df = identify_at_risk_students(df)
        
        if at_risk_df.empty:
            logger.info("No students identified as needing intervention")
            return pd.DataFrame()
        
        # Create intervention priority scoring
        intervention_df = at_risk_df.copy()
        intervention_df['InterventionPriority'] = 'Medium'
        intervention_df['InterventionActions'] = ''
        
        # High priority: High risk score + multiple factors
        high_priority = (
            (intervention_df['RiskScore'] >= 5) |
            (intervention_df['RiskFactors'].str.contains('Low Performance')) &
            (intervention_df['RiskFactors'].str.contains('High Absences'))
        )
        intervention_df.loc[high_priority, 'InterventionPriority'] = 'High'
        
        # Very High priority: Multiple critical factors
        very_high_priority = (
            (intervention_df['RiskScore'] >= 7) |
            (
                (intervention_df['RiskFactors'].str.contains('Low Performance')) &
                (intervention_df['RiskFactors'].str.contains('High Absences')) &
                (intervention_df['RiskFactors'].str.contains('Low Engagement'))
            )
        )
        intervention_df.loc[very_high_priority, 'InterventionPriority'] = 'Very High'
        
        # Generate specific intervention actions
        for idx, row in intervention_df.iterrows():
            actions = []
            
            if 'Low Performance' in row['RiskFactors']:
                actions.append('Academic tutoring')
            
            if 'High Absences' in row['RiskFactors']:
                actions.append('Attendance monitoring')
            
            if 'Low Engagement' in row['RiskFactors']:
                actions.append('Engagement activities')
            
            if 'Poor Parent Satisfaction' in row['RiskFactors']:
                actions.append('Parent consultation')
            
            if 'No Parent Survey' in row['RiskFactors']:
                actions.append('Parent outreach')
            
            intervention_df.at[idx, 'InterventionActions'] = '; '.join(actions)
        
        # Sort by priority and risk score
        priority_order = {'Very High': 4, 'High': 3, 'Medium': 2, 'Low': 1}
        intervention_df['PriorityOrder'] = intervention_df['InterventionPriority'].map(priority_order)
        intervention_df = intervention_df.sort_values(['PriorityOrder', 'RiskScore'], ascending=[False, False])
        
        # Select relevant columns for intervention report
        intervention_columns = [
            'gender', 'Topic', 'Semester', 'Class', 'StudentAbsenceDays',
            'TotalEngagement', 'RiskScore', 'RiskFactors', 
            'InterventionPriority', 'InterventionActions'
        ]
        
        # Filter to existing columns
        available_columns = [col for col in intervention_columns if col in intervention_df.columns]
        result_df = intervention_df[available_columns].reset_index(drop=True)
        
        logger.info(f"Intervention priorities identified for {len(result_df)} students")
        return result_df
        
    except Exception as e:
        logger.error(f"Error identifying intervention priorities: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    # Test the risk analysis module
    try:
        import sys
        import os
        sys.path.append('.')
        
        from src.data.data_loader import load_and_validate_data
        from src.data.data_cleaner import clean_student_data
        
        # Load and clean data
        df = load_and_validate_data()
        df_clean = clean_student_data(df)
        
        print("Data loaded and cleaned successfully!")
        
        # Test risk analysis functions
        at_risk = identify_at_risk_students(df_clean)
        print(f"At-risk students identified: {len(at_risk)}")
        
        attendance_analysis = analyze_attendance_patterns(df_clean)
        print(f"Attendance analysis completed: {len(attendance_analysis)} insights")
        
        low_participation = get_low_participation_students(df_clean)
        print(f"Low participation students: {len(low_participation)}")
        
        risk_report = generate_risk_report(df_clean)
        print(f"Risk report generated with {len(risk_report)} sections")
        
        intervention_priorities = identify_intervention_priorities(df_clean)
        print(f"Intervention priorities identified for {len(intervention_priorities)} students")
        
        # Display summary
        if risk_report and 'summary' in risk_report:
            print(f"\nRisk Summary:")
            print(f"- Total students: {risk_report['summary']['total_students']}")
            if 'at_risk_students' in risk_report:
                print(f"- At-risk students: {risk_report['at_risk_students']['count']} ({risk_report['at_risk_students']['percentage']})")
        
    except Exception as e:
        print(f"Error testing risk analysis module: {e}")
