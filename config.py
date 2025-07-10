"""
Configuration file for the Academic Data Analysis System
Contains all constants and configuration parameters used across the system.
"""

import os
from pathlib import Path

# File paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
CHARTS_DIR = OUTPUT_DIR / "charts"
REPORTS_DIR = OUTPUT_DIR / "reports"

# Data file configuration
DEFAULT_CSV_FILE = "xAPI-Edu-Data.csv"
DATA_FILE_PATH = BASE_DIR / DEFAULT_CSV_FILE

# Academic performance thresholds
MINIMUM_PASSING_GRADE = 60
EXCELLENT_GRADE = 85
RISK_ABSENCE_THRESHOLD = 7  # More than 7 days absent = at risk
LOW_PARTICIPATION_THRESHOLD = 20  # Below 20 interactions = low participation

# Performance categories mapping
PERFORMANCE_MAPPING = {
    'L': 'Low',
    'M': 'Medium', 
    'H': 'High'
}

# Chart styling configuration
CHART_STYLE = 'whitegrid'
FIGURE_SIZE = (12, 8)
DPI = 300
COLOR_PALETTE = 'viridis'

# Report configuration
REPORT_TITLE = "Academic Performance Analysis Report"
REPORT_AUTHOR = "Programming III Team"
INSTITUTION = "Universidad Tecnológica de Panamá"

# Column mappings for analysis
NUMERIC_COLUMNS = [
    'raisedhands', 'VisITedResources', 'AnnouncementsView', 'Discussion'
]

CATEGORICAL_COLUMNS = [
    'gender', 'NationalITy', 'StageID', 'GradeID', 'SectionID', 
    'Topic', 'Semester', 'Relation', 'ParentAnsweringSurvey',
    'ParentschoolSatisfaction', 'StudentAbsenceDays', 'Class'
]

# Ensure output directories exist
OUTPUT_DIR.mkdir(exist_ok=True)
CHARTS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)
