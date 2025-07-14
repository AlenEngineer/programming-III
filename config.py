"""
Configuration file for the Academic Data Analysis System
This file provides backward compatibility while using the new configuration system.
"""

# Import the new configuration system
from src.config import (
    app_config, data_config, analysis_config, 
    visualization_config, report_config, logging_config
)

# Backward compatibility - expose the old variables
BASE_DIR = app_config.base_dir
DATA_DIR = app_config.data_dir
OUTPUT_DIR = app_config.output_dir
CHARTS_DIR = app_config.charts_dir
REPORTS_DIR = app_config.reports_dir

# Data file configuration
DEFAULT_CSV_FILE = data_config.default_csv_file
DATA_FILE_PATH = data_config.data_file_path

# Academic performance thresholds
MINIMUM_PASSING_GRADE = analysis_config.minimum_passing_grade
EXCELLENT_GRADE = analysis_config.excellent_grade
RISK_ABSENCE_THRESHOLD = analysis_config.risk_absence_threshold
LOW_PARTICIPATION_THRESHOLD = analysis_config.low_participation_threshold

# Performance categories mapping
PERFORMANCE_MAPPING = analysis_config.performance_mapping

# Chart styling configuration
CHART_STYLE = visualization_config.chart_style
FIGURE_SIZE = visualization_config.figure_size
DPI = visualization_config.dpi
COLOR_PALETTE = visualization_config.color_palette

# Report configuration
REPORT_TITLE = report_config.title
REPORT_AUTHOR = report_config.author
INSTITUTION = report_config.institution

# Column mappings for analysis
NUMERIC_COLUMNS = data_config.numeric_columns
CATEGORICAL_COLUMNS = data_config.categorical_columns
