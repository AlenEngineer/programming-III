"""
Configuration module for the Academic Data Analysis System.
"""

from .settings import (
    AppConfig,
    DataConfig,
    AnalysisConfig,
    VisualizationConfig,
    ReportConfig,
    LoggingConfig
)

# Create default configuration instances
app_config = AppConfig()
data_config = DataConfig()
analysis_config = AnalysisConfig()
visualization_config = VisualizationConfig()
report_config = ReportConfig()
logging_config = LoggingConfig()

__all__ = [
    'app_config',
    'data_config',
    'analysis_config',
    'visualization_config',
    'report_config',
    'logging_config',
    'AppConfig',
    'DataConfig',
    'AnalysisConfig',
    'VisualizationConfig',
    'ReportConfig',
    'LoggingConfig'
]