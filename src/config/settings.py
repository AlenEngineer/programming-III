"""
Configuration settings for the Academic Data Analysis System.
Uses dataclasses for type safety and better organization.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any


@dataclass
class AppConfig:
    """Application-level configuration."""
    
    # Project information
    name: str = "Academic Data Analysis System"
    version: str = "0.1.0"
    author: str = "Programming III Team"
    institution: str = "Universidad Tecnológica de Panamá"
    
    # Directory paths
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent / "data")
    output_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent / "output")
    
    def __post_init__(self):
        """Ensure directories exist after initialization."""
        self.charts_dir = self.output_dir / "charts"
        self.reports_dir = self.output_dir / "reports"
        self.logs_dir = self.output_dir / "logs"
        
        # Create directories if they don't exist
        for directory in [self.data_dir, self.output_dir, self.charts_dir, self.reports_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)


@dataclass
class DataConfig:
    """Data-related configuration."""
    
    # Default data file
    default_csv_file: str = "xAPI-Edu-Data.csv"
    
    # Column definitions
    numeric_columns: List[str] = field(default_factory=lambda: [
        'raisedhands', 'VisITedResources', 'AnnouncementsView', 'Discussion'
    ])
    
    categorical_columns: List[str] = field(default_factory=lambda: [
        'gender', 'NationalITy', 'StageID', 'GradeID', 'SectionID', 
        'Topic', 'Semester', 'Relation', 'ParentAnsweringSurvey',
        'ParentschoolSatisfaction', 'StudentAbsenceDays', 'Class'
    ])
    
    # Data validation settings
    min_rows: int = 10
    max_missing_percentage: float = 0.5
    
    # Data cleaning settings
    outlier_method: str = "iqr"  # Options: "iqr", "zscore", "none"
    outlier_threshold: float = 1.5
    
    @property
    def data_file_path(self) -> Path:
        """Get the full path to the data file."""
        app_config = AppConfig()
        return app_config.base_dir / self.default_csv_file


@dataclass
class AnalysisConfig:
    """Analysis-related configuration."""
    
    # Performance thresholds
    minimum_passing_grade: int = 60
    excellent_grade: int = 85
    
    # Risk analysis thresholds
    risk_absence_threshold: int = 7
    low_participation_threshold: int = 20
    
    # Performance mapping
    performance_mapping: Dict[str, str] = field(default_factory=lambda: {
        'L': 'Low',
        'M': 'Medium', 
        'H': 'High'
    })
    
    # Statistical analysis settings
    confidence_level: float = 0.95
    alpha: float = 0.05
    
    # Correlation settings
    correlation_threshold: float = 0.5
    correlation_method: str = "pearson"  # Options: "pearson", "spearman", "kendall"


@dataclass
class VisualizationConfig:
    """Visualization-related configuration."""
    
    # Chart styling
    chart_style: str = 'whitegrid'
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 300
    color_palette: str = 'viridis'
    
    # Font settings
    font_size: int = 12
    title_font_size: int = 14
    axis_font_size: int = 12
    tick_font_size: int = 10
    legend_font_size: int = 11
    
    # Grid settings
    grid_alpha: float = 0.3
    
    # Color settings
    colors: Dict[str, str] = field(default_factory=lambda: {
        'primary': '#1f77b4',
        'secondary': '#ff7f0e',
        'success': '#2ca02c',
        'warning': '#d62728',
        'info': '#9467bd',
        'light': '#8c564b',
        'dark': '#e377c2'
    })
    
    # Chart export settings
    export_formats: List[str] = field(default_factory=lambda: ['png', 'pdf'])
    bbox_inches: str = 'tight'


@dataclass
class ReportConfig:
    """Report generation configuration."""
    
    # Report metadata
    title: str = "Academic Performance Analysis Report"
    author: str = "Programming III Team"
    institution: str = "Universidad Tecnológica de Panamá"
    
    # Page settings
    page_size: str = "letter"
    margin_top: float = 1.0
    margin_bottom: float = 1.0
    margin_left: float = 1.0
    margin_right: float = 1.0
    
    # Font settings
    font_family: str = "Helvetica"
    font_size: int = 12
    title_font_size: int = 16
    heading_font_size: int = 14
    
    # Report sections
    include_abstract: bool = True
    include_introduction: bool = True
    include_methods: bool = True
    include_results: bool = True
    include_discussion: bool = True
    include_conclusion: bool = True
    include_references: bool = True
    
    # Export settings
    export_format: str = "pdf"
    include_charts: bool = True
    chart_quality: int = 300


@dataclass
class LoggingConfig:
    """Logging configuration."""
    
    # Log levels
    default_level: str = "INFO"
    console_level: str = "INFO"
    file_level: str = "DEBUG"
    
    # Log format
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    
    # File settings
    log_file: str = "academic_analysis.log"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    
    # Logger names
    loggers: List[str] = field(default_factory=lambda: [
        'academic_analysis',
        'data_loader',
        'data_cleaner',
        'statistics',
        'visualization',
        'reports'
    ])


# Environment-specific configuration
def get_config_from_env() -> Dict[str, Any]:
    """Get configuration values from environment variables."""
    return {
        'debug': os.getenv('DEBUG', 'false').lower() == 'true',
        'log_level': os.getenv('LOG_LEVEL', 'INFO'),
        'data_file': os.getenv('DATA_FILE', None),
        'output_dir': os.getenv('OUTPUT_DIR', None),
    }


# Configuration validation
def validate_config(config: Any) -> bool:
    """Validate configuration settings."""
    if isinstance(config, AppConfig):
        return config.base_dir.exists()
    elif isinstance(config, DataConfig):
        return len(config.numeric_columns) > 0 and len(config.categorical_columns) > 0
    elif isinstance(config, AnalysisConfig):
        return config.minimum_passing_grade < config.excellent_grade
    elif isinstance(config, VisualizationConfig):
        return config.dpi > 0 and config.figure_size[0] > 0 and config.figure_size[1] > 0
    elif isinstance(config, ReportConfig):
        return len(config.title) > 0 and len(config.author) > 0
    elif isinstance(config, LoggingConfig):
        return config.default_level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    return True