# Academic Data Analysis System

A comprehensive system for analyzing academic performance data with statistical analysis, visualization, and reporting capabilities.

## 🎯 Recent Improvements

This project has been significantly improved to follow good software engineering practices:

### ✅ What Was Fixed

1. **Eliminated Hardcoded Values**: Removed all hardcoded paths and magic numbers
2. **Improved Modularization**: Created proper package structure with separation of concerns
3. **Clean Code Implementation**: Added type hints, proper error handling, and consistent code style
4. **Configuration Management**: Centralized all configuration in type-safe dataclasses
5. **Better Error Handling**: Custom exception hierarchy with meaningful error messages
6. **Logging System**: Centralized logging with file rotation and structured output
7. **Data Validation**: Comprehensive validation system for all inputs

### 🏗️ New Architecture

```
src/
├── config/              # Configuration management
│   ├── __init__.py     # Configuration exports
│   └── settings.py     # Type-safe configuration classes
├── core/               # Core utilities and base classes
│   ├── __init__.py     # Core exports
│   ├── exceptions.py   # Custom exception hierarchy
│   ├── logging_setup.py # Centralized logging
│   └── validators.py   # Data validation utilities
├── data/               # Data processing modules
│   ├── data_loader.py  # Data loading and validation
│   └── data_cleaner.py # Data cleaning and preprocessing
├── analysis/           # Statistical analysis modules
│   ├── statistics.py   # Statistical calculations
│   ├── grouping.py     # Data grouping and aggregation
│   └── risk_analysis.py # Risk assessment
├── visualization/      # Chart and graph generation
│   └── charts.py       # Visualization utilities
├── reports/            # Report generation
│   └── apa_report.py   # APA-style report generation
└── utils/              # Utility functions
    └── helpers.py      # Helper functions
```

## 🚀 Getting Started

### Prerequisites

- Python 3.12+
- Required packages: pandas, numpy, matplotlib, seaborn, reportlab

### Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn reportlab
   ```

### Usage

#### Basic Usage

```python
from src.config import app_config, data_config
from src.core import setup_logging, get_logger
from src.data.data_loader import load_and_validate_data

# Setup logging
setup_logging(level="INFO")
logger = get_logger(__name__)

# Load data using configuration
df = load_and_validate_data(data_config.data_file_path)
```

#### Running the Complete Analysis

```python
from main import run_complete_analysis

# Run complete analysis pipeline
results = run_complete_analysis(
    data_file=None,  # Uses default from config
    generate_charts=True,
    generate_report=True,
    verbose=True
)

if results['success']:
    print("Analysis completed successfully!")
else:
    print("Analysis failed:", results['errors'])
```

#### Command Line Usage

```bash
python3 main.py
```

## 📊 Configuration System

The new configuration system uses type-safe dataclasses:

### Configuration Classes

- **AppConfig**: Application-level settings (paths, metadata)
- **DataConfig**: Data processing configuration (columns, validation)
- **AnalysisConfig**: Analysis parameters (thresholds, statistical settings)
- **VisualizationConfig**: Chart styling and export settings
- **ReportConfig**: Report generation configuration
- **LoggingConfig**: Logging system configuration

### Customizing Configuration

```python
from src.config import data_config, analysis_config

# Modify thresholds
analysis_config.risk_absence_threshold = 10
analysis_config.minimum_passing_grade = 65

# Add custom columns
data_config.numeric_columns.append('new_metric')
```

## 🔧 Error Handling

The system uses custom exceptions for better error reporting:

```python
from src.core import DataLoadError, DataValidationError

try:
    df = load_and_validate_data("data.csv")
except DataLoadError as e:
    logger.error(f"Failed to load data: {e}")
except DataValidationError as e:
    logger.error(f"Data validation failed: {e}")
```

## 📝 Logging

Centralized logging system with file rotation:

```python
from src.core import setup_logging, get_logger

# Setup logging
setup_logging(level="INFO")
logger = get_logger(__name__)

# Log messages
logger.info("Processing started")
logger.warning("Data quality issue detected")
logger.error("Processing failed")
```

## 🧪 Testing

Run the test suite to verify the improvements:

```bash
python3 test_improvements.py
```

## 📁 Output Structure

```
output/
├── charts/         # Generated visualization files
├── reports/        # PDF reports
└── logs/          # Application logs
```

## 🔄 Migration Guide

### From Old System

The old `config.py` still works for backward compatibility:

```python
# Old way (still works)
from config import DATA_FILE_PATH, OUTPUT_DIR

# New way (recommended)
from src.config import data_config, app_config
data_file = data_config.data_file_path
output_dir = app_config.output_dir
```

### Key Changes

1. **Import Changes**: Use `from src.config import app_config` instead of `from config import OUTPUT_DIR`
2. **Error Handling**: Catch specific exceptions instead of generic `Exception`
3. **Logging**: Use `get_logger(__name__)` instead of `logging.getLogger(__name__)`
4. **Configuration**: Access config through dataclass attributes

## 🎨 Code Style

The codebase follows these conventions:

- Type hints for all function parameters and return values
- Docstrings for all public functions and classes
- Consistent error handling with custom exceptions
- Structured logging with context information
- Configuration-driven parameters (no magic numbers)

## 📚 Examples

### Data Loading

```python
from src.data.data_loader import load_and_validate_data
from src.core import DataLoadError

try:
    df = load_and_validate_data("data.csv")
    print(f"Loaded {len(df)} rows")
except DataLoadError as e:
    print(f"Failed to load data: {e}")
```

### Data Cleaning

```python
from src.data.data_cleaner import clean_student_data
from src.core import DataCleaningError

try:
    clean_df = clean_student_data(df)
    print("Data cleaned successfully")
except DataCleaningError as e:
    print(f"Data cleaning failed: {e}")
```

### Custom Configuration

```python
from src.config.settings import AnalysisConfig

# Create custom configuration
custom_config = AnalysisConfig(
    minimum_passing_grade=70,
    excellent_grade=90,
    risk_absence_threshold=5
)
```

## 🤝 Contributing

1. Follow the existing code style and patterns
2. Add type hints to all functions
3. Use the custom exception hierarchy
4. Add comprehensive logging
5. Update configuration as needed
6. Write tests for new functionality

## 📄 License

This project is part of the Programming III course at Universidad Tecnológica de Panamá.

## 👥 Team

Programming III Team - Universidad Tecnológica de Panamá

---

For detailed information about the improvements, see [IMPROVEMENTS.md](IMPROVEMENTS.md).