"""
Archivo de configuración para el Sistema de Análisis de Datos Académicos
Contiene todas las constantes y parámetros de configuración utilizados en el sistema.
"""

import os
from pathlib import Path

# Rutas de archivos
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
CHARTS_DIR = OUTPUT_DIR / "charts"
REPORTS_DIR = OUTPUT_DIR / "reports"

# Configuración de archivo de datos
DEFAULT_CSV_FILE = "xAPI-Edu-Data.csv"
DATA_FILE_PATH = BASE_DIR / DEFAULT_CSV_FILE

# Umbrales de rendimiento académico
MINIMUM_PASSING_GRADE = 60
EXCELLENT_GRADE = 85
RISK_ABSENCE_THRESHOLD = 7  # Más de 7 días de ausencia = en riesgo
LOW_PARTICIPATION_THRESHOLD = 20  # Menos de 20 interacciones = baja participación

# Mapeo de categorías de rendimiento
PERFORMANCE_MAPPING = {
    'L': 'Bajo',
    'M': 'Medio', 
    'H': 'Alto'
}

# Configuración de estilo de gráficos
CHART_STYLE = 'whitegrid'
FIGURE_SIZE = (12, 8)
DPI = 300
COLOR_PALETTE = 'viridis'

# Configuración de reportes
REPORT_TITLE = "Reporte de Análisis de Rendimiento Académico"
REPORT_AUTHOR = "Equipo de Programming III"
INSTITUTION = "Universidad Tecnológica de Panamá"

# Mapeo de columnas para análisis
NUMERIC_COLUMNS = [
    'raisedhands', 'VisITedResources', 'AnnouncementsView', 'Discussion'
]

CATEGORICAL_COLUMNS = [
    'gender', 'NationalITy', 'StageID', 'GradeID', 'SectionID', 
    'Topic', 'Semester', 'Relation', 'ParentAnsweringSurvey',
    'ParentschoolSatisfaction', 'StudentAbsenceDays', 'Class'
]

# Asegurar que los directorios de salida existen
OUTPUT_DIR.mkdir(exist_ok=True)
CHARTS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)
