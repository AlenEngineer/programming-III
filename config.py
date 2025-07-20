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
DEFAULT_CSV_FILE = "1SG131 SEM P3 Analisis Reg Acad Estudiantil(P3 Proy Sem datos acad).csv"
DATA_FILE_PATH = BASE_DIR / DEFAULT_CSV_FILE
CSV_SEPARATOR = ";"  # Separador del nuevo archivo CSV

# Umbrales de rendimiento académico
MINIMUM_PASSING_GRADE = 60
EXCELLENT_GRADE = 85
RISK_ABSENCE_THRESHOLD = 75  # Menos de 75% de asistencia = en riesgo
LOW_PARTICIPATION_THRESHOLD = 70  # Menos de 70% de cumplimiento = baja participación

# Mapeo de categorías de rendimiento (basado en Calificacion_Final)
PERFORMANCE_MAPPING = {
    'L': 'Bajo',      # < 60
    'M': 'Medio',     # 60-85
    'H': 'Alto'       # > 85
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

# Mapeo de columnas para análisis (NUEVO ARCHIVO)
NUMERIC_COLUMNS = [
    'Calificacion_Final', 'Porcentaje_Asistencia', 'Cumplimiento_Actividades'
]

CATEGORICAL_COLUMNS = [
    'ID_Estudiante', 'Carrera', 'Semestre', 'Materia', 'Grupo', 'Docente'
]

# Mapeo de columnas del archivo original al nuevo (para compatibilidad)
COLUMN_MAPPING = {
    # Archivo original -> Nuevo archivo
    'Class': 'Calificacion_Final',
    'Topic': 'Materia',
    'Semester': 'Semestre',
    'StudentAbsenceDays': 'Porcentaje_Asistencia',
    'raisedhands': 'Cumplimiento_Actividades',
    'VisITedResources': 'Cumplimiento_Actividades',
    'AnnouncementsView': 'Cumplimiento_Actividades',
    'Discussion': 'Cumplimiento_Actividades',
    'Relation': 'Docente',
    'SectionID': 'Grupo',
    'StageID': 'Semestre',
    'GradeID': 'Semestre',
    'NationalITy': 'Carrera',
    'gender': 'Carrera',  # No hay equivalente directo
    'ParentAnsweringSurvey': 'Cumplimiento_Actividades',  # No hay equivalente directo
    'ParentschoolSatisfaction': 'Cumplimiento_Actividades'  # No hay equivalente directo
}

# Asegurar que los directorios de salida existen
OUTPUT_DIR.mkdir(exist_ok=True)
CHARTS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)
