"""
Archivo de configuración para el Sistema de Análisis de Datos Académicos
Contiene todas las constantes y parámetros de configuración utilizados en el sistema.
"""

import os
from pathlib import Path

# Rutas de archivos
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "datos"
OUTPUT_DIR = BASE_DIR / "salida"
CHARTS_DIR = OUTPUT_DIR / "graficos"
REPORTS_DIR = OUTPUT_DIR / "reportes"

# Configuración de archivos de datos
DEFAULT_CSV_FILE = "xAPI-Edu-Data.csv"
DATA_FILE_PATH = BASE_DIR / DEFAULT_CSV_FILE

# Umbrales de rendimiento académico
NOTA_MINIMA_APROBATORIA = 60
NOTA_EXCELENTE = 85
UMBRAL_RIESGO_AUSENCIAS = 7  # Más de 7 días de ausencia = en riesgo
UMBRAL_BAJA_PARTICIPACION = 20  # Menos de 20 interacciones = baja participación

# Mapeo de categorías de rendimiento
MAPEO_RENDIMIENTO = {
    'L': 'Bajo',
    'M': 'Medio', 
    'H': 'Alto'
}

# Configuración de estilo de gráficos
ESTILO_GRAFICO = 'whitegrid'
TAMAÑO_FIGURA = (12, 8)
DPI = 300
PALETA_COLORES = 'viridis'

# Configuración de reportes
TITULO_REPORTE = "Reporte de Análisis de Rendimiento Académico"
AUTOR_REPORTE = "Alen Ramírez, Ariel García, Joaquín Ortis, Jasser Sánchez"
INSTITUCION = "Universidad Tecnológica de Panamá"
REPOSITORIO_GITHUB = "https://github.com/AlenEngineer/programming-III/tree/spanish-version"

# Mapeo de columnas para análisis
COLUMNAS_NUMERICAS = [
    'raisedhands', 'VisITedResources', 'AnnouncementsView', 'Discussion'
]

COLUMNAS_CATEGORICAS = [
    'gender', 'NationalITy', 'StageID', 'GradeID', 'SectionID', 
    'Topic', 'Semester', 'Relation', 'ParentAnsweringSurvey',
    'ParentschoolSatisfaction', 'StudentAbsenceDays', 'Class'
]

# Asegurar que los directorios de salida existan
OUTPUT_DIR.mkdir(exist_ok=True)
CHARTS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

# Aliases para compatibilidad con código existente (English variable names)
# Compatibility aliases for existing code (English variable names)
NUMERIC_COLUMNS = COLUMNAS_NUMERICAS
CATEGORICAL_COLUMNS = COLUMNAS_CATEGORICAS
PERFORMANCE_MAPPING = MAPEO_RENDIMIENTO
MINIMUM_PASSING_GRADE = NOTA_MINIMA_APROBATORIA
EXCELLENT_GRADE = NOTA_EXCELENTE
RISK_ABSENCE_THRESHOLD = UMBRAL_RIESGO_AUSENCIAS
LOW_PARTICIPATION_THRESHOLD = UMBRAL_BAJA_PARTICIPACION
CHART_STYLE = ESTILO_GRAFICO
FIGURE_SIZE = TAMAÑO_FIGURA
COLOR_PALETTE = PALETA_COLORES
REPORT_TITLE = TITULO_REPORTE
REPORT_AUTHOR = AUTOR_REPORTE
INSTITUTION = INSTITUCION
GITHUB_REPOSITORY = REPOSITORIO_GITHUB
