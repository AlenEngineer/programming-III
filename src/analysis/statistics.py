"""
Módulo de análisis estadístico usando NumPy para el Sistema de Análisis de Datos Académicos.
Maneja todos los cálculos estadísticos incluyendo promedios, desviaciones estándar y distribuciones.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Union
import sys
import os

# Agregar el directorio padre al path para importar config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import NUMERIC_COLUMNS, MINIMUM_PASSING_GRADE, EXCELLENT_GRADE
from src.utils.helpers import log_analysis_step, safe_divide, format_percentage

logger = logging.getLogger(__name__)

def calculate_overall_average(df: pd.DataFrame, column: str = 'TotalEngagement') -> float:
    """
    Calcular el promedio general para una columna numérica especificada.
    
    Args:
        df: DataFrame conteniendo los datos
        column: Nombre de la columna para calcular el promedio
        
    Returns:
        Valor promedio general
    """
    try:
        log_analysis_step(f"Calculando promedio general para {column}")
        
        if column not in df.columns:
            logger.warning(f"Columna {column} no encontrada en DataFrame")
            return 0.0
        
        # Usar NumPy para el cálculo
        values = df[column].dropna().to_numpy()
        if len(values) == 0:
            logger.warning(f"No se encontraron valores válidos en columna {column}")
            return 0.0
        
        average = np.mean(values)
        logger.info(f"Promedio general para {column}: {average:.2f}")
        return float(average)
        
    except Exception as e:
        logger.error(f"Error calculando promedio general para {column}: {e}")
        return 0.0

def calculate_standard_deviation(df: pd.DataFrame, column: str = 'TotalEngagement') -> float:
    """
    Calcular la desviación estándar para una columna numérica especificada.
    
    Args:
        df: DataFrame conteniendo los datos
        column: Nombre de la columna para calcular la desviación estándar
        
    Returns:
        Valor de desviación estándar
    """
    try:
        log_analysis_step(f"Calculando desviación estándar para {column}")
        
        if column not in df.columns:
            logger.warning(f"Columna {column} no encontrada en DataFrame")
            return 0.0
        
        # Usar NumPy para el cálculo
        values = df[column].dropna().to_numpy()
        if len(values) < 2:
            logger.warning(f"Valores insuficientes para cálculo de desviación estándar en {column}")
            return 0.0
        
        std_dev = np.std(values, ddof=1)  # Desviación estándar muestral
        logger.info(f"Desviación estándar para {column}: {std_dev:.2f}")
        return float(std_dev)
        
    except Exception as e:
        logger.error(f"Error calculando desviación estándar para {column}: {e}")
        return 0.0

def get_subject_averages(df: pd.DataFrame, engagement_column: str = 'TotalEngagement') -> Dict[str, float]:
    """
    Calcular promedio de participación/rendimiento por materia/tema.
    
    Args:
        df: DataFrame conteniendo los datos
        engagement_column: Columna para calcular promedios
        
    Returns:
        Diccionario con materias como claves y promedios como valores
    """
    try:
        log_analysis_step("Calculando promedios por materia")
        
        if 'Topic' not in df.columns or engagement_column not in df.columns:
            logger.warning("Columnas requeridas no encontradas para promedios por materia")
            return {}
        
        subject_averages = {}
        for subject in df['Topic'].unique():
            if pd.isna(subject):
                continue
                
            subject_data = df[df['Topic'] == subject][engagement_column].dropna()
            if len(subject_data) > 0:
                avg = np.mean(subject_data.to_numpy())
                subject_averages[subject] = float(avg)
                logger.info(f"Materia {subject} promedio: {avg:.2f}")
        
        return subject_averages
        
    except Exception as e:
        logger.error(f"Error calculando promedios por materia: {e}")
        return {}

def get_semester_averages(df: pd.DataFrame, engagement_column: str = 'TotalEngagement') -> Dict[str, float]:
    """
    Calcular promedio de participación/rendimiento por semestre.
    
    Args:
        df: DataFrame conteniendo los datos
        engagement_column: Columna para calcular promedios
        
    Returns:
        Diccionario con semestres como claves y promedios como valores
    """
    try:
        log_analysis_step("Calculando promedios por semestre")
        
        if 'Semester' not in df.columns or engagement_column not in df.columns:
            logger.warning("Columnas requeridas no encontradas para promedios por semestre")
            return {}
        
        semester_averages = {}
        for semester in df['Semester'].unique():
            if pd.isna(semester):
                continue
                
            semester_data = df[df['Semester'] == semester][engagement_column].dropna()
            if len(semester_data) > 0:
                avg = np.mean(semester_data.to_numpy())
                semester_averages[semester] = float(avg)
                logger.info(f"Semestre {semester} promedio: {avg:.2f}")
        
        return semester_averages
        
    except Exception as e:
        logger.error(f"Error calculando promedios por semestre: {e}")
        return {}

def get_student_averages(df: pd.DataFrame, engagement_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Calcular promedios individuales de estudiantes a través de métricas de participación.
    
    Args:
        df: DataFrame conteniendo los datos
        engagement_columns: Lista de columnas a incluir en promedios de estudiantes
        
    Returns:
        DataFrame con índices de estudiantes y sus puntuaciones promedio
    """
    try:
        log_analysis_step("Calculando promedios individuales de estudiantes")
        
        if engagement_columns is None:
            engagement_columns = ['RaisedHands', 'VisitedResources', 'AnnouncementsView', 'Discussion']
        
        # Filtrar a columnas existentes
        available_columns = [col for col in engagement_columns if col in df.columns]
        
        if not available_columns:
            logger.warning("No se encontraron columnas de participación para promedios de estudiantes")
            return pd.DataFrame()
        
        # Calcular promedios de estudiantes
        student_data = df[available_columns].copy()
        student_averages = np.mean(student_data.to_numpy(), axis=1)
        
        result_df = pd.DataFrame({
            'StudentIndex': df.index,
            'AvgEngagement': student_averages,
            'Class': df['Class'] if 'Class' in df.columns else 'Unknown'
        })
        
        logger.info(f"Promedios calculados para {len(result_df)} estudiantes")
        return result_df
        
    except Exception as e:
        logger.error(f"Error calculando promedios de estudiantes: {e}")
        return pd.DataFrame()

def calculate_performance_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calcular estadísticas de rendimiento integrales para el dataset.
    
    Args:
        df: DataFrame conteniendo los datos
        
    Returns:
        Diccionario conteniendo varias estadísticas de rendimiento
    """
    try:
        log_analysis_step("Calculando estadísticas de rendimiento integrales")
        
        stats = {}
        
        # Distribución general de clases
        if 'Class' in df.columns:
            class_counts = df['Class'].value_counts().to_dict()
            total_students = len(df)
            class_percentages = {k: format_percentage(v/total_students) for k, v in class_counts.items()}
            
            stats['class_distribution'] = {
                'counts': class_counts,
                'percentages': class_percentages
            }
        
        # Estadísticas de participación
        engagement_cols = ['RaisedHands', 'VisitedResources', 'AnnouncementsView', 'Discussion']
        available_engagement = [col for col in engagement_cols if col in df.columns]
        
        if available_engagement:
            engagement_stats = {}
            for col in available_engagement:
                values = df[col].dropna().to_numpy()
                if len(values) > 0:
                    engagement_stats[col] = {
                        'mean': float(np.mean(values)),
                        'median': float(np.median(values)),
                        'std': float(np.std(values, ddof=1)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'q25': float(np.percentile(values, 25)),
                        'q75': float(np.percentile(values, 75))
                    }
            stats['engagement_statistics'] = engagement_stats
        
        # Análisis basado en género
        if 'gender' in df.columns:
            gender_stats = {}
            for gender in df['gender'].unique():
                if pd.isna(gender):
                    continue
                gender_data = df[df['gender'] == gender]
                
                if 'TotalEngagement' in gender_data.columns:
                    values = gender_data['TotalEngagement'].dropna().to_numpy()
                    if len(values) > 0:
                        gender_stats[gender] = {
                            'count': len(gender_data),
                            'avg_engagement': float(np.mean(values)),
                            'performance_distribution': gender_data['Class'].value_counts().to_dict() if 'Class' in gender_data.columns else {}
                        }
            stats['gender_analysis'] = gender_stats
        
        # Análisis de asistencia
        if 'StudentAbsenceDays' in df.columns:
            attendance_stats = df['StudentAbsenceDays'].value_counts().to_dict()
            total = len(df)
            attendance_percentages = {k: format_percentage(v/total) for k, v in attendance_stats.items()}
            
            stats['attendance_analysis'] = {
                'distribution': attendance_stats,
                'percentages': attendance_percentages
            }
        
        logger.info("Estadísticas de rendimiento integrales calculadas exitosamente")
        return stats
        
    except Exception as e:
        logger.error(f"Error calculando estadísticas de rendimiento: {e}")
        return {}

def calculate_correlation_matrix(df: pd.DataFrame, columns: Optional[List[str]] = None) -> np.ndarray:
    """
    Calcular matriz de correlación para columnas numéricas usando NumPy.
    
    Args:
        df: DataFrame conteniendo los datos
        columns: Lista de columnas a incluir en análisis de correlación
        
    Returns:
        Matriz de correlación como array de NumPy
    """
    try:
        log_analysis_step("Calculando matriz de correlación")
        
        if columns is None:
            columns = ['RaisedHands', 'VisitedResources', 'AnnouncementsView', 'Discussion']
        
        # Filtrar a columnas numéricas existentes
        available_columns = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        
        if len(available_columns) < 2:
            logger.warning("Columnas numéricas insuficientes para análisis de correlación")
            return np.array([])
        
        # Crear matriz de correlación usando NumPy
        data_matrix = df[available_columns].dropna().to_numpy()
        correlation_matrix = np.corrcoef(data_matrix.T)
        
        logger.info(f"Matriz de correlación calculada para {len(available_columns)} variables")
        return correlation_matrix
        
    except Exception as e:
        logger.error(f"Error calculando matriz de correlación: {e}")
        return np.array([])

def calculate_z_scores(df: pd.DataFrame, column: str) -> np.ndarray:
    """
    Calcular z-scores para una columna numérica para identificar valores atípicos.
    
    Args:
        df: DataFrame conteniendo los datos
        column: Nombre de la columna para calcular z-scores
        
    Returns:
        Array de z-scores
    """
    try:
        log_analysis_step(f"Calculando z-scores para {column}")
        
        if column not in df.columns:
            logger.warning(f"Columna {column} no encontrada")
            return np.array([])
        
        values = df[column].dropna().to_numpy()
        if len(values) < 2:
            logger.warning(f"Datos insuficientes para cálculo de z-score en {column}")
            return np.array([])
        
        # Calcular z-scores usando NumPy
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)
        
        if std_val == 0:
            logger.warning(f"Desviación estándar cero en {column}, no se pueden calcular z-scores")
            return np.zeros(len(values))
        
        z_scores = (values - mean_val) / std_val
        
        # Registrar valores atípicos (|z| > 3)
        outliers = np.abs(z_scores) > 3
        outlier_count = np.sum(outliers)
        
        logger.info(f"Z-scores calculados para {column}. Se encontraron {outlier_count} valores atípicos")
        return z_scores
        
    except Exception as e:
        logger.error(f"Error calculando z-scores para {column}: {e}")
        return np.array([])

def calculate_all_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calcular todas las medidas estadísticas para el dataset.
    
    Args:
        df: DataFrame limpio listo para análisis
        
    Returns:
        Diccionario integral de todas las medidas estadísticas
    """
    try:
        log_analysis_step("Calculando todas las medidas estadísticas")
        
        all_stats = {}
        
        # Estadísticas básicas
        if 'TotalEngagement' in df.columns:
            all_stats['overall_average'] = calculate_overall_average(df, 'TotalEngagement')
            all_stats['overall_std'] = calculate_standard_deviation(df, 'TotalEngagement')
        
        # Promedios basados en grupos
        all_stats['subject_averages'] = get_subject_averages(df)
        all_stats['semester_averages'] = get_semester_averages(df)
        
        # Estadísticas de rendimiento
        all_stats['performance_stats'] = calculate_performance_statistics(df)
        
        # Estadísticas a nivel de estudiante
        student_stats = get_student_averages(df)
        if not student_stats.empty:
            all_stats['student_statistics'] = {
                'total_students': len(student_stats),
                'avg_student_engagement': float(np.mean(student_stats['AvgEngagement'].to_numpy())),
                'std_student_engagement': float(np.std(student_stats['AvgEngagement'].to_numpy(), ddof=1))
            }
        
        # Análisis de correlación
        correlation_matrix = calculate_correlation_matrix(df)
        if correlation_matrix.size > 0:
            all_stats['correlation_matrix'] = correlation_matrix.tolist()
        
        logger.info("Todas las medidas estadísticas calculadas exitosamente")
        return all_stats
        
    except Exception as e:
        logger.error(f"Error calculando todas las estadísticas: {e}")
        return {}

if __name__ == "__main__":
    # Probar el módulo de estadísticas
    try:
        from src.data.data_loader import load_and_validate_data
        from src.data.data_cleaner import clean_student_data
        
        # Cargar y limpiar datos
        df = load_and_validate_data()
        df_clean = clean_student_data(df)
        
        print("¡Datos cargados y limpiados exitosamente!")
        
        # Calcular estadísticas
        all_stats = calculate_all_statistics(df_clean)
        print(f"Estadísticas calculadas: {list(all_stats.keys())}")
        
        # Mostrar algunas estadísticas clave
        if 'overall_average' in all_stats:
            print(f"Promedio general de participación: {all_stats['overall_average']:.2f}")
        
        if 'subject_averages' in all_stats:
            print(f"Promedios por materia: {all_stats['subject_averages']}")
        
    except Exception as e:
        print(f"Error probando módulo de estadísticas: {e}")
