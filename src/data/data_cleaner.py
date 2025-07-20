"""
Módulo de limpieza de datos para el Sistema de Análisis de Datos Académicos.
Proporciona funciones para limpiar, validar y preparar datos para análisis.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional
import sys
import os

# Agregar el directorio padre al path para importar config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import NUMERIC_COLUMNS, CATEGORICAL_COLUMNS, PERFORMANCE_MAPPING, COLUMN_MAPPING
from src.utils.helpers import log_analysis_step, setup_logging

# Configurar logging
setup_logging()
logger = logging.getLogger(__name__)

def clean_student_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpiar y preparar datos de estudiantes para análisis.
    
    Args:
        df: DataFrame con datos crudos de estudiantes
        
    Returns:
        DataFrame limpio y preparado para análisis
    """
    try:
        log_analysis_step("Iniciando limpieza de datos de estudiantes")
        
        # Aplicar mapeo de columnas si es necesario (para compatibilidad con archivo original)
        df_mapped = apply_column_mapping(df)
        
        # Manejar valores faltantes
        df_clean = handle_missing_values(df_mapped)
        
        # Estandarizar nombres de columnas
        df_clean = standardize_columns(df_clean)
        
        # Convertir tipos de datos
        df_clean = convert_data_types(df_clean)
        
        # Limpiar datos categóricos
        df_clean = clean_categorical_data(df_clean)
        
        # Agregar características derivadas
        df_enhanced = add_derived_features(df_clean)
        
        logger.info(f"Limpieza de datos completada. Forma final: {df_enhanced.shape}")
        return df_enhanced
        
    except Exception as e:
        logger.error(f"Error durante la limpieza de datos: {e}")
        raise

def apply_column_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplicar mapeo de columnas para compatibilidad con el sistema existente.
    
    Args:
        df: DataFrame original
        
    Returns:
        DataFrame con columnas mapeadas
    """
    try:
        log_analysis_step("Aplicando mapeo de columnas")
        
        df_mapped = df.copy()
        
        # Crear columnas virtuales basadas en el mapeo
        if 'Calificacion_Final' in df_mapped.columns:
            # Crear columna 'Class' basada en Calificacion_Final
            df_mapped['Class'] = df_mapped['Calificacion_Final'].apply(
                lambda x: 'H' if x >= 85 else ('M' if x >= 60 else 'L')
            )
        
        if 'Materia' in df_mapped.columns:
            df_mapped['Topic'] = df_mapped['Materia']
        
        if 'Semestre' in df_mapped.columns:
            df_mapped['Semester'] = df_mapped['Semestre']
        
        if 'Porcentaje_Asistencia' in df_mapped.columns:
            # Convertir porcentaje a categoría de ausencia
            df_mapped['StudentAbsenceDays'] = df_mapped['Porcentaje_Asistencia'].apply(
                lambda x: 'Under-7' if x >= 75 else 'Above-7'
            )
        
        if 'Cumplimiento_Actividades' in df_mapped.columns:
            # Crear columnas de participación basadas en cumplimiento
            df_mapped['RaisedHands'] = df_mapped['Cumplimiento_Actividades']
            df_mapped['VisitedResources'] = df_mapped['Cumplimiento_Actividades']
            df_mapped['AnnouncementsView'] = df_mapped['Cumplimiento_Actividades']
            df_mapped['Discussion'] = df_mapped['Cumplimiento_Actividades']
        
        if 'Docente' in df_mapped.columns:
            df_mapped['Relation'] = df_mapped['Docente']
        
        if 'Grupo' in df_mapped.columns:
            df_mapped['SectionID'] = df_mapped['Grupo']
        
        if 'Carrera' in df_mapped.columns:
            df_mapped['Nationality'] = df_mapped['Carrera']
            df_mapped['gender'] = 'Unknown'  # No hay información de género en el nuevo archivo
        
        # Crear columnas faltantes con valores por defecto
        df_mapped['ParentAnsweringSurvey'] = 'Unknown'
        df_mapped['ParentschoolSatisfaction'] = 'Unknown'
        
        logger.info("Mapeo de columnas aplicado exitosamente")
        return df_mapped
        
    except Exception as e:
        logger.error(f"Error aplicando mapeo de columnas: {e}")
        raise

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Manejar valores faltantes en el dataset.
    
    Args:
        df: DataFrame con valores faltantes
        
    Returns:
        DataFrame con valores faltantes manejados
    """
    try:
        log_analysis_step("Manejando valores faltantes")
        
        df_clean = df.copy()
        original_missing = df_clean.isnull().sum().sum()
        
        # Para columnas numéricas, usar mediana
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_clean[col].isnull().any():
                median_val = df_clean[col].median()
                df_clean[col].fillna(median_val, inplace=True)
                logger.info(f"Valores faltantes en {col} reemplazados con mediana: {median_val}")
        
        # Para columnas categóricas, usar moda
        categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if df_clean[col].isnull().any():
                mode_val = df_clean[col].mode().iloc[0] if not df_clean[col].mode().empty else 'Unknown'
                df_clean[col].fillna(mode_val, inplace=True)
                logger.info(f"Valores faltantes en {col} reemplazados con moda: {mode_val}")
        
        final_missing = df_clean.isnull().sum().sum()
        logger.info(f"Valores faltantes manejados: {original_missing} -> {final_missing}")
        
        return df_clean
        
    except Exception as e:
        logger.error(f"Error manejando valores faltantes: {e}")
        raise

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estandarizar nombres de columnas para consistencia.
    
    Args:
        df: DataFrame con nombres de columnas a estandarizar
        
    Returns:
        DataFrame con nombres de columnas estandarizados
    """
    try:
        log_analysis_step("Estandarizando nombres de columnas")
        
        df_clean = df.copy()
        
        # Registrar nombres de columnas originales para referencia
        logger.info(f"Columnas originales: {list(df.columns)}")
        
        # Estandarizar nombres de columnas específicos si es necesario
        column_mapping = {
            'NationalITy': 'Nationality',
            'PlaceofBirth': 'PlaceOfBirth',
            'VisITedResources': 'VisitedResources',
            'raisedhands': 'RaisedHands'
        }
        
        df_clean.rename(columns=column_mapping, inplace=True)
        logger.info(f"Columnas estandarizadas: {list(df_clean.columns)}")
        
        return df_clean
        
    except Exception as e:
        logger.error(f"Error estandarizando columnas: {e}")
        raise

def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convertir columnas a tipos de datos apropiados para análisis.
    
    Args:
        df: DataFrame con tipos de datos potencialmente incorrectos
        
    Returns:
        DataFrame con tipos de datos corregidos
    """
    try:
        log_analysis_step("Convirtiendo tipos de datos")
        
        df_clean = df.copy()
        
        # Convertir columnas numéricas del nuevo archivo
        numeric_columns_to_convert = ['Calificacion_Final', 'Porcentaje_Asistencia', 'Cumplimiento_Actividades']
        for col in numeric_columns_to_convert:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                logger.info(f"Convertido {col} a tipo numérico")
        
        # Convertir columnas categóricas a tipo category para eficiencia de memoria
        categorical_columns_to_convert = ['ID_Estudiante', 'Carrera', 'Semestre', 'Materia', 'Grupo', 'Docente']
        for col in categorical_columns_to_convert:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype('category')
                logger.info(f"Convertido {col} a tipo category")
        
        logger.info("Conversión de tipos de datos completada")
        return df_clean
        
    except Exception as e:
        logger.error(f"Error convirtiendo tipos de datos: {e}")
        raise

def clean_categorical_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpiar y estandarizar valores de datos categóricos.
    
    Args:
        df: DataFrame con columnas categóricas a limpiar
        
    Returns:
        DataFrame con datos categóricos limpiados
    """
    try:
        log_analysis_step("Limpiando datos categóricos")
        
        df_clean = df.copy()
        
        # Estandarizar valores de género
        if 'gender' in df_clean.columns:
            # Convertir a string temporalmente para usar .str.upper()
            df_clean['gender'] = df_clean['gender'].astype(str).str.upper()
            logger.info(f"Valores de género: {df_clean['gender'].unique()}")
        
        # Estandarizar valores de semestre
        if 'Semester' in df_clean.columns:
            # Convertir a string temporalmente para usar .str.upper()
            df_clean['Semester'] = df_clean['Semester'].astype(str).str.upper()
            logger.info(f"Valores de semestre: {df_clean['Semester'].unique()}")
        
        # Limpiar días de ausencia - asegurar formato consistente
        if 'StudentAbsenceDays' in df_clean.columns:
            # Estandarizar categorías de días de ausencia
            absence_mapping = {
                'under-7': 'Under-7',
                'above-7': 'Above-7',
                'Under-7': 'Under-7',
                'Above-7': 'Above-7'
            }
            df_clean['StudentAbsenceDays'] = df_clean['StudentAbsenceDays'].map(absence_mapping).fillna(df_clean['StudentAbsenceDays'])
            logger.info(f"Valores de días de ausencia: {df_clean['StudentAbsenceDays'].unique()}")
        
        # Estandarizar clase de rendimiento
        if 'Class' in df_clean.columns:
            # Convertir a string temporalmente para usar .str.upper()
            df_clean['Class'] = df_clean['Class'].astype(str).str.upper()
            logger.info(f"Clases de rendimiento: {df_clean['Class'].unique()}")
        
        logger.info("Limpieza de datos categóricos completada")
        return df_clean
        
    except Exception as e:
        logger.error(f"Error limpiando datos categóricos: {e}")
        raise

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agregar características derivadas que serán útiles para análisis.
    
    Args:
        df: DataFrame limpiado
        
    Returns:
        DataFrame con características derivadas adicionales
    """
    try:
        log_analysis_step("Agregando características derivadas")
        
        df_enhanced = df.copy()
        
        # Agregar puntuación total de participación basada en cumplimiento
        if 'Cumplimiento_Actividades' in df_enhanced.columns:
            df_enhanced['TotalEngagement'] = df_enhanced['Cumplimiento_Actividades']
            df_enhanced['AvgEngagement'] = df_enhanced['Cumplimiento_Actividades']
            logger.info("Agregadas métricas de participación basadas en cumplimiento")
        
        # Agregar mapeo de categoría de rendimiento
        if 'Class' in df_enhanced.columns:
            df_enhanced['PerformanceCategory'] = df_enhanced['Class'].map(PERFORMANCE_MAPPING)
            logger.info("Agregado mapeo de categoría de rendimiento")
        
        # Agregar indicador de riesgo basado en asistencia
        if 'Porcentaje_Asistencia' in df_enhanced.columns:
            df_enhanced['HighAbsenceRisk'] = (df_enhanced['Porcentaje_Asistencia'] < 75).astype(int)
            logger.info("Agregado indicador de riesgo por baja asistencia")
        
        # Agregar indicador de rendimiento académico
        if 'Calificacion_Final' in df_enhanced.columns:
            df_enhanced['AcademicRisk'] = (df_enhanced['Calificacion_Final'] < 60).astype(int)
            logger.info("Agregado indicador de riesgo académico")
        
        logger.info(f"Características derivadas agregadas. Nueva forma: {df_enhanced.shape}")
        return df_enhanced
        
    except Exception as e:
        logger.error(f"Error agregando características derivadas: {e}")
        raise

def get_data_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generar un reporte integral de calidad de datos.
    
    Args:
        df: DataFrame a analizar
        
    Returns:
        Diccionario conteniendo métricas de calidad de datos
    """
    try:
        log_analysis_step("Generando reporte de calidad de datos")
        
        report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        # Agregar análisis de columnas categóricas
        categorical_analysis = {}
        for col in df.select_dtypes(include=['category', 'object']).columns:
            categorical_analysis[col] = {
                'unique_values': df[col].nunique(),
                'most_common': df[col].mode().iloc[0] if not df[col].mode().empty else None,
                'unique_value_list': df[col].unique().tolist()[:10]  # Primeros 10 valores únicos
            }
        report['categorical_analysis'] = categorical_analysis
        
        # Agregar análisis de columnas numéricas
        numeric_analysis = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            numeric_analysis[col] = {
                'count': len(df[col].dropna()),
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max())
            }
        report['numeric_analysis'] = numeric_analysis
        
        logger.info("Reporte de calidad de datos generado exitosamente")
        return report
        
    except Exception as e:
        logger.error(f"Error generando reporte de calidad de datos: {e}")
        return {}

if __name__ == "__main__":
    # Probar el limpiador de datos con datos de muestra
    try:
        from data_loader import load_and_validate_data
        
        # Cargar datos
        df = load_and_validate_data()
        print("¡Datos originales cargados exitosamente!")
        print(f"Forma original: {df.shape}")
        
        # Limpiar datos
        df_clean = clean_student_data(df)
        print(f"Forma de datos limpiados: {df_clean.shape}")
        print(f"Columnas limpiadas: {list(df_clean.columns)}")
        
        # Generar reporte de calidad
        quality_report = get_data_quality_report(df_clean)
        print(f"Reporte de calidad generado: {len(quality_report)} métricas")
        
    except Exception as e:
        print(f"Error probando limpiador de datos: {e}")
