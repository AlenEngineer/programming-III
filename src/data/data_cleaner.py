"""
Módulo de limpieza y preprocesamiento de datos para el Sistema de Análisis de Datos Académicos.
Maneja la limpieza de datos, conversión de tipos y tareas de preprocesamiento.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional
import sys
import os

# Agregar el directorio padre al path para importar config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import NUMERIC_COLUMNS, CATEGORICAL_COLUMNS, PERFORMANCE_MAPPING
from src.utils.helpers import log_analysis_step, get_numeric_summary

logger = logging.getLogger(__name__)

def clean_student_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Función principal de limpieza de datos que orquesta todas las operaciones de limpieza.
    
    Args:
        df: DataFrame crudo del cargador de datos
        
    Returns:
        DataFrame limpio listo para análisis
    """
    try:
        log_analysis_step("Iniciando proceso de limpieza de datos")
        
        df_clean = df.copy()
        
        # Limpiar nombres de columnas
        df_clean = standardize_columns(df_clean)
        
        # Manejar valores faltantes
        df_clean = handle_missing_values(df_clean)
        
        # Convertir tipos de datos
        df_clean = convert_data_types(df_clean)
        
        # Limpiar columnas específicas
        df_clean = clean_categorical_data(df_clean)
        
        # Agregar columnas derivadas
        df_clean = add_derived_features(df_clean)
        
        logger.info(f"Limpieza de datos completada. Forma final: {df_clean.shape}")
        return df_clean
        
    except Exception as e:
        logger.error(f"Error durante la limpieza de datos: {e}")
        raise

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Manejar valores faltantes en el dataset usando estrategias apropiadas.
    
    Args:
        df: DataFrame con valores faltantes potenciales
        
    Returns:
        DataFrame con valores faltantes manejados
    """
    try:
        log_analysis_step("Manejando valores faltantes")
        
        df_clean = df.copy()
        
        # Registrar conteos de valores faltantes
        missing_counts = df_clean.isnull().sum()
        missing_cols = missing_counts[missing_counts > 0]
        
        if len(missing_cols) > 0:
            logger.info(f"Se encontraron valores faltantes en columnas: {missing_cols.to_dict()}")
        
        # Manejar columnas numéricas - rellenar con mediana
        for col in NUMERIC_COLUMNS:
            if col in df_clean.columns and df_clean[col].isnull().sum() > 0:
                median_value = df_clean[col].median()
                df_clean[col].fillna(median_value, inplace=True)
                logger.info(f"Rellenados valores faltantes de {col} con mediana: {median_value}")
        
        # Manejar columnas categóricas - rellenar con moda o 'Unknown'
        for col in CATEGORICAL_COLUMNS:
            if col in df_clean.columns and df_clean[col].isnull().sum() > 0:
                if df_clean[col].mode().empty:
                    fill_value = 'Unknown'
                else:
                    fill_value = df_clean[col].mode()[0]
                df_clean[col].fillna(fill_value, inplace=True)
                logger.info(f"Rellenados valores faltantes de {col} con: {fill_value}")
        
        logger.info("Manejo de valores faltantes completado")
        return df_clean
        
    except Exception as e:
        logger.error(f"Error manejando valores faltantes: {e}")
        raise

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estandarizar nombres de columnas y estructura.
    
    Args:
        df: DataFrame con nombres de columnas potencialmente inconsistentes
        
    Returns:
        DataFrame con columnas estandarizadas
    """
    try:
        log_analysis_step("Estandarizando nombres de columnas")
        
        df_clean = df.copy()
        
        # Remover espacios en blanco al inicio/final de nombres de columnas
        df_clean.columns = df_clean.columns.str.strip()
        
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
        
        # Convertir columnas numéricas
        numeric_columns_to_convert = ['RaisedHands', 'VisitedResources', 'AnnouncementsView', 'Discussion']
        for col in numeric_columns_to_convert:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                logger.info(f"Convertido {col} a tipo numérico")
        
        # Convertir columnas categóricas a tipo category para eficiencia de memoria
        categorical_columns_to_convert = ['gender', 'Nationality', 'StageID', 'GradeID', 'SectionID', 
                                        'Topic', 'Semester', 'Relation', 'Class']
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
            df_clean['gender'] = df_clean['gender'].str.upper()
            logger.info(f"Valores de género: {df_clean['gender'].unique()}")
        
        # Estandarizar valores de semestre
        if 'Semester' in df_clean.columns:
            df_clean['Semester'] = df_clean['Semester'].str.upper()
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
            df_clean['Class'] = df_clean['Class'].str.upper()
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
        
        # Agregar puntuación total de participación
        engagement_cols = ['RaisedHands', 'VisitedResources', 'AnnouncementsView', 'Discussion']
        available_engagement_cols = [col for col in engagement_cols if col in df_enhanced.columns]
        
        if available_engagement_cols:
            df_enhanced['TotalEngagement'] = df_enhanced[available_engagement_cols].sum(axis=1)
            df_enhanced['AvgEngagement'] = df_enhanced[available_engagement_cols].mean(axis=1)
            logger.info("Agregadas métricas de participación")
        
        # Agregar mapeo de categoría de rendimiento
        if 'Class' in df_enhanced.columns:
            df_enhanced['PerformanceCategory'] = df_enhanced['Class'].map(PERFORMANCE_MAPPING)
            logger.info("Agregado mapeo de categoría de rendimiento")
        
        # Agregar indicador de riesgo basado en días de ausencia
        if 'StudentAbsenceDays' in df_enhanced.columns:
            df_enhanced['HighAbsenceRisk'] = (df_enhanced['StudentAbsenceDays'] == 'Above-7').astype(int)
            logger.info("Agregado indicador de riesgo por alta ausencia")
        
        # Agregar indicador de satisfacción de padres
        if 'ParentschoolSatisfaction' in df_enhanced.columns:
            df_enhanced['ParentSatisfied'] = (df_enhanced['ParentschoolSatisfaction'] == 'Good').astype(int)
            logger.info("Agregado indicador de satisfacción de padres")
        
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
            numeric_analysis[col] = get_numeric_summary(df[col])
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
