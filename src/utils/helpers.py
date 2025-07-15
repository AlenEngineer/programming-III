"""
Funciones utilitarias para el Sistema de Análisis de Datos Académicos
Funciones auxiliares comunes usadas en diferentes módulos.
"""

import os
import logging
from pathlib import Path
from typing import Union, Optional
import pandas as pd
import numpy as np

def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Formatear un valor decimal como una cadena de porcentaje.
    
    Args:
        value: Valor decimal a convertir a porcentaje
        decimals: Número de decimales a mostrar
        
    Returns:
        Cadena de porcentaje formateada
    """
    return f"{value * 100:.{decimals}f}%"

def safe_divide(numerator: Union[int, float], denominator: Union[int, float]) -> float:
    """
    Dividir dos números de forma segura, devolviendo 0 si el denominador es 0.
    
    Args:
        numerator: El dividendo
        denominator: El divisor
        
    Returns:
        Resultado de la división o 0 si el denominador es 0
    """
    if denominator == 0:
        return 0.0
    return numerator / denominator

def create_output_directory(directory_path: Union[str, Path]) -> bool:
    """
    Crear un directorio si no existe.
    
    Args:
        directory_path: Ruta al directorio a crear
        
    Returns:
        True si el directorio fue creado o ya existe, False de lo contrario
    """
    try:
        Path(directory_path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logging.error(f"Error al crear directorio {directory_path}: {e}")
        return False

def setup_logging(log_level: str = "INFO") -> None:
    """
    Configurar la configuración de logging para la aplicación.
    
    Args:
        log_level: Nivel de logging (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('analysis.log'),
            logging.StreamHandler()
        ]
    )

def log_analysis_step(step_name: str, details: Optional[str] = None) -> None:
    """
    Registrar un paso de análisis con detalles opcionales.
    
    Args:
        step_name: Nombre del paso de análisis
        details: Detalles adicionales sobre el paso
    """
    logger = logging.getLogger(__name__)
    message = f"Paso de Análisis: {step_name}"
    if details:
        message += f" - {details}"
    logger.info(message)

def validate_dataframe_columns(df: pd.DataFrame, required_columns: list) -> bool:
    """
    Validar que un DataFrame contenga todas las columnas requeridas.
    
    Args:
        df: DataFrame a validar
        required_columns: Lista de nombres de columnas que deben estar presentes
        
    Returns:
        True si todas las columnas requeridas están presentes, False de lo contrario
    """
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        logging.error(f"Faltan columnas requeridas: {missing_columns}")
        return False
    return True

def get_numeric_summary(series: pd.Series) -> dict:
    """
    Obtener un resumen numérico integral de una Serie de pandas.
    
    Args:
        series: Serie de Pandas con datos numéricos
        
    Returns:
        Diccionario con resumen estadístico
    """
    if not pd.api.types.is_numeric_dtype(series):
        raise ValueError("La serie debe contener datos numéricos")
    
    return {
        'count': len(series),
        'mean': float(series.mean()),
        'median': float(series.median()),
        'std': float(series.std()),
        'min': float(series.min()),
        'max': float(series.max()),
        'q25': float(series.quantile(0.25)),
        'q75': float(series.quantile(0.75))
    }

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpiar y estandarizar nombres de columnas en un DataFrame.
    
    Args:
        df: DataFrame con nombres de columnas potencialmente desordenados
        
    Returns:
        DataFrame con nombres de columnas limpiados
    """
    df_copy = df.copy()
    df_copy.columns = df_copy.columns.str.strip().str.lower().str.replace(' ', '_')
    return df_copy

def export_to_csv(df: pd.DataFrame, filename: str, output_dir: Union[str, Path]) -> str:
    """
    Exportar un DataFrame a archivo CSV en el directorio especificado.
    
    Args:
        df: DataFrame a exportar
        filename: Nombre del archivo de salida (sin extensión)
        output_dir: Directorio donde guardar el archivo
        
    Returns:
        Ruta completa al archivo exportado
    """
    output_path = Path(output_dir) / f"{filename}.csv"
    create_output_directory(output_dir)
    df.to_csv(output_path, index=False)
    logging.info(f"Datos exportados a: {output_path}")
    return str(output_path)
