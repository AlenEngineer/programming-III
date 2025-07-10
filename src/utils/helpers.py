"""
Funciones utilitarias para el Sistema de Análisis de Datos Académicos
Funciones auxiliares comunes utilizadas en diferentes módulos.
"""

import os
import logging
from pathlib import Path
from typing import Union, Optional
import pandas as pd
import numpy as np

def formatear_porcentaje(valor: float, decimales: int = 2) -> str:
    """
    Formatear un valor decimal como cadena de porcentaje.
    
    Args:
        valor: Valor decimal a convertir a porcentaje
        decimales: Número de lugares decimales a mostrar
        
    Returns:
        Cadena de porcentaje formateada
    """
    return f"{valor * 100:.{decimales}f}%"

def division_segura(numerador: Union[int, float], denominador: Union[int, float]) -> float:
    """
    Dividir dos números de forma segura, retornando 0 si el denominador es 0.
    
    Args:
        numerador: El dividendo
        denominador: El divisor
        
    Returns:
        Resultado de la división o 0 si el denominador es 0
    """
    if denominador == 0:
        return 0.0
    return numerador / denominador

def crear_directorio_salida(ruta_directorio: Union[str, Path]) -> bool:
    """
    Crear un directorio si no existe.
    
    Args:
        ruta_directorio: Ruta al directorio a crear
        
    Returns:
        True si el directorio fue creado o ya existe, False en caso contrario
    """
    try:
        Path(ruta_directorio).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logging.error(f"Falló al crear directorio {ruta_directorio}: {e}")
        return False

def configurar_logging(nivel_log: str = "INFO") -> None:
    """
    Configurar el logging para la aplicación.
    
    Args:
        nivel_log: Nivel de logging (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, nivel_log.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('analisis.log'),
            logging.StreamHandler()
        ]
    )

def log_analysis_step(step_name: str, details: Optional[str] = None) -> None:
    """
    Log an analysis step with optional details.
    
    Args:
        step_name: Name of the analysis step
        details: Additional details about the step
    """
    logger = logging.getLogger(__name__)
    message = f"Analysis Step: {step_name}"
    if details:
        message += f" - {details}"
    logger.info(message)

def validate_dataframe_columns(df: pd.DataFrame, required_columns: list) -> bool:
    """
    Validate that a DataFrame contains all required columns.
    
    Args:
        df: DataFrame to validate
        required_columns: List of column names that must be present
        
    Returns:
        True if all required columns are present, False otherwise
    """
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        logging.error(f"Missing required columns: {missing_columns}")
        return False
    return True

def get_numeric_summary(series: pd.Series) -> dict:
    """
    Get a comprehensive numeric summary of a pandas Series.
    
    Args:
        series: Pandas Series with numeric data
        
    Returns:
        Dictionary with statistical summary
    """
    if not pd.api.types.is_numeric_dtype(series):
        raise ValueError("Series must contain numeric data")
    
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
    Clean and standardize column names in a DataFrame.
    
    Args:
        df: DataFrame with potentially messy column names
        
    Returns:
        DataFrame with cleaned column names
    """
    df_copy = df.copy()
    df_copy.columns = df_copy.columns.str.strip().str.lower().str.replace(' ', '_')
    return df_copy

def export_to_csv(df: pd.DataFrame, filename: str, output_dir: Union[str, Path]) -> str:
    """
    Export a DataFrame to CSV file in the specified directory.
    
    Args:
        df: DataFrame to export
        filename: Name of the output file (without extension)
        output_dir: Directory to save the file
        
    Returns:
        Full path to the exported file
    """
    output_path = Path(output_dir) / f"{filename}.csv"
    create_output_directory(output_dir)
    df.to_csv(output_path, index=False)
    logging.info(f"Data exported to: {output_path}")
    return str(output_path)

# Aliases para compatibilidad con código existente (English function names)
# Compatibility aliases for existing code (English function names)
format_percentage = formatear_porcentaje
safe_divide = division_segura
create_output_directory = crear_directorio_salida
setup_logging = configurar_logging
