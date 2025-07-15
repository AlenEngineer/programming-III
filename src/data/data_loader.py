import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Union, Optional, Dict, Any
import sys
import os

# Agregar el directorio padre al path para importar config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import DATA_FILE_PATH, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS
from src.utils.helpers import validate_dataframe_columns, log_analysis_step, setup_logging

# Configurar logging
setup_logging()
logger = logging.getLogger(__name__)

def load_csv_data(file_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Cargar datos de un archivo CSV con manejo de errores y validación apropiada.
    
    Args:
        file_path: Ruta al archivo CSV. Si es None, usa el valor por defecto de config.
        
    Returns:
        DataFrame cargado
        
    Raises:
        FileNotFoundError: Si el archivo no existe
        ValueError: Si el archivo está vacío o tiene formato inválido
    """
    if file_path is None:
        file_path = DATA_FILE_PATH
    
    file_path = Path(file_path)
    
    try:
        log_analysis_step("Cargando datos CSV", f"Archivo: {file_path}")
        
        if not file_path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
        
        # Cargar el archivo CSV
        df = pd.read_csv(file_path)
        
        if df.empty:
            raise ValueError("El archivo CSV está vacío")
        
        logger.info(f"Datos CSV cargados exitosamente: {len(df)} filas, {len(df.columns)} columnas")
        return df
        
    except Exception as e:
        logger.error(f"Error cargando archivo CSV {file_path}: {e}")
        raise

def load_excel_data(file_path: Union[str, Path], sheet_name: Union[str, int] = 0) -> pd.DataFrame:
    """
    Cargar datos de un archivo Excel con manejo de errores apropiado.
    
    Args:
        file_path: Ruta al archivo Excel
        sheet_name: Nombre o índice de la hoja a cargar
        
    Returns:
        DataFrame cargado
        
    Raises:
        FileNotFoundError: Si el archivo no existe
        ValueError: Si el archivo está vacío o tiene formato inválido
    """
    file_path = Path(file_path)
    
    try:
        log_analysis_step("Cargando datos Excel", f"Archivo: {file_path}, Hoja: {sheet_name}")
        
        if not file_path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
        
        # Cargar el archivo Excel
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        
        if df.empty:
            raise ValueError("El archivo Excel está vacío")
        
        logger.info(f"Datos Excel cargados exitosamente: {len(df)} filas, {len(df.columns)} columnas")
        return df
        
    except Exception as e:
        logger.error(f"Error cargando archivo Excel {file_path}: {e}")
        raise

def validate_data_structure(df: pd.DataFrame) -> bool:
    """
    Validar que los datos cargados tengan la estructura esperada para análisis académico.
    
    Args:
        df: DataFrame a validar
        
    Returns:
        True si la estructura de datos es válida, False de lo contrario
    """
    try:
        log_analysis_step("Validando estructura de datos")
        
        # Verificar columnas requeridas
        all_required_columns = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS
        if not validate_dataframe_columns(df, all_required_columns):
            return False
        
        # Verificar tipos de datos
        for col in NUMERIC_COLUMNS:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                logger.warning(f"La columna {col} no es numérica, se intentará conversión")
        
        # Verificar columnas completamente vacías
        empty_columns = df.columns[df.isnull().all()].tolist()
        if empty_columns:
            logger.warning(f"Se encontraron columnas completamente vacías: {empty_columns}")
        
        # Verificar número mínimo de filas
        if len(df) < 10:
            logger.warning("El dataset tiene muy pocas filas, el análisis puede no ser significativo")
        
        logger.info("Validación de estructura de datos completada exitosamente")
        return True
        
    except Exception as e:
        logger.error(f"Error durante la validación de datos: {e}")
        return False

def get_data_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Obtener información integral sobre el dataset cargado.
    
    Args:
        df: DataFrame a analizar
        
    Returns:
        Diccionario conteniendo información del dataset
    """
    try:
        log_analysis_step("Generando resumen de información de datos")
        
        info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'data_types': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'categorical_columns': [col for col in df.columns if col in CATEGORICAL_COLUMNS],
            'numeric_columns': [col for col in df.columns if col in NUMERIC_COLUMNS],
        }
        
        # Agregar conteos de valores únicos para columnas categóricas
        unique_counts = {}
        for col in info['categorical_columns']:
            if col in df.columns:
                unique_counts[col] = df[col].nunique()
        info['unique_value_counts'] = unique_counts
        
        # Agregar estadísticas básicas para columnas numéricas
        numeric_stats = {}
        for col in info['numeric_columns']:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                numeric_stats[col] = {
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std())
                }
        info['numeric_statistics'] = numeric_stats
        
        logger.info("Resumen de información de datos generado exitosamente")
        return info
        
    except Exception as e:
        logger.error(f"Error generando información de datos: {e}")
        return {}

def load_and_validate_data(file_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Pipeline completo de carga de datos con validación.
    
    Args:
        file_path: Ruta al archivo de datos (CSV o Excel)
        
    Returns:
        DataFrame validado listo para análisis
        
    Raises:
        ValueError: Si la validación de datos falla
    """
    try:
        # Determinar el tipo de archivo y cargar según corresponda
        if file_path is None:
            file_path = DATA_FILE_PATH
        
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.csv':
            df = load_csv_data(file_path)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            df = load_excel_data(file_path)
        else:
            raise ValueError(f"Formato de archivo no soportado: {file_path.suffix}")
        
        # Validar los datos cargados
        if not validate_data_structure(df):
            raise ValueError("Validación de datos falló")
        
        # Obtener y registrar información de datos
        data_info = get_data_info(df)
        logger.info(f"Dataset cargado: {data_info['shape'][0]} filas, {data_info['shape'][1]} columnas")
        
        return df
        
    except Exception as e:
        logger.error(f"Error en el pipeline de carga de datos: {e}")
        raise

if __name__ == "__main__":
    # Probar el cargador de datos
    try:
        df = load_and_validate_data()
        print("¡Datos cargados exitosamente!")
        print(f"Forma: {df.shape}")
        print(f"Columnas: {list(df.columns)}")
        print("\nPrimeras filas:")
        print(df.head())
    except Exception as e:
        print(f"Error: {e}")
