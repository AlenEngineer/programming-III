"""
Sistema de Análisis de Datos Académicos - Orquestador Principal

Este módulo coordina el pipeline completo de análisis de datos académicos:
1. Carga y validación de datos
2. Limpieza y preprocesamiento de datos
3. Análisis estadístico y agrupación
4. Identificación y evaluación de riesgos
5. Generación de visualizaciones
6. Generación de reportes estilo APA

El sistema proporciona un enfoque modular y amigable para equipos para analizar
datos de rendimiento estudiantil con capacidades integrales de reportes.
"""

import sys
import os
import logging
import traceback
from pathlib import Path
from typing import Optional, Dict, Any

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Importar configuración
from config import (
    DATA_FILE_PATH, OUTPUT_DIR, CHARTS_DIR, REPORTS_DIR,
    TITULO_REPORTE, AUTOR_REPORTE, INSTITUCION, REPOSITORIO_GITHUB
)

# Import data modules
from src.data.data_loader import load_and_validate_data, get_data_info
from src.data.data_cleaner import clean_student_data, get_data_quality_report

# Import analysis modules
from src.analysis.statistics import calculate_all_statistics
from src.analysis.grouping import group_by_demographics, perform_all_groupings
from src.analysis.risk_analysis import generate_risk_report, identify_intervention_priorities

# Import visualization modules
from src.visualization.charts import crear_todas_visualizaciones

# Import reporting modules
from src.reports.apa_report import generate_apa_report

# Importar utilidades
from src.utils.helpers import configurar_logging, log_analysis_step


def ejecutar_analisis_completo(archivo_datos: Optional[str] = None, 
                               generar_graficos: bool = True,
                               generar_reporte: bool = True,
                               verboso: bool = True) -> Dict[str, Any]:
    """
    Ejecutar el pipeline completo de análisis de datos académicos.
    
    Args:
        archivo_datos: Ruta al archivo de datos. Si es None, usa el por defecto de config.
        generar_graficos: Si se deben generar gráficos de visualización.
        generar_reporte: Si se debe generar el reporte PDF estilo APA.
        verboso: Si se debe habilitar la salida de logging detallada.
    
    Returns:
        Dict que contiene todos los resultados del análisis y rutas de archivos.
    """
    # Configurar logging
    configurar_logging(nivel_log="INFO" if verboso else "WARNING")
    logger = logging.getLogger(__name__)
    
    resultados = {
        'exito': False,
        'info_datos': None,
        'estadisticas': None,
        'analisis_demografico': None,
        'analisis_participacion': None,
        'reporte_riesgo': None,
        'prioridades_intervencion': None,
        'archivos_graficos': [],
        'archivo_reporte': None,
        'errores': []
    }
    
    try:
        # Paso 1: Cargar y validar datos
        log_analysis_step("Cargando y validando datos")
        df = load_and_validate_data(archivo_datos)
        info_datos = get_data_info(df)
        resultados['info_datos'] = info_datos
        logger.info(f"Cargados {len(df)} registros de estudiantes con {len(df.columns)} características")
        
        # Paso 2: Limpiar y preprocesar datos
        log_analysis_step("Limpiando y preprocesando datos")
        df_limpio = clean_student_data(df)
        reporte_calidad = get_data_quality_report(df_limpio)
        logger.info(f"Limpieza de datos completada. Puntuación de calidad: {reporte_calidad.get('overall_quality_score', 'N/A')}")
        
        # Paso 3: Análisis estadístico
        log_analysis_step("Realizando análisis estadístico")
        estadisticas = calculate_all_statistics(df_limpio)
        resultados['estadisticas'] = estadisticas
        logger.info("Análisis estadístico completado")
        
        # Paso 4: Análisis demográfico y de participación
        log_analysis_step("Analizando rendimiento por demografía")
        analisis_demografico = group_by_demographics(df_limpio)
        resultados['analisis_demografico'] = analisis_demografico
        
        log_analysis_step("Analizando patrones de participación")
        analisis_participacion = perform_all_groupings(df_limpio)
        resultados['analisis_participacion'] = analisis_participacion
        logger.info("Análisis demográfico y de participación completado")
        
        # Paso 5: Análisis de riesgo
        log_analysis_step("Generando reporte de evaluación de riesgo")
        reporte_riesgo = generate_risk_report(df_limpio)
        resultados['reporte_riesgo'] = reporte_riesgo
        
        log_analysis_step("Identificando prioridades de intervención")
        prioridades_intervencion = identify_intervention_priorities(df_limpio)
        resultados['prioridades_intervencion'] = prioridades_intervencion
        logger.info(f"Análisis de riesgo completado. {reporte_riesgo['estudiantes_en_riesgo']['cantidad']} estudiantes en riesgo identificados")
        
        # Paso 6: Generar visualizaciones
        archivos_graficos = {}
        if generar_graficos:
            log_analysis_step("Generando gráficos de visualización")
            archivos_graficos = crear_todas_visualizaciones(df_limpio, estadisticas, analisis_participacion, guardar_graficos=True)
            resultados['archivos_graficos'] = archivos_graficos
            logger.info(f"Generados {len(archivos_graficos)} gráficos de visualización")
        
        # Paso 7: Generar reporte estilo APA
        if generar_reporte:
            log_analysis_step("Generando reporte PDF estilo APA")
            archivo_reporte = generate_apa_report(
                df_limpio, estadisticas, analisis_participacion, 
                reporte_riesgo, archivos_graficos
            )
            resultados['archivo_reporte'] = archivo_reporte
            logger.info(f"Reporte estilo APA generado: {archivo_reporte}")
        
        resultados['exito'] = True
        logger.info("Pipeline de análisis completo ejecutado exitosamente")
        
    except Exception as e:
        mensaje_error = f"Error en el pipeline de análisis: {str(e)}"
        logger.error(mensaje_error)
        logger.error(traceback.format_exc())
        resultados['errores'].append(mensaje_error)
        resultados['exito'] = False
    
    return resultados


def imprimir_resumen_analisis(resultados: Dict[str, Any]) -> None:
    """Imprimir un resumen de los resultados del análisis."""
    print("\n" + "="*80)
    print("ANÁLISIS DE DATOS ACADÉMICOS - RESUMEN DE EJECUCIÓN")
    print("="*80)
    
    if not resultados['exito']:
        print("❌ ¡El análisis falló!")
        for error in resultados['errores']:
            print(f"   Error: {error}")
        return
    
    print("✅ ¡Análisis completado exitosamente!")
    
    # Información de datos
    if resultados['info_datos']:
        info_datos = resultados['info_datos']
        print(f"\n📊 Información del Dataset:")
        print(f"   • Estudiantes: {info_datos['shape'][0]:,}")
        print(f"   • Características: {info_datos['shape'][1]}")
        print(f"   • Calidad de datos: {info_datos.get('quality_score', 'N/A')}")
    
    # Resumen estadístico
    if resultados['estadisticas']:
        stats = resultados['estadisticas']
        print(f"\n📈 Resumen Estadístico:")
        if 'overall_stats' in stats:
            general = stats['overall_stats']
            print(f"   • Participación promedio: {general.get('mean', 0):.2f}")
            print(f"   • Desviación estándar: {general.get('std', 0):.2f}")
    
    # Resumen del análisis de riesgo
    if resultados['reporte_riesgo']:
        riesgo = resultados['reporte_riesgo']
        print(f"\n⚠️  Análisis de Riesgo:")
        print(f"   • Estudiantes en riesgo: {riesgo.get('estudiantes_en_riesgo', {}).get('cantidad', 0)}")
        print(f"   • Estudiantes con altas ausencias: {riesgo.get('analisis_asistencia', {}).get('cantidad_ausencias_altas', 0)}")
        print(f"   • Estudiantes con baja participación: {riesgo.get('analisis_participacion', {}).get('cantidad_baja_participacion', 0)}")
    
    # Archivos de salida
    print(f"\n📁 Archivos Generados:")
    if resultados['archivos_graficos']:
        print(f"   • Gráficos: {len(resultados['archivos_graficos'])} archivos en {CHARTS_DIR}")
    if resultados['archivo_reporte']:
        print(f"   • Reporte: {resultados['archivo_reporte']}")
    
    print(f"\n📍 Directorio de salida: {OUTPUT_DIR}")
    print("="*80)


def main():
    """
    Punto de entrada principal para el Sistema de Análisis de Datos Académicos.
    
    Ejecuta el pipeline completo de análisis y muestra el resumen de resultados.
    """
    print("🎓 Sistema de Análisis de Datos Académicos")
    print(f"📊 Analizando datos de: {DATA_FILE_PATH}")
    print(f"🏫 Institución: {INSTITUCION}")
    print(f"👥 Autores: {AUTOR_REPORTE}")
    print(f"🔗 Repositorio: {REPOSITORIO_GITHUB}")
    
    # Verificar si el archivo de datos existe
    if not DATA_FILE_PATH.exists():
        print(f"\n❌ Error: Archivo de datos no encontrado en {DATA_FILE_PATH}")
        print("Por favor asegúrese de que el archivo de datos esté en la ubicación correcta.")
        return
    
    # Ejecutar el análisis completo
    try:
        resultados = ejecutar_analisis_completo(
            archivo_datos=None,  # Usar el por defecto de config
            generar_graficos=True,
            generar_reporte=True,
            verboso=True
        )
        
        # Imprimir resumen
        imprimir_resumen_analisis(resultados)
        
        if resultados['exito']:
            print(f"\n🎉 ¡Análisis completo! Revise el directorio de salida: {OUTPUT_DIR}")
        else:
            print(f"\n💥 El análisis falló. Revise los logs para más detalles.")
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Análisis interrumpido por el usuario.")
    except Exception as e:
        print(f"\n💥 Error inesperado: {str(e)}")
        print("Revise los logs para información detallada del error.")


if __name__ == "__main__":
    main()
