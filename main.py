"""
Sistema de Análisis de Datos Académicos - Orquestador Principal

Este módulo coordina el pipeline completo de análisis de datos académicos:
1. Carga y validación de datos
2. Limpieza y preprocesamiento de datos
3. Análisis estadístico y agrupación
4. Identificación y evaluación de riesgos
5. Generación de visualizaciones
6. Generación de reportes estilo APA

El sistema proporciona un enfoque modular y amigable para el equipo para analizar
datos de rendimiento estudiantil con capacidades de reporte integral.
"""

import sys
import os
import logging
import traceback
from pathlib import Path
from typing import Optional, Dict, Any

# Agregar la raíz del proyecto al path para imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Importar configuración
from config import (
    DATA_FILE_PATH, OUTPUT_DIR, CHARTS_DIR, REPORTS_DIR,
    REPORT_TITLE, REPORT_AUTHOR, INSTITUTION
)

# Importar módulos de datos
from src.data.data_loader import load_and_validate_data, get_data_info
from src.data.data_cleaner import clean_student_data, get_data_quality_report

# Importar módulos de análisis
from src.analysis.statistics import calculate_all_statistics
from src.analysis.grouping import group_by_demographics, perform_all_groupings
from src.analysis.risk_analysis import generate_risk_report, identify_intervention_priorities

# Importar módulos de visualización
from src.visualization.charts import create_all_visualizations

# Importar módulos de reportes
from src.reports.apa_report import generate_apa_report

# Importar utilidades
from src.utils.helpers import setup_logging, log_analysis_step


def run_complete_analysis(data_file: Optional[str] = None, 
                         generate_charts: bool = True,
                         generate_report: bool = True,
                         verbose: bool = True) -> Dict[str, Any]:
    """
    Ejecuta el pipeline completo de análisis de datos académicos.
    
    Args:
        data_file: Ruta al archivo de datos. Si es None, usa el valor por defecto de config.
        generate_charts: Si generar gráficos de visualización.
        generate_report: Si generar el reporte PDF estilo APA.
        verbose: Si habilitar salida de logging detallada.
    
    Returns:
        Dict conteniendo todos los resultados del análisis y rutas de archivos.
    """
    # Configurar logging
    setup_logging(log_level="INFO" if verbose else "WARNING")
    logger = logging.getLogger(__name__)
    
    results = {
        'success': False,
        'data_info': None,
        'statistics': None,
        'demographic_analysis': None,
        'engagement_analysis': None,
        'risk_report': None,
        'intervention_priorities': None,
        'chart_files': [],
        'report_file': None,
        'errors': []
    }
    
    try:
        # Paso 1: Cargar y validar datos
        log_analysis_step("Cargando y validando datos")
        df = load_and_validate_data(data_file)
        data_info = get_data_info(df)
        results['data_info'] = data_info
        logger.info(f"Cargados {len(df)} registros de estudiantes con {len(df.columns)} características")
        
        # Paso 2: Limpiar y preprocesar datos
        log_analysis_step("Limpiando y preprocesando datos")
        df_clean = clean_student_data(df)
        quality_report = get_data_quality_report(df_clean)
        logger.info(f"Limpieza de datos completada. Puntuación de calidad: {quality_report.get('overall_quality_score', 'N/A')}")
        
        # Paso 3: Análisis estadístico
        log_analysis_step("Realizando análisis estadístico")
        statistics = calculate_all_statistics(df_clean)
        results['statistics'] = statistics
        logger.info("Análisis estadístico completado")
        
        # Paso 4: Análisis demográfico y de participación
        log_analysis_step("Analizando rendimiento por demografía")
        demographic_analysis = group_by_demographics(df_clean)
        results['demographic_analysis'] = demographic_analysis
        
        log_analysis_step("Analizando patrones de participación")
        engagement_analysis = perform_all_groupings(df_clean)
        results['engagement_analysis'] = engagement_analysis
        logger.info("Análisis demográfico y de participación completado")
        
        # Paso 5: Análisis de riesgos
        log_analysis_step("Generando reporte de evaluación de riesgos")
        risk_report = generate_risk_report(df_clean)
        results['risk_report'] = risk_report
        
        log_analysis_step("Identificando prioridades de intervención")
        intervention_priorities = identify_intervention_priorities(df_clean)
        results['intervention_priorities'] = intervention_priorities
        logger.info(f"Análisis de riesgos completado. {len(risk_report['at_risk_students'])} estudiantes en riesgo identificados")
        
        # Paso 6: Generar visualizaciones
        if generate_charts:
            log_analysis_step("Generando gráficos de visualización")
            chart_files = create_all_visualizations(df_clean, statistics, risk_report)
            results['chart_files'] = chart_files
            logger.info(f"Generados {len(chart_files)} gráficos de visualización")
        
        # Paso 7: Generar reporte estilo APA
        if generate_report:
            log_analysis_step("Generando reporte PDF estilo APA")
            report_file = generate_apa_report(
                df_clean, statistics, engagement_analysis, 
                risk_report, chart_files
            )
            results['report_file'] = report_file
            logger.info(f"Reporte estilo APA generado: {report_file}")
        
        results['success'] = True
        logger.info("Pipeline de análisis completo ejecutado exitosamente")
        
    except Exception as e:
        error_msg = f"Error en el pipeline de análisis: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        results['errors'].append(error_msg)
        results['success'] = False
    
    return results


def print_analysis_summary(results: Dict[str, Any]) -> None:
    """Imprime un resumen de los resultados del análisis."""
    print("\n" + "="*80)
    print("ANÁLISIS DE DATOS ACADÉMICOS - RESUMEN DE EJECUCIÓN")
    print("="*80)
    
    if not results['success']:
        print("❌ ¡El análisis falló!")
        for error in results['errors']:
            print(f"   Error: {error}")
        return
    
    print("✅ ¡Análisis completado exitosamente!")
    
    # Información de datos
    if results['data_info']:
        data_info = results['data_info']
        print(f"\n📊 Información del Dataset:")
        print(f"   • Estudiantes: {data_info['shape'][0]:,}")
        print(f"   • Características: {data_info['shape'][1]}")
        print(f"   • Calidad de datos: {data_info.get('quality_score', 'N/A')}")
    
    # Resumen estadístico
    if results['statistics']:
        stats = results['statistics']
        print(f"\n📈 Resumen Estadístico:")
        if 'overall_stats' in stats:
            overall = stats['overall_stats']
            print(f"   • Participación promedio: {overall.get('mean', 0):.2f}")
            print(f"   • Desviación estándar: {overall.get('std', 0):.2f}")
    
    # Resumen de análisis de riesgos
    if results['risk_report']:
        risk = results['risk_report']
        print(f"\n⚠️  Análisis de Riesgos:")
        print(f"   • Estudiantes en riesgo: {len(risk.get('at_risk_students', []))}")
        print(f"   • Estudiantes con alta ausencia: {len(risk.get('high_absence_students', []))}")
        print(f"   • Estudiantes con baja participación: {len(risk.get('low_participation_students', []))}")
    
    # Archivos de salida
    print(f"\n📁 Archivos Generados:")
    if results['chart_files']:
        print(f"   • Gráficos: {len(results['chart_files'])} archivos en {CHARTS_DIR}")
    if results['report_file']:
        print(f"   • Reporte: {results['report_file']}")
    
    print(f"\n📍 Directorio de salida: {OUTPUT_DIR}")
    print("="*80)


def main():
    """
    Punto de entrada principal para el Sistema de Análisis de Datos Académicos.
    
    Ejecuta el pipeline completo de análisis y muestra el resumen de resultados.
    """
    print("🎓 Sistema de Análisis de Datos Académicos")
    print(f"📊 Analizando datos de: {DATA_FILE_PATH}")
    print(f"🏫 Institución: {INSTITUTION}")
    print(f"👥 Autores: {REPORT_AUTHOR}")
    
    # Verificar si el archivo de datos existe
    if not DATA_FILE_PATH.exists():
        print(f"\n❌ Error: Archivo de datos no encontrado en {DATA_FILE_PATH}")
        print("Por favor asegúrese de que el archivo de datos esté en la ubicación correcta.")
        return
    
    # Ejecutar el análisis completo
    try:
        results = run_complete_analysis(
            data_file=None,  # Usar valor por defecto de config
            generate_charts=True,
            generate_report=True,
            verbose=True
        )
        
        # Imprimir resumen
        print_analysis_summary(results)
        
        if results['success']:
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
