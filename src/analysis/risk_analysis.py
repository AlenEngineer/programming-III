"""
Módulo de análisis de riesgo para el Sistema de Análisis de Datos Académicos.
Identifica estudiantes en riesgo basado en varios indicadores académicos y comportamentales.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
import sys
import os

# Agregar el directorio padre al path para importar config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import MINIMUM_PASSING_GRADE, RISK_ABSENCE_THRESHOLD, LOW_PARTICIPATION_THRESHOLD
from src.utils.helpers import log_analysis_step, format_percentage

logger = logging.getLogger(__name__)

def identificar_estudiantes_en_riesgo(df: pd.DataFrame, min_engagement: Optional[float] = None) -> pd.DataFrame:
    """
    Identifica estudiantes en riesgo basado en múltiples criterios.
    
    Args:
        df: DataFrame conteniendo datos de estudiantes
        min_engagement: Umbral mínimo de participación (usa el valor por defecto del config si es None)
        
    Returns:
        DataFrame conteniendo estudiantes en riesgo con factores de riesgo
    """
    try:
        log_analysis_step("Identificando estudiantes en riesgo")
        
        if min_engagement is None:
            min_engagement = LOW_PARTICIPATION_THRESHOLD
        
        # Inicializar seguimiento de factores de riesgo
        estudiantes_en_riesgo = df.copy()
        estudiantes_en_riesgo['FactoresRiesgo'] = ''
        estudiantes_en_riesgo['PuntajeRiesgo'] = 0
        estudiantes_en_riesgo['EstaEnRiesgo'] = False
        
        # Factor de Riesgo 1: Bajo rendimiento (Class = 'L')
        if 'Class' in df.columns:
            bajo_rendimiento = estudiantes_en_riesgo['Class'] == 'L'
            estudiantes_en_riesgo.loc[bajo_rendimiento, 'FactoresRiesgo'] += 'Bajo Rendimiento; '
            estudiantes_en_riesgo.loc[bajo_rendimiento, 'PuntajeRiesgo'] += 3
            logger.info(f"Encontrados {bajo_rendimiento.sum()} estudiantes con bajo rendimiento")
        
        # Factor de Riesgo 2: Días de ausencia altos
        if 'StudentAbsenceDays' in df.columns:
            ausencias_altas = estudiantes_en_riesgo['StudentAbsenceDays'] == 'Above-7'
            estudiantes_en_riesgo.loc[ausencias_altas, 'FactoresRiesgo'] += 'Ausencias Altas; '
            estudiantes_en_riesgo.loc[ausencias_altas, 'PuntajeRiesgo'] += 2
            logger.info(f"Encontrados {ausencias_altas.sum()} estudiantes con ausencias altas")
        
        # Factor de Riesgo 3: Baja participación total
        if 'TotalEngagement' in df.columns:
            baja_participacion = estudiantes_en_riesgo['TotalEngagement'] < min_engagement
            estudiantes_en_riesgo.loc[baja_participacion, 'FactoresRiesgo'] += 'Baja Participación; '
            estudiantes_en_riesgo.loc[baja_participacion, 'PuntajeRiesgo'] += 2
            logger.info(f"Encontrados {baja_participacion.sum()} estudiantes con baja participación (< {min_engagement})")
        
        # Factor de Riesgo 4: Métricas de participación individual bajas
        columnas_participacion = ['RaisedHands', 'Discussion']
        for col in columnas_participacion:
            if col in df.columns:
                participacion_baja = estudiantes_en_riesgo[col] < (min_engagement / 4)  # Cuarto del umbral total
                estudiantes_en_riesgo.loc[participacion_baja, 'FactoresRiesgo'] += f'Bajo {col}; '
                estudiantes_en_riesgo.loc[participacion_baja, 'PuntajeRiesgo'] += 1
                logger.info(f"Encontrados {participacion_baja.sum()} estudiantes con bajo {col}")
        
        # Factor de Riesgo 5: Pobre satisfacción de los padres
        if 'ParentschoolSatisfaction' in df.columns:
            satisfaccion_pobre = estudiantes_en_riesgo['ParentschoolSatisfaction'] == 'Bad'
            estudiantes_en_riesgo.loc[satisfaccion_pobre, 'FactoresRiesgo'] += 'Pobre Satisfacción Padres; '
            estudiantes_en_riesgo.loc[satisfaccion_pobre, 'PuntajeRiesgo'] += 1
            logger.info(f"Encontrados {satisfaccion_pobre.sum()} estudiantes con pobre satisfacción de padres")
        
        # Factor de Riesgo 6: Sin respuesta de encuesta de padres
        if 'ParentAnsweringSurvey' in df.columns:
            sin_encuesta = estudiantes_en_riesgo['ParentAnsweringSurvey'] == 'No'
            estudiantes_en_riesgo.loc[sin_encuesta, 'FactoresRiesgo'] += 'Sin Encuesta Padres; '
            estudiantes_en_riesgo.loc[sin_encuesta, 'PuntajeRiesgo'] += 1
            logger.info(f"Encontrados {sin_encuesta.sum()} estudiantes sin respuesta de encuesta de padres")
        
        # Determinar estado general de riesgo (puntaje de riesgo >= 3 considerado en riesgo)
        estudiantes_en_riesgo['EstaEnRiesgo'] = estudiantes_en_riesgo['PuntajeRiesgo'] >= 3
        
        # Limpiar cadena de factores de riesgo
        estudiantes_en_riesgo['FactoresRiesgo'] = estudiantes_en_riesgo['FactoresRiesgo'].str.rstrip('; ')
        
        # Filtrar solo estudiantes en riesgo
        solo_en_riesgo = estudiantes_en_riesgo[estudiantes_en_riesgo['EstaEnRiesgo']].copy()
        
        logger.info(f"Identificados {len(solo_en_riesgo)} estudiantes en riesgo de {len(df)} estudiantes totales")
        return solo_en_riesgo.sort_values('PuntajeRiesgo', ascending=False)
        
    except Exception as e:
        logger.error(f"Error identificando estudiantes en riesgo: {e}")
        return pd.DataFrame()

# Alias de compatibilidad para el nombre en inglés
def identify_at_risk_students(df: pd.DataFrame, min_engagement: Optional[float] = None) -> pd.DataFrame:
    """Alias de compatibilidad para identificar_estudiantes_en_riesgo"""
    return identificar_estudiantes_en_riesgo(df, min_engagement)

def analizar_patrones_asistencia(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analiza patrones de asistencia e identifica tendencias preocupantes.
    
    Args:
        df: DataFrame conteniendo datos de estudiantes
        
    Returns:
        DataFrame con resultados del análisis de asistencia
    """
    try:
        log_analysis_step("Analizando patrones de asistencia")
        
        if 'StudentAbsenceDays' not in df.columns:
            logger.warning("Columna StudentAbsenceDays no encontrada para análisis de asistencia")
            return pd.DataFrame()
        
        # Crear análisis de asistencia
        analisis_asistencia = []
        
        # Distribución general de asistencia
        distribucion_asistencia = df['StudentAbsenceDays'].value_counts()
        total_estudiantes = len(df)
        
        for categoria_ausencia, count in distribucion_asistencia.items():
            porcentaje = (count / total_estudiantes) * 100
            analisis_asistencia.append({
                'Categoria': f'Estudiantes con {categoria_ausencia} días de ausencia',
                'Cantidad': count,
                'Porcentaje': f'{porcentaje:.1f}%',
                'Tipo_Analisis': 'Distribución General'
            })
        
        # Asistencia por materia
        if 'Topic' in df.columns:
            for materia in df['Topic'].unique():
                if pd.isna(materia):
                    continue
                    
                datos_materia = df[df['Topic'] == materia]
                cantidad_ausencias_altas = (datos_materia['StudentAbsenceDays'] == 'Above-7').sum()
                total_materia = len(datos_materia)
                
                if total_materia > 0:
                    tasa_ausencias_altas = (cantidad_ausencias_altas / total_materia) * 100
                    analisis_asistencia.append({
                        'Categoria': f'{materia} - Tasa de Ausencias Altas',
                        'Cantidad': cantidad_ausencias_altas,
                        'Porcentaje': f'{tasa_ausencias_altas:.1f}%',
                        'Tipo_Analisis': 'Por Materia'
                    })
        
        # Asistencia por rendimiento
        if 'Class' in df.columns:
            for rendimiento in df['Class'].unique():
                if pd.isna(rendimiento):
                    continue
                    
                datos_rendimiento = df[df['Class'] == rendimiento]
                cantidad_ausencias_altas = (datos_rendimiento['StudentAbsenceDays'] == 'Above-7').sum()
                total_rendimiento = len(datos_rendimiento)
                
                if total_rendimiento > 0:
                    tasa_ausencias_altas = (cantidad_ausencias_altas / total_rendimiento) * 100
                    analisis_asistencia.append({
                        'Categoria': f'Rendimiento {rendimiento} - Tasa de Ausencias Altas',
                        'Cantidad': cantidad_ausencias_altas,
                        'Porcentaje': f'{tasa_ausencias_altas:.1f}%',
                        'Tipo_Analisis': 'Por Rendimiento'
                    })
        
        # Asistencia por género
        if 'gender' in df.columns:
            for genero in df['gender'].unique():
                if pd.isna(genero):
                    continue
                    
                datos_genero = df[df['gender'] == genero]
                cantidad_ausencias_altas = (datos_genero['StudentAbsenceDays'] == 'Above-7').sum()
                total_genero = len(datos_genero)
                
                if total_genero > 0:
                    tasa_ausencias_altas = (cantidad_ausencias_altas / total_genero) * 100
                    analisis_asistencia.append({
                        'Categoria': f'Género {genero} - Tasa de Ausencias Altas',
                        'Cantidad': cantidad_ausencias_altas,
                        'Porcentaje': f'{tasa_ausencias_altas:.1f}%',
                        'Tipo_Analisis': 'Por Género'
                    })
        
        df_resultado = pd.DataFrame(analisis_asistencia)
        logger.info(f"Análisis de asistencia completado con {len(df_resultado)} insights")
        return df_resultado
        
    except Exception as e:
        logger.error(f"Error analizando patrones de asistencia: {e}")
        return pd.DataFrame()

# Alias de compatibilidad
def analyze_attendance_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Alias de compatibilidad para analizar_patrones_asistencia"""
    return analizar_patrones_asistencia(df)

def obtener_estudiantes_baja_participacion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifica estudiantes con consistentemente baja participación en todas las métricas.
    
    Args:
        df: DataFrame conteniendo datos de estudiantes
        
    Returns:
        DataFrame conteniendo estudiantes con baja participación
    """
    try:
        log_analysis_step("Identificando estudiantes con baja participación")
        
        columnas_participacion = ['RaisedHands', 'VisitedResources', 'AnnouncementsView', 'Discussion']
        columnas_disponibles = [col for col in columnas_participacion if col in df.columns]
        
        if len(columnas_disponibles) < 2:
            logger.warning("Columnas de participación insuficientes para el análisis")
            return pd.DataFrame()
        
        # Calcular umbrales de participación (percentil 25 inferior)
        umbrales = {}
        for col in columnas_disponibles:
            umbrales[col] = df[col].quantile(0.25)
        
        # Identificar estudiantes con baja participación
        baja_participacion = df.copy()
        baja_participacion['ContadorBajaParticipacion'] = 0
        baja_participacion['AreasBajaParticipacion'] = ''
        
        for col in columnas_disponibles:
            es_bajo = baja_participacion[col] <= umbrales[col]
            
            # Usar iloc para asignación más segura
            for idx in baja_participacion.index[es_bajo]:
                baja_participacion.at[idx, 'ContadorBajaParticipacion'] += 1
                baja_participacion.at[idx, 'AreasBajaParticipacion'] += f'{col}; '
        
        # Considerar estudiantes con baja participación en 50% o más áreas como preocupante
        umbral_contador = max(1, len(columnas_disponibles) // 2)
        estudiantes_preocupantes = baja_participacion[baja_participacion['ContadorBajaParticipacion'] >= umbral_contador].copy()
        
        # Limpiar cadena de áreas
        estudiantes_preocupantes['AreasBajaParticipacion'] = estudiantes_preocupantes['AreasBajaParticipacion'].str.rstrip('; ')
        
        # Agregar puntaje general de participación
        if columnas_disponibles:
            estudiantes_preocupantes['PuntajeParticipacionGeneral'] = estudiantes_preocupantes[columnas_disponibles].mean(axis=1)
        
        logger.info(f"Encontrados {len(estudiantes_preocupantes)} estudiantes con baja participación")
        return estudiantes_preocupantes.sort_values('ContadorBajaParticipacion', ascending=False)
        
    except Exception as e:
        logger.error(f"Error identificando estudiantes con baja participación: {e}")
        return pd.DataFrame()

# Alias de compatibilidad
def get_low_participation_students(df: pd.DataFrame) -> pd.DataFrame:
    """Alias de compatibilidad para obtener_estudiantes_baja_participacion"""
    return obtener_estudiantes_baja_participacion(df)

def generar_reporte_riesgo(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Genera un reporte completo de evaluación de riesgo.
    
    Args:
        df: DataFrame conteniendo datos de estudiantes
        
    Returns:
        Diccionario conteniendo análisis completo de riesgo
    """
    try:
        log_analysis_step("Generando reporte completo de riesgo")
        
        reporte = {
            'resumen': {},
            'estudiantes_en_riesgo': {},
            'analisis_asistencia': {},
            'analisis_participacion': {},
            'recomendaciones': []
        }
        
        total_estudiantes = len(df)
        reporte['resumen']['total_estudiantes'] = total_estudiantes
        
        # Obtener estudiantes en riesgo
        df_en_riesgo = identificar_estudiantes_en_riesgo(df)
        reporte['estudiantes_en_riesgo']['cantidad'] = len(df_en_riesgo)
        reporte['estudiantes_en_riesgo']['porcentaje'] = format_percentage(len(df_en_riesgo) / total_estudiantes)
        
        if not df_en_riesgo.empty:
            # Distribución de puntajes de riesgo
            distribucion_puntaje_riesgo = df_en_riesgo['PuntajeRiesgo'].value_counts().sort_index()
            reporte['estudiantes_en_riesgo']['distribucion_puntaje_riesgo'] = distribucion_puntaje_riesgo.to_dict()
            
            # Factores de riesgo más comunes
            todos_factores = ' '.join(df_en_riesgo['FactoresRiesgo'].fillna(''))
            palabras_factores = [factor.strip() for factor in todos_factores.split(';') if factor.strip()]
            from collections import Counter
            conteos_factores = Counter(palabras_factores)
            reporte['estudiantes_en_riesgo']['factores_riesgo_comunes'] = dict(conteos_factores.most_common(5))
            
            # Distribución de riesgo por materia
            if 'Topic' in df_en_riesgo.columns:
                riesgo_materia = df_en_riesgo['Topic'].value_counts().head(5).to_dict()
                reporte['estudiantes_en_riesgo']['materias_con_mas_riesgo'] = riesgo_materia
        
        # Análisis de asistencia
        df_asistencia = analizar_patrones_asistencia(df)
        if not df_asistencia.empty:
            ausencias_altas_general = df_asistencia[
                df_asistencia['Categoria'].str.contains('Above-7')
            ].iloc[0] if len(df_asistencia) > 0 else None
            
            if ausencias_altas_general is not None:
                reporte['analisis_asistencia']['tasa_ausencias_altas'] = ausencias_altas_general['Porcentaje']
        
        # Análisis de participación
        df_baja_participacion = obtener_estudiantes_baja_participacion(df)
        reporte['analisis_participacion']['cantidad_baja_participacion'] = len(df_baja_participacion)
        reporte['analisis_participacion']['porcentaje_baja_participacion'] = format_percentage(
            len(df_baja_participacion) / total_estudiantes
        )
        
        # Correlación de rendimiento con factores de riesgo
        if 'Class' in df.columns:
            riesgo_rendimiento = {}
            for rendimiento in df['Class'].unique():
                if pd.isna(rendimiento):
                    continue
                estudiantes_rendimiento = df[df['Class'] == rendimiento]
                en_riesgo_en_rendimiento = identificar_estudiantes_en_riesgo(estudiantes_rendimiento)
                tasa_riesgo = len(en_riesgo_en_rendimiento) / len(estudiantes_rendimiento) if len(estudiantes_rendimiento) > 0 else 0
                riesgo_rendimiento[rendimiento] = format_percentage(tasa_riesgo)
            
            reporte['resumen']['riesgo_por_rendimiento'] = riesgo_rendimiento
        
        # Generar recomendaciones
        recomendaciones = []
        
        if len(df_en_riesgo) > 0:
            recomendaciones.append(f"Se necesita atención inmediata para {len(df_en_riesgo)} estudiantes identificados en riesgo")
        
        if not df_asistencia.empty:
            tasa_ausencias_altas = df_asistencia[df_asistencia['Categoria'].str.contains('Above-7')]
            if not tasa_ausencias_altas.empty:
                tasa = tasa_ausencias_altas.iloc[0]['Porcentaje']
                recomendaciones.append(f"Abordar problemas de asistencia - {tasa} de estudiantes tienen altas tasas de ausencia")
        
        if len(df_baja_participacion) > 0:
            recomendaciones.append(f"Implementar estrategias de participación para {len(df_baja_participacion)} estudiantes con baja participación")
        
        # Recomendaciones específicas por materia
        if 'Topic' in df.columns and len(df_en_riesgo) > 0:
            riesgos_materia = df_en_riesgo['Topic'].value_counts()
            if len(riesgos_materia) > 0:
                materia_mayor_riesgo = riesgos_materia.index[0]
                recomendaciones.append(f"Enfocar esfuerzos de intervención en la materia {materia_mayor_riesgo} con mayor cantidad de estudiantes en riesgo")
        
        recomendaciones.append("Monitorear regularmente los niveles de participación y satisfacción de los padres")
        recomendaciones.append("Implementar sistema de alerta temprana para estudiantes que muestren múltiples factores de riesgo")
        
        reporte['recomendaciones'] = recomendaciones
        
        logger.info("Reporte completo de riesgo generado exitosamente")
        return reporte
        
    except Exception as e:
        logger.error(f"Error generando reporte de riesgo: {e}")
        return {}

# Alias de compatibilidad
def generate_risk_report(df: pd.DataFrame) -> Dict[str, Any]:
    """Alias de compatibilidad para generar_reporte_riesgo"""
    return generar_reporte_riesgo(df)

def identificar_prioridades_intervencion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifica y prioriza estudiantes que necesitan intervención inmediata.
    
    Args:
        df: DataFrame conteniendo datos de estudiantes
        
    Returns:
        DataFrame con lista priorizada de intervención
    """
    try:
        log_analysis_step("Identificando prioridades de intervención")
        
        # Obtener estudiantes en riesgo
        df_en_riesgo = identificar_estudiantes_en_riesgo(df)
        
        if df_en_riesgo.empty:
            logger.info("No se identificaron estudiantes que necesiten intervención")
            return pd.DataFrame()
        
        # Crear puntuación de prioridad de intervención
        df_intervencion = df_en_riesgo.copy()
        df_intervencion['PrioridadIntervencion'] = 'Media'
        df_intervencion['AccionesIntervencion'] = ''
        
        # Alta prioridad: Alto puntaje de riesgo + múltiples factores
        prioridad_alta = (
            (df_intervencion['PuntajeRiesgo'] >= 5) |
            (df_intervencion['FactoresRiesgo'].str.contains('Bajo Rendimiento')) &
            (df_intervencion['FactoresRiesgo'].str.contains('Ausencias Altas'))
        )
        df_intervencion.loc[prioridad_alta, 'PrioridadIntervencion'] = 'Alta'
        
        # Prioridad muy alta: Múltiples factores críticos
        prioridad_muy_alta = (
            (df_intervencion['PuntajeRiesgo'] >= 7) |
            (
                (df_intervencion['FactoresRiesgo'].str.contains('Bajo Rendimiento')) &
                (df_intervencion['FactoresRiesgo'].str.contains('Ausencias Altas')) &
                (df_intervencion['FactoresRiesgo'].str.contains('Baja Participación'))
            )
        )
        df_intervencion.loc[prioridad_muy_alta, 'PrioridadIntervencion'] = 'Muy Alta'
        
        # Generar acciones específicas de intervención
        for idx, fila in df_intervencion.iterrows():
            acciones = []
            
            if 'Bajo Rendimiento' in fila['FactoresRiesgo']:
                acciones.append('Tutoría académica')
            
            if 'Ausencias Altas' in fila['FactoresRiesgo']:
                acciones.append('Monitoreo de asistencia')
            
            if 'Baja Participación' in fila['FactoresRiesgo']:
                acciones.append('Actividades de participación')
            
            if 'Pobre Satisfacción Padres' in fila['FactoresRiesgo']:
                acciones.append('Consulta con padres')
            
            if 'Sin Encuesta Padres' in fila['FactoresRiesgo']:
                acciones.append('Alcance a padres')
            
            df_intervencion.at[idx, 'AccionesIntervencion'] = '; '.join(acciones)
        
        # Ordenar por prioridad y puntaje de riesgo
        orden_prioridad = {'Muy Alta': 4, 'Alta': 3, 'Media': 2, 'Baja': 1}
        df_intervencion['OrdenPrioridad'] = df_intervencion['PrioridadIntervencion'].map(orden_prioridad)
        df_intervencion = df_intervencion.sort_values(['OrdenPrioridad', 'PuntajeRiesgo'], ascending=[False, False])
        
        # Seleccionar columnas relevantes para reporte de intervención
        columnas_intervencion = [
            'gender', 'Topic', 'Semester', 'Class', 'StudentAbsenceDays',
            'TotalEngagement', 'PuntajeRiesgo', 'FactoresRiesgo', 
            'PrioridadIntervencion', 'AccionesIntervencion'
        ]
        
        # Filtrar a columnas existentes
        columnas_disponibles = [col for col in columnas_intervencion if col in df_intervencion.columns]
        df_resultado = df_intervencion[columnas_disponibles].reset_index(drop=True)
        
        logger.info(f"Prioridades de intervención identificadas para {len(df_resultado)} estudiantes")
        return df_resultado
        
    except Exception as e:
        logger.error(f"Error identificando prioridades de intervención: {e}")
        return pd.DataFrame()

# Alias de compatibilidad
def identify_intervention_priorities(df: pd.DataFrame) -> pd.DataFrame:
    """Alias de compatibilidad para identificar_prioridades_intervencion"""
    return identificar_prioridades_intervencion(df)

if __name__ == "__main__":
    # Prueba del módulo de análisis de riesgo
    try:
        import sys
        import os
        sys.path.append('.')
        
        from src.data.data_loader import load_and_validate_data
        from src.data.data_cleaner import clean_student_data
        
        # Cargar y limpiar datos
        df = load_and_validate_data()
        df_limpio = clean_student_data(df)
        
        print("¡Datos cargados y limpiados exitosamente!")
        
        # Probar funciones de análisis de riesgo
        en_riesgo = identificar_estudiantes_en_riesgo(df_limpio)
        print(f"Estudiantes en riesgo identificados: {len(en_riesgo)}")
        
        analisis_asistencia = analizar_patrones_asistencia(df_limpio)
        print(f"Análisis de asistencia completado: {len(analisis_asistencia)} insights")
        
        baja_participacion = obtener_estudiantes_baja_participacion(df_limpio)
        print(f"Estudiantes con baja participación: {len(baja_participacion)}")
        
        reporte_riesgo = generar_reporte_riesgo(df_limpio)
        print(f"Reporte de riesgo generado con {len(reporte_riesgo)} secciones")
        
        prioridades_intervencion = identificar_prioridades_intervencion(df_limpio)
        print(f"Prioridades de intervención identificadas para {len(prioridades_intervencion)} estudiantes")
        
        # Mostrar resumen
        if reporte_riesgo and 'resumen' in reporte_riesgo:
            print(f"\nResumen de Riesgo:")
            print(f"- Total de estudiantes: {reporte_riesgo['resumen']['total_estudiantes']}")
            if 'estudiantes_en_riesgo' in reporte_riesgo:
                print(f"- Estudiantes en riesgo: {reporte_riesgo['estudiantes_en_riesgo']['cantidad']} ({reporte_riesgo['estudiantes_en_riesgo']['porcentaje']})")
        
    except Exception as e:
        print(f"Error probando módulo de análisis de riesgo: {e}")
