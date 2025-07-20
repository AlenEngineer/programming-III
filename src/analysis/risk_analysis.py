"""
Módulo de análisis de riesgos para el Sistema de Análisis de Datos Académicos.
Identifica estudiantes en riesgo basándose en varios indicadores académicos y conductuales.
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

def identify_at_risk_students(df: pd.DataFrame, min_engagement: Optional[float] = None) -> pd.DataFrame:
    """
    Identificar estudiantes en riesgo basándose en múltiples criterios.
    
    Args:
        df: DataFrame conteniendo datos de estudiantes
        min_engagement: Umbral mínimo de participación (usa valor por defecto de config si es None)
        
    Returns:
        DataFrame conteniendo estudiantes en riesgo con factores de riesgo
    """
    try:
        log_analysis_step("Identificando estudiantes en riesgo")
        
        if min_engagement is None:
            min_engagement = LOW_PARTICIPATION_THRESHOLD
        
        # Inicializar seguimiento de factores de riesgo
        at_risk_students = df.copy()
        at_risk_students['RiskFactors'] = ''
        at_risk_students['RiskScore'] = 0
        at_risk_students['IsAtRisk'] = False
        
        # Factor de Riesgo 1: Bajo rendimiento (Class = 'L')
        if 'Class' in df.columns:
            low_performance = at_risk_students['Class'] == 'L'
            at_risk_students.loc[low_performance, 'RiskFactors'] += 'Bajo Rendimiento; '
            at_risk_students.loc[low_performance, 'RiskScore'] += 3
            logger.info(f"Se encontraron {low_performance.sum()} estudiantes con bajo rendimiento")
        
        # Factor de Riesgo 2: Altos días de ausencia
        if 'StudentAbsenceDays' in df.columns:
            high_absence = at_risk_students['StudentAbsenceDays'] == 'Above-7'
            at_risk_students.loc[high_absence, 'RiskFactors'] += 'Altas Ausencias; '
            at_risk_students.loc[high_absence, 'RiskScore'] += 2
            logger.info(f"Se encontraron {high_absence.sum()} estudiantes con altas ausencias")
        
        # Factor de Riesgo 3: Baja participación total
        if 'TotalEngagement' in df.columns:
            low_engagement = at_risk_students['TotalEngagement'] < min_engagement
            at_risk_students.loc[low_engagement, 'RiskFactors'] += 'Baja Participación; '
            at_risk_students.loc[low_engagement, 'RiskScore'] += 2
            logger.info(f"Se encontraron {low_engagement.sum()} estudiantes con baja participación (< {min_engagement})")
        
        # Factor de Riesgo 4: Bajas métricas de participación individual
        participation_cols = ['RaisedHands', 'Discussion']
        for col in participation_cols:
            if col in df.columns:
                low_participation = at_risk_students[col] < (min_engagement / 4)  # Un cuarto del umbral total
                at_risk_students.loc[low_participation, 'RiskFactors'] += f'Baja {col}; '
                at_risk_students.loc[low_participation, 'RiskScore'] += 1
                logger.info(f"Se encontraron {low_participation.sum()} estudiantes con baja {col}")
        
        # Factor de Riesgo 5: Mala satisfacción de padres
        if 'ParentschoolSatisfaction' in df.columns:
            poor_satisfaction = at_risk_students['ParentschoolSatisfaction'] == 'Bad'
            at_risk_students.loc[poor_satisfaction, 'RiskFactors'] += 'Mala Satisfacción de Padres; '
            at_risk_students.loc[poor_satisfaction, 'RiskScore'] += 1
            logger.info(f"Se encontraron {poor_satisfaction.sum()} estudiantes con mala satisfacción de padres")
        
        # Factor de Riesgo 6: Sin respuesta de encuesta de padres
        if 'ParentAnsweringSurvey' in df.columns:
            no_survey = at_risk_students['ParentAnsweringSurvey'] == 'No'
            at_risk_students.loc[no_survey, 'RiskFactors'] += 'Sin Encuesta de Padres; '
            at_risk_students.loc[no_survey, 'RiskScore'] += 1
            logger.info(f"Se encontraron {no_survey.sum()} estudiantes sin respuesta de encuesta de padres")
        
        # Determinar estado de riesgo general (puntuación de riesgo >= 3 considerado en riesgo)
        at_risk_students['IsAtRisk'] = at_risk_students['RiskScore'] >= 3
        
        # Limpiar cadena de factores de riesgo
        at_risk_students['RiskFactors'] = at_risk_students['RiskFactors'].str.rstrip('; ')
        
        # Filtrar solo estudiantes en riesgo
        at_risk_only = at_risk_students[at_risk_students['IsAtRisk']].copy()
        
        logger.info(f"Se identificaron {len(at_risk_only)} estudiantes en riesgo de {len(df)} estudiantes totales")
        return at_risk_only.sort_values('RiskScore', ascending=False)
        
    except Exception as e:
        logger.error(f"Error identificando estudiantes en riesgo: {e}")
        return pd.DataFrame()

def analyze_attendance_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analizar patrones de asistencia e identificar tendencias preocupantes.
    
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
        attendance_analysis = []
        
        # Distribución general de asistencia
        attendance_dist = df['StudentAbsenceDays'].value_counts()
        total_students = len(df)
        
        for absence_category, count in attendance_dist.items():
            percentage = (count / total_students) * 100
            attendance_analysis.append({
                'Category': f'Estudiantes con {absence_category} días de ausencia',
                'Count': count,
                'Percentage': f'{percentage:.1f}%',
                'Analysis_Type': 'Distribución General'
            })
        
        # Asistencia por materia
        if 'Topic' in df.columns:
            for subject in df['Topic'].unique():
                if pd.isna(subject):
                    continue
                    
                subject_data = df[df['Topic'] == subject]
                high_absence_count = (subject_data['StudentAbsenceDays'] == 'Above-7').sum()
                subject_total = len(subject_data)
                
                if subject_total > 0:
                    high_absence_rate = (high_absence_count / subject_total) * 100
                    attendance_analysis.append({
                        'Category': f'{subject} - Tasa Alta de Ausencia',
                        'Count': high_absence_count,
                        'Percentage': f'{high_absence_rate:.1f}%',
                        'Analysis_Type': 'Por Materia'
                    })
        
        # Asistencia por rendimiento
        if 'Class' in df.columns:
            for performance in df['Class'].unique():
                if pd.isna(performance):
                    continue
                    
                perf_data = df[df['Class'] == performance]
                high_absence_count = (perf_data['StudentAbsenceDays'] == 'Above-7').sum()
                perf_total = len(perf_data)
                
                if perf_total > 0:
                    high_absence_rate = (high_absence_count / perf_total) * 100
                    attendance_analysis.append({
                        'Category': f'Rendimiento {performance} - Tasa Alta de Ausencia',
                        'Count': high_absence_count,
                        'Percentage': f'{high_absence_rate:.1f}%',
                        'Analysis_Type': 'Por Rendimiento'
                    })
        
        # Asistencia por género
        if 'gender' in df.columns:
            for gender in df['gender'].unique():
                if pd.isna(gender):
                    continue
                    
                gender_data = df[df['gender'] == gender]
                high_absence_count = (gender_data['StudentAbsenceDays'] == 'Above-7').sum()
                gender_total = len(gender_data)
                
                if gender_total > 0:
                    high_absence_rate = (high_absence_count / gender_total) * 100
                    attendance_analysis.append({
                        'Category': f'Género {gender} - Tasa Alta de Ausencia',
                        'Count': high_absence_count,
                        'Percentage': f'{high_absence_rate:.1f}%',
                        'Analysis_Type': 'Por Género'
                    })
        
        result_df = pd.DataFrame(attendance_analysis)
        logger.info(f"Análisis de asistencia completado con {len(result_df)} insights")
        return result_df
        
    except Exception as e:
        logger.error(f"Error analizando patrones de asistencia: {e}")
        return pd.DataFrame()

def get_low_participation_students(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identificar estudiantes con baja participación en actividades académicas.
    
    Args:
        df: DataFrame conteniendo datos de estudiantes
        
    Returns:
        DataFrame con estudiantes de baja participación
    """
    try:
        log_analysis_step("Identificando estudiantes con baja participación")
        
        # Definir umbrales de baja participación
        low_engagement_threshold = LOW_PARTICIPATION_THRESHOLD
        
        # Crear DataFrame de resultados
        low_participation = df.copy()
        low_participation['LowParticipationFactors'] = ''
        low_participation['ParticipationScore'] = 0
        low_participation['IsLowParticipation'] = False
        
        # Factor 1: Baja participación total
        if 'TotalEngagement' in df.columns:
            low_total = low_participation['TotalEngagement'] < low_engagement_threshold
            low_participation.loc[low_total, 'LowParticipationFactors'] += 'Baja Participación Total; '
            low_participation.loc[low_total, 'ParticipationScore'] += 3
            logger.info(f"Se encontraron {low_total.sum()} estudiantes con baja participación total")
        
        # Factor 2: Pocos levantamientos de mano
        if 'RaisedHands' in df.columns:
            low_hands = low_participation['RaisedHands'] < (low_engagement_threshold / 4)
            low_participation.loc[low_hands, 'LowParticipationFactors'] += 'Pocos Levantamientos de Mano; '
            low_participation.loc[low_hands, 'ParticipationScore'] += 2
            logger.info(f"Se encontraron {low_hands.sum()} estudiantes con pocos levantamientos de mano")
        
        # Factor 3: Baja participación en discusiones
        if 'Discussion' in df.columns:
            low_discussion = low_participation['Discussion'] < (low_engagement_threshold / 4)
            low_participation.loc[low_discussion, 'LowParticipationFactors'] += 'Baja Participación en Discusiones; '
            low_participation.loc[low_discussion, 'ParticipationScore'] += 2
            logger.info(f"Se encontraron {low_discussion.sum()} estudiantes con baja participación en discusiones")
        
        # Factor 4: Pocas visitas a recursos
        if 'VisitedResources' in df.columns:
            low_resources = low_participation['VisitedResources'] < (low_engagement_threshold / 4)
            low_participation.loc[low_resources, 'LowParticipationFactors'] += 'Pocas Visitas a Recursos; '
            low_participation.loc[low_resources, 'ParticipationScore'] += 1
            logger.info(f"Se encontraron {low_resources.sum()} estudiantes con pocas visitas a recursos")
        
        # Factor 5: Pocas vistas de anuncios
        if 'AnnouncementsView' in df.columns:
            low_announcements = low_participation['AnnouncementsView'] < (low_engagement_threshold / 4)
            low_participation.loc[low_announcements, 'LowParticipationFactors'] += 'Pocas Vistas de Anuncios; '
            low_participation.loc[low_announcements, 'ParticipationScore'] += 1
            logger.info(f"Se encontraron {low_announcements.sum()} estudiantes con pocas vistas de anuncios")
        
        # Determinar estado de baja participación (puntuación >= 2 considerado baja participación)
        low_participation['IsLowParticipation'] = low_participation['ParticipationScore'] >= 2
        
        # Limpiar cadena de factores
        low_participation['LowParticipationFactors'] = low_participation['LowParticipationFactors'].str.rstrip('; ')
        
        # Filtrar solo estudiantes con baja participación
        low_participation_only = low_participation[low_participation['IsLowParticipation']].copy()
        
        logger.info(f"Se identificaron {len(low_participation_only)} estudiantes con baja participación de {len(df)} estudiantes totales")
        return low_participation_only.sort_values('ParticipationScore', ascending=False)
        
    except Exception as e:
        logger.error(f"Error identificando estudiantes con baja participación: {e}")
        return pd.DataFrame()

def generate_risk_report(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generar reporte completo de análisis de riesgos.
    
    Args:
        df: DataFrame conteniendo datos de estudiantes
        
    Returns:
        Diccionario con resultados completos del análisis de riesgos
    """
    try:
        log_analysis_step("Generando reporte de análisis de riesgos")
        
        risk_report = {
            'at_risk_students': {},
            'attendance_analysis': {},
            'low_participation_students': {},
            'risk_factors_summary': {},
            'intervention_recommendations': []
        }
        
        # 1. Identificar estudiantes en riesgo
        at_risk_df = identify_at_risk_students(df)
        if not at_risk_df.empty:
            risk_report['at_risk_students'] = {
                'count': len(at_risk_df),
                'percentage': format_percentage(len(at_risk_df) / len(df) * 100),
                'data': at_risk_df,
                'risk_score_distribution': at_risk_df['RiskScore'].value_counts().to_dict()
            }
            
            # Análisis de factores de riesgo comunes
            all_risk_factors = []
            for factors in at_risk_df['RiskFactors']:
                if pd.notna(factors) and factors.strip():
                    all_risk_factors.extend([f.strip() for f in factors.split(';')])
            
            factor_counts = {}
            for factor in all_risk_factors:
                factor_counts[factor] = factor_counts.get(factor, 0) + 1
            
            risk_report['risk_factors_summary'] = {
                'common_factors': dict(sorted(factor_counts.items(), key=lambda x: x[1], reverse=True)[:5]),
                'total_factors_identified': len(all_risk_factors)
            }
        
        # 2. Análisis de patrones de asistencia
        attendance_df = analyze_attendance_patterns(df)
        if not attendance_df.empty:
            risk_report['attendance_analysis'] = {
                'data': attendance_df,
                'high_absence_rate': attendance_df[attendance_df['Analysis_Type'] == 'Distribución General'].to_dict('records')
            }
        
        # 3. Estudiantes con baja participación
        low_participation_df = get_low_participation_students(df)
        if not low_participation_df.empty:
            risk_report['low_participation_students'] = {
                'count': len(low_participation_df),
                'percentage': format_percentage(len(low_participation_df) / len(df) * 100),
                'data': low_participation_df
            }
        
        # 4. Generar recomendaciones de intervención
        recommendations = []
        
        if risk_report['at_risk_students'].get('count', 0) > 0:
            recommendations.append("Implementar programa de tutoría personalizada para estudiantes en riesgo")
            recommendations.append("Establecer sistema de alertas tempranas para monitorear progreso académico")
        
        if risk_report['attendance_analysis']:
            recommendations.append("Desarrollar estrategias para mejorar la asistencia estudiantil")
            recommendations.append("Implementar programa de incentivos para la asistencia regular")
        
        if risk_report['low_participation_students'].get('count', 0) > 0:
            recommendations.append("Crear actividades de participación más atractivas y accesibles")
            recommendations.append("Proporcionar capacitación a docentes en técnicas de participación activa")
        
        risk_report['intervention_recommendations'] = recommendations
        
        # 5. Resumen ejecutivo
        total_risk_students = risk_report['at_risk_students'].get('count', 0)
        total_low_participation = risk_report['low_participation_students'].get('count', 0)
        
        risk_report['executive_summary'] = {
            'total_students_analyzed': len(df),
            'students_requiring_intervention': total_risk_students + total_low_participation,
            'intervention_percentage': format_percentage((total_risk_students + total_low_participation) / len(df) * 100),
            'primary_concerns': list(risk_report['risk_factors_summary'].get('common_factors', {}).keys())[:3]
        }
        
        logger.info(f"Reporte de riesgos generado exitosamente: {total_risk_students} estudiantes en riesgo identificados")
        return risk_report
        
    except Exception as e:
        logger.error(f"Error generando reporte de riesgos: {e}")
        return {}

def identify_intervention_priorities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identificar prioridades de intervención basándose en múltiples factores de riesgo.
    
    Args:
        df: DataFrame conteniendo datos de estudiantes
        
    Returns:
        DataFrame con estudiantes priorizados para intervención
    """
    try:
        log_analysis_step("Identificando prioridades de intervención")
        
        # Obtener estudiantes en riesgo
        at_risk_df = identify_at_risk_students(df)
        
        if at_risk_df.empty:
            logger.info("No se encontraron estudiantes en riesgo para priorización")
            return pd.DataFrame()
        
        # Crear sistema de priorización
        intervention_priorities = at_risk_df.copy()
        
        # Calcular puntuación de prioridad (mayor puntuación = mayor prioridad)
        intervention_priorities['PriorityScore'] = 0
        
        # Factor 1: Puntuación de riesgo (peso: 40%)
        max_risk_score = intervention_priorities['RiskScore'].max()
        if max_risk_score > 0:
            intervention_priorities['PriorityScore'] += (intervention_priorities['RiskScore'] / max_risk_score) * 40
        
        # Factor 2: Rendimiento académico (peso: 30%)
        if 'Class' in intervention_priorities.columns:
            performance_weights = {'L': 30, 'M': 15, 'H': 0}
            intervention_priorities['PriorityScore'] += intervention_priorities['Class'].map(performance_weights)
        
        # Factor 3: Asistencia (peso: 20%)
        if 'StudentAbsenceDays' in intervention_priorities.columns:
            attendance_weights = {'Above-7': 20, 'Under-7': 0}
            intervention_priorities['PriorityScore'] += intervention_priorities['StudentAbsenceDays'].map(attendance_weights)
        
        # Factor 4: Participación (peso: 10%)
        if 'TotalEngagement' in intervention_priorities.columns:
            # Normalizar participación (menor participación = mayor prioridad)
            max_engagement = intervention_priorities['TotalEngagement'].max()
            min_engagement = intervention_priorities['TotalEngagement'].min()
            if max_engagement > min_engagement:
                normalized_engagement = (max_engagement - intervention_priorities['TotalEngagement']) / (max_engagement - min_engagement)
                intervention_priorities['PriorityScore'] += normalized_engagement * 10
        
        # Clasificar por prioridad
        intervention_priorities = intervention_priorities.sort_values('PriorityScore', ascending=False)
        
        # Agregar categoría de prioridad
        intervention_priorities['PriorityCategory'] = 'Baja'
        intervention_priorities.loc[intervention_priorities['PriorityScore'] >= 70, 'PriorityCategory'] = 'Alta'
        intervention_priorities.loc[(intervention_priorities['PriorityScore'] >= 40) & 
                                   (intervention_priorities['PriorityScore'] < 70), 'PriorityCategory'] = 'Media'
        
        # Seleccionar columnas relevantes para el reporte
        priority_columns = ['RiskScore', 'RiskFactors', 'PriorityScore', 'PriorityCategory']
        if 'Class' in intervention_priorities.columns:
            priority_columns.append('Class')
        if 'TotalEngagement' in intervention_priorities.columns:
            priority_columns.append('TotalEngagement')
        
        result_df = intervention_priorities[priority_columns].copy()
        
        logger.info(f"Prioridades de intervención identificadas: {len(result_df)} estudiantes priorizados")
        return result_df
        
    except Exception as e:
        logger.error(f"Error identificando prioridades de intervención: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    # Probar el módulo de análisis de riesgos
    try:
        import sys
        import os
        
        # Agregar raíz del proyecto al path
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        from src.data.data_loader import load_and_validate_data
        from src.data.data_cleaner import clean_student_data
        
        print("Probando módulo de análisis de riesgos...")
        
        # Cargar y preparar datos
        df = load_and_validate_data()
        df_clean = clean_student_data(df)
        
        print(f"Datos cargados: {len(df_clean)} estudiantes")
        
        # Generar reporte de riesgos
        risk_report = generate_risk_report(df_clean)
        
        if risk_report:
            print("Reporte de riesgos generado exitosamente!")
            print(f"Estudiantes en riesgo: {risk_report['at_risk_students'].get('count', 0)}")
            print(f"Estudiantes con baja participación: {risk_report['low_participation_students'].get('count', 0)}")
            
            # Mostrar recomendaciones
            print("\nRecomendaciones de intervención:")
            for i, rec in enumerate(risk_report.get('intervention_recommendations', []), 1):
                print(f"{i}. {rec}")
        else:
            print("Error generando reporte de riesgos")
        
    except Exception as e:
        print(f"Error probando módulo de análisis de riesgos: {e}")
