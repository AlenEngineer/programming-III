"""
Módulo de visualización usando Seaborn para el Sistema de Análisis de Datos Académicos.
Crea gráficos y diagramas integrales para el análisis de rendimiento académico.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure as mfig
import seaborn as sns
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import sys
import os

# Add the parent directory to the path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import CHART_STYLE, FIGURE_SIZE, DPI, COLOR_PALETTE, CHARTS_DIR
from src.utils.helpers import log_analysis_step, create_output_directory

# Set up logging and styling
logger = logging.getLogger(__name__)
sns.set_style(CHART_STYLE)
plt.rcParams['figure.dpi'] = DPI
sns.set_palette(COLOR_PALETTE)

def configurar_estilo_graficos() -> None:
    """Configurar el estilo global de gráficos para visualización consistente."""
    try:
        sns.set_style(CHART_STYLE)
        plt.rcParams.update({
            'figure.figsize': FIGURE_SIZE,
            'figure.dpi': DPI,
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 11,
            'axes.grid': True,
            'grid.alpha': 0.3
        })
        logger.info("Estilo de gráficos configurado exitosamente")
    except Exception as e:
        logger.error(f"Error configurando estilo de gráficos: {e}")

def crear_grafico_distribucion_calificaciones(df: pd.DataFrame, ruta_guardado: Optional[str] = None) -> mfig.Figure:
    """
    Crear un gráfico de barras mostrando la distribución de calificaciones de rendimiento.
    
    Args:
        df: DataFrame conteniendo datos de estudiantes
        ruta_guardado: Ruta opcional para guardar el gráfico
        
    Returns:
        Objeto figura de Matplotlib
    """
    try:
        log_analysis_step("Creando gráfico de distribución de calificaciones")
        
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        
        if 'Class' not in df.columns:
            logger.warning("Columna Class no encontrada para distribución de calificaciones")
            return fig
        
        # Crear gráfico de conteo
        conteo_calificaciones = df['Class'].value_counts().sort_index()
        colores = sns.color_palette(COLOR_PALETTE, n_colors=len(conteo_calificaciones))
        
        barras = ax.bar(conteo_calificaciones.index, conteo_calificaciones.values.tolist(), color=colores, alpha=0.8)
        
        # Agregar etiquetas de valor en las barras
        for barra in barras:
            altura = barra.get_height()
            ax.text(barra.get_x() + barra.get_width()/2., altura,
                   f'{int(altura)}',
                   ha='center', va='bottom', fontweight='bold')
        
        # Personalizar el gráfico
        ax.set_title('Distribución de Calificaciones de Rendimiento Estudiantil', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Calificación de Rendimiento', fontsize=12)
        ax.set_ylabel('Número de Estudiantes', fontsize=12)
        
        # Agregar etiquetas de porcentaje
        total_estudiantes = len(df)
        for i, (calificacion, conteo) in enumerate(conteo_calificaciones.items()):
            porcentaje = (conteo / total_estudiantes) * 100
            ax.text(i, conteo + total_estudiantes * 0.01, f'{porcentaje:.1f}%',
                   ha='center', va='bottom', fontsize=10, style='italic')
        
        plt.tight_layout()
        
        # Guardar si se proporciona ruta
        if ruta_guardado:
            create_output_directory(Path(ruta_guardado).parent)
            plt.savefig(ruta_guardado, dpi=DPI, bbox_inches='tight')
            logger.info(f"Gráfico de distribución de calificaciones guardado en {ruta_guardado}")
        
        logger.info("Gráfico de distribución de calificaciones creado exitosamente")
        return fig
        
    except Exception as e:
        logger.error(f"Error creando gráfico de distribución de calificaciones: {e}")
        return plt.figure()

def create_subject_comparison_bar(df: pd.DataFrame, save_path: Optional[str] = None) -> mfig.Figure:
    """
    Create a bar chart comparing average engagement across subjects.
    
    Args:
        df: DataFrame containing student data
        save_path: Optional path to save the chart
        
    Returns:
        Matplotlib figure object
    """
    try:
        log_analysis_step("Creating subject comparison bar chart")
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        if 'Topic' not in df.columns or 'TotalEngagement' not in df.columns:
            logger.warning("Required columns not found for subject comparison")
            return fig
        
        # Calculate subject averages
        subject_avg = df.groupby('Topic')['TotalEngagement'].agg(['mean', 'std', 'count']).round(2)
        subject_avg = subject_avg.sort_values('mean', ascending=False)
        
        # Create bar plot
        bars = ax.bar(range(len(subject_avg)), subject_avg['mean'], 
                     yerr=subject_avg['std'], capsize=5, alpha=0.8,
                     color=sns.color_palette(COLOR_PALETTE, n_colors=len(subject_avg)))
        
        # Customize the chart
        ax.set_title('Average Student Engagement by Subject', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Subject', fontsize=12)
        ax.set_ylabel('Average Total Engagement Score', fontsize=12)
        ax.set_xticks(range(len(subject_avg)))
        ax.set_xticklabels(subject_avg.index, rotation=45, ha='right')
        
        # Add value labels on bars
        for i, (bar, avg_val, count) in enumerate(zip(bars, subject_avg['mean'], subject_avg['count'])):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                   f'{avg_val:.1f}\n(n={count})',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add average line
        overall_avg = df['TotalEngagement'].mean()
        ax.axhline(y=overall_avg, color='red', linestyle='--', alpha=0.7, 
                  label=f'Overall Average: {overall_avg:.1f}')
        ax.legend()
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            create_output_directory(Path(save_path).parent)
            plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
            logger.info(f"Subject comparison chart saved to {save_path}")
        
        logger.info("Subject comparison bar chart created successfully")
        return fig
        
    except Exception as e:
        logger.error(f"Error creating subject comparison chart: {e}")
        return plt.figure()

def create_semester_trend_line(df: pd.DataFrame, save_path: Optional[str] = None) -> mfig.Figure:
    """
    Create a line chart showing engagement trends between semesters.
    
    Args:
        df: DataFrame containing student data
        save_path: Optional path to save the chart
        
    Returns:
        Matplotlib figure object
    """
    try:
        log_analysis_step("Creating semester trend line chart")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        if 'Semester' not in df.columns:
            logger.warning("Semester column not found for trend analysis")
            return fig
        
        # Chart 1: Overall engagement by semester
        if 'TotalEngagement' in df.columns:
            semester_stats = df.groupby('Semester')['TotalEngagement'].agg(['mean', 'std', 'count'])
            
            ax1.bar(semester_stats.index, semester_stats['mean'], 
                   yerr=semester_stats['std'], capsize=10, alpha=0.7,
                   color=sns.color_palette(COLOR_PALETTE, n_colors=len(semester_stats)))
            
            ax1.set_title('Average Engagement by Semester', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Semester')
            ax1.set_ylabel('Average Total Engagement')
            
            # Add value labels
            for i, (sem, avg_val, count) in enumerate(zip(semester_stats.index, 
                                                         semester_stats['mean'], 
                                                         semester_stats['count'])):
                ax1.text(i, avg_val + semester_stats['std'].iloc[i] + 5,
                        f'{avg_val:.1f}\n(n={count})',
                        ha='center', va='bottom', fontweight='bold')
        
        # Chart 2: Performance distribution by semester
        if 'Class' in df.columns:
            semester_performance = pd.crosstab(df['Semester'], df['Class'], normalize='index') * 100
            
            semester_performance.plot(kind='bar', ax=ax2, alpha=0.8,
                                    color=sns.color_palette(COLOR_PALETTE, n_colors=len(semester_performance.columns)))
            
            ax2.set_title('Performance Distribution by Semester', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Semester')
            ax2.set_ylabel('Percentage of Students (%)')
            ax2.legend(title='Performance Grade', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            create_output_directory(Path(save_path).parent)
            plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
            logger.info(f"Semester trend chart saved to {save_path}")
        
        logger.info("Semester trend line chart created successfully")
        return fig
        
    except Exception as e:
        logger.error(f"Error creating semester trend chart: {e}")
        return plt.figure()

def create_performance_scatter(df: pd.DataFrame, save_path: Optional[str] = None) -> mfig.Figure:
    """
    Create scatter plots showing relationships between engagement metrics and performance.
    
    Args:
        df: DataFrame containing student data
        save_path: Optional path to save the chart
        
    Returns:
        Matplotlib figure object
    """
    try:
        log_analysis_step("Creating performance scatter plot")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        engagement_cols = ['RaisedHands', 'VisitedResources', 'AnnouncementsView', 'Discussion']
        axes = [ax1, ax2, ax3, ax4]
        
        for i, (col, ax) in enumerate(zip(engagement_cols, axes)):
            if col in df.columns and 'TotalEngagement' in df.columns:
                # Create scatter plot with performance color coding
                if 'Class' in df.columns:
                    sns.scatterplot(data=df, x=col, y='TotalEngagement', hue='Class',
                                  ax=ax, alpha=0.7, s=60)
                else:
                    ax.scatter(df[col], df['TotalEngagement'], alpha=0.7)
                
                # Add trend line
                z = np.polyfit(df[col].dropna(), df['TotalEngagement'].dropna(), 1)
                p = np.poly1d(z)
                ax.plot(df[col], p(df[col]), "r--", alpha=0.8, linewidth=2)
                
                # Calculate correlation
                correlation = df[col].corr(df['TotalEngagement'])
                
                ax.set_title(f'{col} vs Total Engagement\n(r = {correlation:.3f})', 
                           fontsize=12, fontweight='bold')
                ax.set_xlabel(col.replace('_', ' '))
                ax.set_ylabel('Total Engagement')
                
                if 'Class' in df.columns:
                    ax.legend(title='Performance Grade', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.suptitle('Engagement Metrics vs Performance Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            create_output_directory(Path(save_path).parent)
            plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
            logger.info(f"Performance scatter plot saved to {save_path}")
        
        logger.info("Performance scatter plot created successfully")
        return fig
        
    except Exception as e:
        logger.error(f"Error creating performance scatter plot: {e}")
        return plt.figure()

def create_attendance_heatmap(df: pd.DataFrame, save_path: Optional[str] = None) -> mfig.Figure:
    """
    Create a heatmap showing attendance patterns across different dimensions.
    
    Args:
        df: DataFrame containing student data
        save_path: Optional path to save the chart
        
    Returns:
        Matplotlib figure object
    """
    try:
        log_analysis_step("Creating attendance heatmap")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Heatmap 1: Attendance by Topic and Semester
        if all(col in df.columns for col in ['Topic', 'Semester', 'StudentAbsenceDays']):
            # Create a pivot table for attendance
            attendance_pivot = pd.crosstab([df['Topic']], 
                                         [df['Semester'], df['StudentAbsenceDays']], 
                                         normalize='index') * 100
            
            sns.heatmap(attendance_pivot, annot=True, fmt='.1f', cmap='RdYlGn_r',
                       ax=ax1, cbar_kws={'label': 'Percentage of Students (%)'})
            ax1.set_title('Attendance Patterns by Subject and Semester', fontweight='bold')
            ax1.set_xlabel('Semester - Absence Days')
            ax1.set_ylabel('Subject')
            
        # Heatmap 2: Performance vs Attendance
        if all(col in df.columns for col in ['Class', 'StudentAbsenceDays']):
            performance_attendance = pd.crosstab(df['Class'], df['StudentAbsenceDays'], 
                                               normalize='index') * 100
            
            sns.heatmap(performance_attendance, annot=True, fmt='.1f', cmap='RdYlBu_r',
                       ax=ax2, cbar_kws={'label': 'Percentage of Students (%)'})
            ax2.set_title('Performance vs Attendance Relationship', fontweight='bold')
            ax2.set_xlabel('Attendance Category')
            ax2.set_ylabel('Performance Grade')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            create_output_directory(Path(save_path).parent)
            plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
            logger.info(f"Attendance heatmap saved to {save_path}")
        
        logger.info("Attendance heatmap created successfully")
        return fig
        
    except Exception as e:
        logger.error(f"Error creating attendance heatmap: {e}")
        return plt.figure()

def create_demographic_analysis_charts(df: pd.DataFrame, save_path: Optional[str] = None) -> mfig.Figure:
    """
    Create comprehensive demographic analysis charts.
    
    Args:
        df: DataFrame containing student data
        save_path: Optional path to save the chart
        
    Returns:
        Matplotlib figure object
    """
    try:
        log_analysis_step("Creating demographic analysis charts")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Chart 1: Gender distribution
        if 'gender' in df.columns:
            gender_counts = df['gender'].value_counts()
            colors = sns.color_palette(COLOR_PALETTE, n_colors=len(gender_counts))
            
            wedges, texts, autotexts = ax1.pie(gender_counts.values, labels=gender_counts.index, 
                                              autopct='%1.1f%%', startangle=90, colors=colors)
            ax1.set_title('Student Distribution by Gender', fontweight='bold')
            
            # Make percentage text bold
            for autotext in autotexts:
                autotext.set_fontweight('bold')
        
        # Chart 2: Performance by Gender
        if all(col in df.columns for col in ['gender', 'Class']):
            gender_performance = pd.crosstab(df['gender'], df['Class'], normalize='index') * 100
            gender_performance.plot(kind='bar', ax=ax2, alpha=0.8,
                                  color=sns.color_palette(COLOR_PALETTE, n_colors=len(gender_performance.columns)))
            ax2.set_title('Performance Distribution by Gender', fontweight='bold')
            ax2.set_xlabel('Gender')
            ax2.set_ylabel('Percentage (%)')
            ax2.legend(title='Performance Grade')
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
        
        # Chart 3: Top Nationalities
        if 'Nationality' in df.columns:
            top_nationalities = df['Nationality'].value_counts().head(8)
            bars = ax3.bar(range(len(top_nationalities)), top_nationalities.values, 
                          color=sns.color_palette(COLOR_PALETTE, n_colors=len(top_nationalities)))
            ax3.set_title('Top Student Nationalities', fontweight='bold')
            ax3.set_xlabel('Nationality')
            ax3.set_ylabel('Number of Students')
            ax3.set_xticks(range(len(top_nationalities)))
            ax3.set_xticklabels(top_nationalities.index, rotation=45, ha='right')
            
            # Add value labels
            for bar, count in zip(bars, top_nationalities.values):
                ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                        f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # Chart 4: Stage distribution
        if 'StageID' in df.columns:
            stage_counts = df['StageID'].value_counts()
            sns.barplot(x=stage_counts.index, y=stage_counts.values, ax=ax4,
                       palette=COLOR_PALETTE)
            ax4.set_title('Student Distribution by Educational Stage', fontweight='bold')
            ax4.set_xlabel('Educational Stage')
            ax4.set_ylabel('Number of Students')
            
            # Add value labels
            for i, count in enumerate(stage_counts.values):
                ax4.text(i, count + stage_counts.max() * 0.01,
                        f'{count}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            create_output_directory(Path(save_path).parent)
            plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
            logger.info(f"Demographic analysis charts saved to {save_path}")
        
        logger.info("Demographic analysis charts created successfully")
        return fig
        
    except Exception as e:
        logger.error(f"Error creating demographic analysis charts: {e}")
        return plt.figure()

def create_engagement_correlation_matrix(df: pd.DataFrame, save_path: Optional[str] = None) -> mfig.Figure:
    """
    Create a correlation matrix heatmap for engagement metrics.
    
    Args:
        df: DataFrame containing student data
        save_path: Optional path to save the chart
        
    Returns:
        Matplotlib figure object
    """
    try:
        log_analysis_step("Creating engagement correlation matrix")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Select engagement columns
        engagement_cols = ['RaisedHands', 'VisitedResources', 'AnnouncementsView', 'Discussion']
        if 'TotalEngagement' in df.columns:
            engagement_cols.append('TotalEngagement')
        
        # Filter to existing columns
        available_cols = [col for col in engagement_cols if col in df.columns]
        
        if len(available_cols) < 2:
            logger.warning("Insufficient engagement columns for correlation analysis")
            return fig
        
        # Calculate correlation matrix
        corr_matrix = df[available_cols].corr()
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.3f', 
                   cmap='coolwarm', center=0, square=True, ax=ax,
                   cbar_kws={'label': 'Correlation Coefficient'})
        
        ax.set_title('Engagement Metrics Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
        
        # Rotate labels for better readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            create_output_directory(Path(save_path).parent)
            plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
            logger.info(f"Correlation matrix saved to {save_path}")
        
        logger.info("Engagement correlation matrix created successfully")
        return fig
        
    except Exception as e:
        logger.error(f"Error creating correlation matrix: {e}")
        return plt.figure()

def create_all_visualizations(df: pd.DataFrame, stats: Dict[str, Any], 
                            groups: Dict[str, Any], save_charts: bool = True) -> Dict[str, mfig.Figure]:
    """
    Create all visualization charts for the academic analysis.
    
    Args:
        df: Cleaned DataFrame with student data
        stats: Statistical analysis results
        groups: Grouping analysis results
        save_charts: Whether to save charts to files
        
    Returns:
        Dictionary containing all created figures
    """
    try:
        log_analysis_step("Creating all visualization charts")
        
        # Set up chart styling
        setup_chart_style()
        
        charts = {}
        
        # Define chart creation functions and their save paths
        chart_configs = [
            ('grade_distribution', crear_grafico_distribucion_calificaciones, 'grade_distribution.png'),
            ('subject_comparison', create_subject_comparison_bar, 'subject_comparison.png'),
            ('semester_trends', create_semester_trend_line, 'semester_trends.png'),
            ('performance_scatter', create_performance_scatter, 'performance_scatter.png'),
            ('attendance_heatmap', create_attendance_heatmap, 'attendance_heatmap.png'),
            ('demographic_analysis', create_demographic_analysis_charts, 'demographic_analysis.png'),
            ('correlation_matrix', create_engagement_correlation_matrix, 'correlation_matrix.png')
        ]
        
        # Create each chart
        for chart_name, chart_function, filename in chart_configs:
            try:
                save_path = str(CHARTS_DIR / filename) if save_charts else None
                charts[chart_name] = chart_function(df, save_path)
                logger.info(f"Created {chart_name} chart")
            except Exception as e:
                logger.error(f"Error creating {chart_name} chart: {e}")
                charts[chart_name] = plt.figure()  # Empty figure as fallback
        
        logger.info(f"All visualizations created successfully: {len(charts)} charts")
        return charts
        
    except Exception as e:
        logger.error(f"Error creating all visualizations: {e}")
        return {}

def crear_todas_visualizaciones(df: pd.DataFrame, stats: Dict[str, Any], 
                              groups: Dict[str, Any], guardar_graficos: bool = False) -> Dict[str, Any]:
    """
    Crear todos los gráficos de visualización para el análisis académico.
    
    Args:
        df: DataFrame limpio con datos de estudiantes
        stats: Resultados del análisis estadístico
        groups: Resultados del análisis de agrupación
        guardar_graficos: Si guardar gráficos en archivos
        
    Returns:
        Diccionario conteniendo todas las figuras creadas
    """
    try:
        log_analysis_step("Creando todos los gráficos de visualización")
        
        # Configurar estilo de gráficos
        configurar_estilo_graficos()
        
        graficos = {}
        
        # Definir configuraciones de gráficos con nombres en español
        configuraciones_graficos = [
            ('distribucion_calificaciones', crear_grafico_distribucion_calificaciones, 'distribucion_calificaciones.png'),
            ('comparacion_materias', create_subject_comparison_bar, 'comparacion_materias.png'),
            ('tendencias_semestre', create_semester_trend_line, 'tendencias_semestre.png'),
            ('dispersion_rendimiento', create_performance_scatter, 'dispersion_rendimiento.png'),
            ('mapa_calor_asistencia', create_attendance_heatmap, 'mapa_calor_asistencia.png'),
            ('analisis_demografico', create_demographic_analysis_charts, 'analisis_demografico.png'),
            ('matriz_correlacion', create_engagement_correlation_matrix, 'matriz_correlacion.png')
        ]
        
        # Crear cada gráfico
        for nombre_grafico, funcion_grafico, nombre_archivo in configuraciones_graficos:
            try:
                ruta_guardado = str(CHARTS_DIR / nombre_archivo) if guardar_graficos else None
                
                # Crear el gráfico
                figura = funcion_grafico(df, ruta_guardado)
                
                # Mapear nombres españoles a tipos originales para aplicar títulos
                mapeo_tipos = {
                    'distribucion_calificaciones': 'grade_distribution',
                    'comparacion_materias': 'subject_comparison',
                    'tendencias_semestre': 'semester_trends',
                    'dispersion_rendimiento': 'performance_scatter',
                    'mapa_calor_asistencia': 'attendance_heatmap',
                    'analisis_demografico': 'demographic_analysis',
                    'matriz_correlacion': 'correlation_matrix'
                }
                
                tipo_original = mapeo_tipos.get(nombre_grafico)
                
                # Aplicar títulos en español y guardar nuevamente si es necesario
                if guardar_graficos and figura and tipo_original and hasattr(figura, 'axes') and figura.axes:
                    # Para gráficos con múltiples subplots, aplicar solo al primero
                    ax_principal = figura.axes[0]
                    aplicar_titulos_espanol(figura, ax_principal, tipo_original)
                    
                    # Guardar nuevamente con títulos en español
                    figura.savefig(ruta_guardado, dpi=DPI, bbox_inches='tight')
                    logger.info(f"Títulos en español aplicados y gráfico guardado: {ruta_guardado}")
                
                graficos[nombre_grafico] = figura
                logger.info(f"Creado gráfico {nombre_grafico}")
            except Exception as e:
                logger.error(f"Error creando gráfico {nombre_grafico}: {e}")
                graficos[nombre_grafico] = plt.figure()  # Figura vacía como respaldo
        
        logger.info(f"Todas las visualizaciones creadas exitosamente: {len(graficos)} gráficos")
        return graficos
        
    except Exception as e:
        logger.error(f"Error creando todas las visualizaciones: {e}")
        return {}

# Aliases en español para compatibilidad
setup_chart_style = configurar_estilo_graficos
crear_grafico_comparacion_materias = create_subject_comparison_bar
crear_grafico_tendencias_semestre = create_semester_trend_line
crear_grafico_dispersion_rendimiento = create_performance_scatter
crear_mapa_calor_asistencia = create_attendance_heatmap
crear_graficos_analisis_demografico = create_demographic_analysis_charts
crear_matriz_correlacion_participacion = create_engagement_correlation_matrix

# Alias para compatibilidad con nombres en inglés (manteniendo la función original)
def create_all_visualizations_wrapper(df: pd.DataFrame, stats: Dict[str, Any], 
                                     groups: Dict[str, Any], save_charts: bool = True) -> Dict[str, Any]:
    """Wrapper function to maintain English API compatibility."""
    return crear_todas_visualizaciones(df, stats, groups, save_charts)

def traducir_titulos_graficos():
    """Configurar matplotlib para mostrar títulos en español."""
    plt.rcParams.update({
        'axes.unicode_minus': False,
        'font.family': ['DejaVu Sans', 'Arial', 'sans-serif']
    })

def aplicar_titulos_espanol(fig, ax, tipo_grafico: str):
    """
    Aplicar títulos en español a los gráficos existentes.
    
    Args:
        fig: Figura de matplotlib
        ax: Axes del gráfico (puede ser un solo ax o lista de axes)
        tipo_grafico: Tipo de gráfico para determinar el título
    """
    titulos_espanol = {
        'grade_distribution': {
            'title': 'Distribución de Calificaciones de Rendimiento Estudiantil',
            'xlabel': 'Calificación de Rendimiento',
            'ylabel': 'Número de Estudiantes'
        },
        'subject_comparison': {
            'title': 'Participación Promedio de Estudiantes por Materia',
            'xlabel': 'Materia',
            'ylabel': 'Puntuación Promedio de Participación Total'
        },
        'semester_trends': {
            'title': 'Tendencias de Participación por Semestre',
            'xlabel': 'Semestre',
            'ylabel': 'Participación Promedio'
        },
        'performance_scatter': {
            'title': 'Análisis de Métricas de Participación vs Rendimiento',
            'xlabel': 'Participación Total',
            'ylabel': 'Rendimiento Académico',
            'subplots': {
                'RaisedHands': {
                    'title': 'Manos Alzadas vs Participación Total',
                    'xlabel': 'Manos Alzadas',
                    'ylabel': 'Participación Total'
                },
                'VisitedResources': {
                    'title': 'Recursos Visitados vs Participación Total', 
                    'xlabel': 'Recursos Visitados',
                    'ylabel': 'Participación Total'
                },
                'AnnouncementsView': {
                    'title': 'Anuncios Vistos vs Participación Total',
                    'xlabel': 'Anuncios Vistos', 
                    'ylabel': 'Participación Total'
                },
                'Discussion': {
                    'title': 'Discusión vs Participación Total',
                    'xlabel': 'Participación en Discusión',
                    'ylabel': 'Participación Total'
                }
            }
        },
        'attendance_heatmap': {
            'title': 'Mapa de Calor de Asistencia por Materia y Semestre',
            'xlabel': 'Semestre',
            'ylabel': 'Materia'
        },
        'demographic_analysis': {
            'title': 'Análisis Demográfico de Participación Estudiantil',
            'xlabel': 'Categorías Demográficas',
            'ylabel': 'Distribución'
        },
        'correlation_matrix': {
            'title': 'Matriz de Correlación de Métricas de Participación',
            'xlabel': 'Variables de Participación',
            'ylabel': 'Variables de Participación'
        }
    }
    
    if tipo_grafico in titulos_espanol:
        titulos = titulos_espanol[tipo_grafico]
        
        # Caso especial para performance_scatter con subgráficos
        if tipo_grafico == 'performance_scatter' and hasattr(fig, 'axes') and len(fig.axes) >= 4:
            # Actualizar título principal
            fig.suptitle(titulos['title'], fontsize=16, fontweight='bold')
            
            # Actualizar subgráficos individuales
            engagement_cols = ['RaisedHands', 'VisitedResources', 'AnnouncementsView', 'Discussion']
            for i, (col, subplot_ax) in enumerate(zip(engagement_cols, fig.axes[:4])):
                if col in titulos['subplots']:
                    subplot_titulos = titulos['subplots'][col]
                    # Mantener la correlación si existe en el título original
                    titulo_original = subplot_ax.get_title()
                    if 'r =' in titulo_original:
                        correlacion = titulo_original.split('r =')[1].strip().rstrip(')')
                        nuevo_titulo = f"{subplot_titulos['title']}\n(r = {correlacion}"
                    else:
                        nuevo_titulo = subplot_titulos['title']
                    
                    subplot_ax.set_title(nuevo_titulo, fontsize=12, fontweight='bold')
                    subplot_ax.set_xlabel(subplot_titulos['xlabel'], fontsize=10)
                    subplot_ax.set_ylabel(subplot_titulos['ylabel'], fontsize=10)
                    
                    # Traducir leyenda si existe
                    legend = subplot_ax.get_legend()
                    if legend:
                        legend.set_title('Calificación de Rendimiento')
        else:
            # Caso normal para gráficos simples
            if isinstance(ax, list):
                ax = ax[0]  # Tomar el primer axis si es una lista
            ax.set_title(titulos['title'], fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel(titulos['xlabel'], fontsize=12)
            ax.set_ylabel(titulos['ylabel'], fontsize=12)
        
        # Asegurar que el layout se ajuste bien
        fig.tight_layout()
        
        logger.info(f"Títulos en español aplicados para gráfico tipo: {tipo_grafico}")

# Test the visualization module
try:
    import sys
    import os
    
    # Add project root to path  
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Import modules - ignore static analysis warnings, runtime works fine
    from src.data.data_loader import load_and_validate_data  # type: ignore
    from src.data.data_cleaner import clean_student_data  # type: ignore
    from src.analysis.statistics import calculate_all_statistics  # type: ignore
    from src.analysis.grouping import perform_all_groupings  # type: ignore
    
    # Load and prepare data
    df = load_and_validate_data()
    df_clean = clean_student_data(df)
    stats = calculate_all_statistics(df_clean)
    groups = perform_all_groupings(df_clean)
    
    print("Data prepared successfully!")
    
    # Create visualizations
    charts = create_all_visualizations(df_clean, stats, groups, save_charts=True)
    print(f"Created {len(charts)} visualization charts")
    
    # Display the charts
    print("Charts created:")
    for chart_name in charts.keys():
        print(f"  - {chart_name}")
    
    # Show one chart as example
    plt.show()
    
except Exception as e:
    print(f"Error testing visualization module: {e}")
