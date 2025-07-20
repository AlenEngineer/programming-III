"""
Módulo de visualización usando Seaborn para el Sistema de Análisis de Datos Académicos.
Crea gráficos y visualizaciones integrales para el análisis de rendimiento académico.
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

# Agregar el directorio padre al path para importar config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import CHART_STYLE, FIGURE_SIZE, DPI, COLOR_PALETTE, CHARTS_DIR
from src.utils.helpers import log_analysis_step, create_output_directory

# Configurar logging y estilo
logger = logging.getLogger(__name__)
sns.set_style(CHART_STYLE)
plt.rcParams['figure.dpi'] = DPI
sns.set_palette(COLOR_PALETTE)

def setup_chart_style() -> None:
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

def create_grade_distribution_chart(df: pd.DataFrame, save_path: Optional[str] = None) -> mfig.Figure:
    """
    Crear un gráfico de barras mostrando la distribución de calificaciones de rendimiento.
    
    Args:
        df: DataFrame conteniendo datos de estudiantes
        save_path: Ruta opcional para guardar el gráfico
        
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
        grade_counts = df['Class'].value_counts().sort_index()
        colors = sns.color_palette(COLOR_PALETTE, n_colors=len(grade_counts))
        
        bars = ax.bar(grade_counts.index, grade_counts.values.tolist(), color=colors, alpha=0.8)
        
        # Agregar etiquetas de valores en las barras
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontweight='bold')
        
        # Personalizar el gráfico
        ax.set_title('Distribución de Calificaciones de Rendimiento Estudiantil', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Calificación de Rendimiento', fontsize=12)
        ax.set_ylabel('Número de Estudiantes', fontsize=12)
        
        # Agregar etiquetas de porcentaje
        total_students = len(df)
        for i, (grade, count) in enumerate(grade_counts.items()):
            percentage = (count / total_students) * 100
            ax.text(i, count + total_students * 0.01, f'{percentage:.1f}%',
                   ha='center', va='bottom', fontsize=10, style='italic')
        
        plt.tight_layout()
        
        # Guardar si se proporciona ruta
        if save_path:
            create_output_directory(Path(save_path).parent)
            plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
            logger.info(f"Gráfico de distribución de calificaciones guardado en {save_path}")
        
        logger.info("Gráfico de distribución de calificaciones creado exitosamente")
        return fig
        
    except Exception as e:
        logger.error(f"Error creando gráfico de distribución de calificaciones: {e}")
        return plt.figure()

def create_subject_comparison_bar(df: pd.DataFrame, save_path: Optional[str] = None) -> mfig.Figure:
    """
    Crear un gráfico de barras comparando el promedio de participación entre materias.
    
    Args:
        df: DataFrame conteniendo datos de estudiantes
        save_path: Ruta opcional para guardar el gráfico
        
    Returns:
        Objeto figura de Matplotlib
    """
    try:
        log_analysis_step("Creando gráfico de comparación de materias")
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        if 'Topic' not in df.columns or 'TotalEngagement' not in df.columns:
            logger.warning("Columnas requeridas no encontradas para comparación de materias")
            return fig
        
        # Calcular promedios por materia
        subject_avg = df.groupby('Topic')['TotalEngagement'].agg(['mean', 'std', 'count']).round(2)
        subject_avg = subject_avg.sort_values('mean', ascending=False)
        
        # Crear gráfico de barras
        bars = ax.bar(range(len(subject_avg)), subject_avg['mean'], 
                     yerr=subject_avg['std'], capsize=5, alpha=0.8,
                     color=sns.color_palette(COLOR_PALETTE, n_colors=len(subject_avg)))
        
        # Personalizar el gráfico
        ax.set_title('Promedio de Participación Estudiantil por Materia', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Materia', fontsize=12)
        ax.set_ylabel('Promedio de Puntuación de Participación Total', fontsize=12)
        ax.set_xticks(range(len(subject_avg)))
        ax.set_xticklabels(subject_avg.index, rotation=45, ha='right')
        
        # Agregar etiquetas de valores en las barras
        for i, (bar, avg_val, count) in enumerate(zip(bars, subject_avg['mean'], subject_avg['count'])):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                   f'{avg_val:.1f}\n(n={count})',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Agregar línea de promedio general
        overall_avg = df['TotalEngagement'].mean()
        ax.axhline(y=overall_avg, color='red', linestyle='--', alpha=0.7, 
                  label=f'Promedio General: {overall_avg:.1f}')
        ax.legend()
        
        plt.tight_layout()
        
        # Guardar si se proporciona ruta
        if save_path:
            create_output_directory(Path(save_path).parent)
            plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
            logger.info(f"Gráfico de comparación de materias guardado en {save_path}")
        
        logger.info("Gráfico de comparación de materias creado exitosamente")
        return fig
        
    except Exception as e:
        logger.error(f"Error creando gráfico de comparación de materias: {e}")
        return plt.figure()

def create_semester_trend_line(df: pd.DataFrame, save_path: Optional[str] = None) -> mfig.Figure:
    """
    Crear un gráfico de líneas mostrando tendencias de participación entre semestres.
    
    Args:
        df: DataFrame conteniendo datos de estudiantes
        save_path: Ruta opcional para guardar el gráfico
        
    Returns:
        Objeto figura de Matplotlib
    """
    try:
        log_analysis_step("Creando gráfico de tendencias por semestre")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        if 'Semester' not in df.columns:
            logger.warning("Columna Semester no encontrada para análisis de tendencias")
            return fig
        
        # Gráfico 1: Participación general por semestre
        if 'TotalEngagement' in df.columns:
            semester_stats = df.groupby('Semester')['TotalEngagement'].agg(['mean', 'std', 'count'])
            
            ax1.bar(semester_stats.index, semester_stats['mean'], 
                   yerr=semester_stats['std'], capsize=10, alpha=0.7,
                   color=sns.color_palette(COLOR_PALETTE, n_colors=len(semester_stats)))
            
            ax1.set_title('Promedio de Participación por Semestre', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Semestre')
            ax1.set_ylabel('Promedio de Participación Total')
            
            # Agregar etiquetas
            for i, (sem, avg_val, count) in enumerate(zip(semester_stats.index, 
                                                         semester_stats['mean'], 
                                                         semester_stats['count'])):
                ax1.text(i, avg_val + semester_stats['std'].iloc[i] + 5,
                        f'{avg_val:.1f}\n(n={count})',
                        ha='center', va='bottom', fontweight='bold')
        
        # Gráfico 2: Distribución de rendimiento por semestre
        if 'Class' in df.columns:
            semester_performance = pd.crosstab(df['Semester'], df['Class'], normalize='index') * 100
            
            semester_performance.plot(kind='bar', ax=ax2, alpha=0.8,
                                    color=sns.color_palette(COLOR_PALETTE, n_colors=len(semester_performance.columns)))
            
            ax2.set_title('Distribución de Rendimiento por Semestre', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Semestre')
            ax2.set_ylabel('Porcentaje de Estudiantes (%)')
            ax2.legend(title='Calificación de Rendimiento', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
        
        plt.tight_layout()
        
        # Guardar si se proporciona ruta
        if save_path:
            create_output_directory(Path(save_path).parent)
            plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
            logger.info(f"Gráfico de tendencias por semestre guardado en {save_path}")
        
        logger.info("Gráfico de tendencias por semestre creado exitosamente")
        return fig
        
    except Exception as e:
        logger.error(f"Error creando gráfico de tendencias por semestre: {e}")
        return plt.figure()

def create_performance_scatter(df: pd.DataFrame, save_path: Optional[str] = None) -> mfig.Figure:
    """
    Crear gráficos de dispersión mostrando relaciones entre métricas de participación y rendimiento.
    
    Args:
        df: DataFrame conteniendo datos de estudiantes
        save_path: Ruta opcional para guardar el gráfico
        
    Returns:
        Objeto figura de Matplotlib
    """
    try:
        log_analysis_step("Creando gráfico de dispersión de rendimiento")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        engagement_cols = ['RaisedHands', 'VisitedResources', 'AnnouncementsView', 'Discussion']
        axes = [ax1, ax2, ax3, ax4]
        
        for i, (col, ax) in enumerate(zip(engagement_cols, axes)):
            if col in df.columns and 'TotalEngagement' in df.columns:
                # Crear gráfico de dispersión con codificación de color por rendimiento
                if 'Class' in df.columns:
                    sns.scatterplot(data=df, x=col, y='TotalEngagement', hue='Class',
                                  ax=ax, alpha=0.7, s=60)
                else:
                    ax.scatter(df[col], df['TotalEngagement'], alpha=0.7)
                
                # Agregar línea de tendencia
                z = np.polyfit(df[col].dropna(), df['TotalEngagement'].dropna(), 1)
                p = np.poly1d(z)
                ax.plot(df[col], p(df[col]), "r--", alpha=0.8, linewidth=2)
                
                # Calcular correlación
                correlation = df[col].corr(df['TotalEngagement'])
                
                ax.set_title(f'{col} vs Puntuación Total de Participación\n(r = {correlation:.3f})', 
                           fontsize=12, fontweight='bold')
                ax.set_xlabel(col.replace('_', ' '))
                ax.set_ylabel('Puntuación Total de Participación')
                
                if 'Class' in df.columns:
                    ax.legend(title='Calificación de Rendimiento', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.suptitle('Métricas de Participación vs Análisis de Rendimiento', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Guardar si se proporciona ruta
        if save_path:
            create_output_directory(Path(save_path).parent)
            plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
            logger.info(f"Gráfico de dispersión de rendimiento guardado en {save_path}")
        
        logger.info("Gráfico de dispersión de rendimiento creado exitosamente")
        return fig
        
    except Exception as e:
        logger.error(f"Error creando gráfico de dispersión de rendimiento: {e}")
        return plt.figure()

def create_attendance_heatmap(df: pd.DataFrame, save_path: Optional[str] = None) -> mfig.Figure:
    """
    Crear un mapa de calor mostrando patrones de asistencia en diferentes dimensiones.
    
    Args:
        df: DataFrame conteniendo datos de estudiantes
        save_path: Ruta opcional para guardar el gráfico
        
    Returns:
        Objeto figura de Matplotlib
    """
    try:
        log_analysis_step("Creando mapa de calor de asistencia")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Mapa de calor 1: Asistencia por Tema y Semestre
        if all(col in df.columns for col in ['Topic', 'Semester', 'StudentAbsenceDays']):
            # Crear una tabla pivot para asistencia
            attendance_pivot = pd.crosstab([df['Topic']], 
                                         [df['Semester'], df['StudentAbsenceDays']], 
                                         normalize='index') * 100
            
            sns.heatmap(attendance_pivot, annot=True, fmt='.1f', cmap='RdYlGn_r',
                       ax=ax1, cbar_kws={'label': 'Porcentaje de Estudiantes (%)'})
            ax1.set_title('Patrones de Asistencia por Tema y Semestre', fontweight='bold')
            ax1.set_xlabel('Semestre - Días de Ausencia')
            ax1.set_ylabel('Tema')
            
        # Mapa de calor 2: Rendimiento vs Asistencia
        if all(col in df.columns for col in ['Class', 'StudentAbsenceDays']):
            performance_attendance = pd.crosstab(df['Class'], df['StudentAbsenceDays'], 
                                               normalize='index') * 100
            
            sns.heatmap(performance_attendance, annot=True, fmt='.1f', cmap='RdYlBu_r',
                       ax=ax2, cbar_kws={'label': 'Porcentaje de Estudiantes (%)'})
            ax2.set_title('Relación entre Rendimiento y Asistencia', fontweight='bold')
            ax2.set_xlabel('Categoría de Asistencia')
            ax2.set_ylabel('Calificación de Rendimiento')
        
        plt.tight_layout()
        
        # Guardar si se proporciona ruta
        if save_path:
            create_output_directory(Path(save_path).parent)
            plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
            logger.info(f"Mapa de calor de asistencia guardado en {save_path}")
        
        logger.info("Mapa de calor de asistencia creado exitosamente")
        return fig
        
    except Exception as e:
        logger.error(f"Error creando mapa de calor de asistencia: {e}")
        return plt.figure()

def create_demographic_analysis_charts(df: pd.DataFrame, save_path: Optional[str] = None) -> mfig.Figure:
    """
    Crear gráficos de análisis demográfico integrales.
    
    Args:
        df: DataFrame conteniendo datos de estudiantes
        save_path: Ruta opcional para guardar el gráfico
        
    Returns:
        Objeto figura de Matplotlib
    """
    try:
        log_analysis_step("Creando gráficos de análisis demográfico")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Gráfico 1: Distribución de género
        if 'gender' in df.columns:
            gender_counts = df['gender'].value_counts()
            colors = sns.color_palette(COLOR_PALETTE, n_colors=len(gender_counts))
            
            wedges, texts, autotexts = ax1.pie(gender_counts.values, labels=gender_counts.index, 
                                              autopct='%1.1f%%', startangle=90, colors=colors)
            ax1.set_title('Distribución de Estudiantes por Género', fontweight='bold')
            
            # Hacer el texto de porcentaje en negrita
            for autotext in autotexts:
                autotext.set_fontweight('bold')
        
        # Gráfico 2: Rendimiento por Género
        if all(col in df.columns for col in ['gender', 'Class']):
            gender_performance = pd.crosstab(df['gender'], df['Class'], normalize='index') * 100
            gender_performance.plot(kind='bar', ax=ax2, alpha=0.8,
                                  color=sns.color_palette(COLOR_PALETTE, n_colors=len(gender_performance.columns)))
            ax2.set_title('Distribución de Rendimiento por Género', fontweight='bold')
            ax2.set_xlabel('Género')
            ax2.set_ylabel('Porcentaje (%)')
            ax2.legend(title='Calificación de Rendimiento')
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
        
        # Gráfico 3: Top Nacionalidades
        if 'Nationality' in df.columns:
            top_nationalities = df['Nationality'].value_counts().head(8)
            bars = ax3.bar(range(len(top_nationalities)), top_nationalities.values, 
                          color=sns.color_palette(COLOR_PALETTE, n_colors=len(top_nationalities)))
            ax3.set_title('Top Nacionalidades de Estudiantes', fontweight='bold')
            ax3.set_xlabel('Nacionalidad')
            ax3.set_ylabel('Número de Estudiantes')
            ax3.set_xticks(range(len(top_nationalities)))
            ax3.set_xticklabels(top_nationalities.index, rotation=45, ha='right')
            
            # Agregar etiquetas de valores
            for bar, count in zip(bars, top_nationalities.values):
                ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                        f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # Gráfico 4: Distribución por Etapa
        if 'StageID' in df.columns:
            stage_counts = df['StageID'].value_counts()
            sns.barplot(x=stage_counts.index, y=stage_counts.values, ax=ax4,
                       palette=COLOR_PALETTE)
            ax4.set_title('Distribución de Estudiantes por Etapa Educativa', fontweight='bold')
            ax4.set_xlabel('Etapa Educativa')
            ax4.set_ylabel('Número de Estudiantes')
            
            # Agregar etiquetas de valores
            for i, count in enumerate(stage_counts.values):
                ax4.text(i, count + stage_counts.max() * 0.01,
                        f'{count}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Guardar si se proporciona ruta
        if save_path:
            create_output_directory(Path(save_path).parent)
            plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
            logger.info(f"Gráficos de análisis demográfico guardados en {save_path}")
        
        logger.info("Gráficos de análisis demográfico creados exitosamente")
        return fig
        
    except Exception as e:
        logger.error(f"Error creando gráficos de análisis demográfico: {e}")
        return plt.figure()

def create_engagement_correlation_matrix(df: pd.DataFrame, save_path: Optional[str] = None) -> mfig.Figure:
    """
    Crear una matriz de correlación de calor para métricas de participación.
    
    Args:
        df: DataFrame conteniendo datos de estudiantes
        save_path: Ruta opcional para guardar el gráfico
        
    Returns:
        Objeto figura de Matplotlib
    """
    try:
        log_analysis_step("Creando matriz de correlación de métricas de participación")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Seleccionar columnas de participación
        engagement_cols = ['RaisedHands', 'VisitedResources', 'AnnouncementsView', 'Discussion']
        if 'TotalEngagement' in df.columns:
            engagement_cols.append('TotalEngagement')
        
        # Filtrar a las columnas existentes
        available_cols = [col for col in engagement_cols if col in df.columns]
        
        if len(available_cols) < 2:
            logger.warning("Métricas de participación insuficientes para análisis de correlación")
            return fig
        
        # Calcular matriz de correlación
        corr_matrix = df[available_cols].corr()
        
        # Crear mapa de calor
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.3f', 
                   cmap='coolwarm', center=0, square=True, ax=ax,
                   cbar_kws={'label': 'Coeficiente de Correlación'})
        
        ax.set_title('Matriz de Correlación de Métricas de Participación', fontsize=16, fontweight='bold', pad=20)
        
        # Rotar etiquetas para mejor legibilidad
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        
        # Guardar si se proporciona ruta
        if save_path:
            create_output_directory(Path(save_path).parent)
            plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
            logger.info(f"Matriz de correlación guardada en {save_path}")
        
        logger.info("Matriz de correlación creada exitosamente")
        return fig
        
    except Exception as e:
        logger.error(f"Error creando matriz de correlación: {e}")
        return plt.figure()

def create_regression_analysis_chart(df: pd.DataFrame, save_path: Optional[str] = None) -> mfig.Figure:
    """
    Crear un gráfico de regresión mostrando la relación entre calificación final y participación.
    
    Args:
        df: DataFrame conteniendo datos de estudiantes
        save_path: Ruta opcional para guardar el gráfico
        
    Returns:
        Objeto figura de Matplotlib
    """
    try:
        log_analysis_step("Creando gráfico de análisis de regresión")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Verificar que las columnas necesarias existan
        required_cols = ['Calificacion_Final', 'TotalEngagement']
        available_cols = [col for col in required_cols if col in df.columns]
        
        if len(available_cols) < 2:
            logger.warning("Columnas requeridas no encontradas para análisis de regresión")
            return fig
        
        # Gráfico 1: Regresión curvilínea entre calificación final y participación total
        if 'Calificacion_Final' in df.columns and 'TotalEngagement' in df.columns:
            # Preparar datos
            x = df['Calificacion_Final'].values
            y = df['TotalEngagement'].values
            
            # Crear scatter plot base
            ax1.scatter(x, y, alpha=0.6, s=50, color='blue', label='Datos observados')
            
            # Ordenar datos para las líneas de regresión
            sort_idx = np.argsort(x)
            x_sorted = x[sort_idx]
            y_sorted = y[sort_idx]
            
            # 1. Regresión lineal
            z_linear = np.polyfit(x, y, 1)
            p_linear = np.poly1d(z_linear)
            y_linear = p_linear(x_sorted)
            r2_linear = np.corrcoef(x, y)[0, 1]**2
            
            # 2. Regresión cuadrática
            z_quad = np.polyfit(x, y, 2)
            p_quad = np.poly1d(z_quad)
            y_quad = p_quad(x_sorted)
            r2_quad = 1 - np.sum((y - p_quad(x))**2) / np.sum((y - np.mean(y))**2)
            
            # 3. Regresión cúbica
            z_cubic = np.polyfit(x, y, 3)
            p_cubic = np.poly1d(z_cubic)
            y_cubic = p_cubic(x_sorted)
            r2_cubic = 1 - np.sum((y - p_cubic(x))**2) / np.sum((y - np.mean(y))**2)
            
            # 4. Regresión logarítmica (y = a + b*ln(x))
            # Asegurar que x sea positivo para log
            x_positive = x[x > 0]
            y_positive = y[x > 0]
            if len(x_positive) > 0:
                log_x = np.log(x_positive)
                z_log = np.polyfit(log_x, y_positive, 1)
                p_log = np.poly1d(z_log)
                y_log = p_log(np.log(x_sorted[x_sorted > 0]))
                r2_log = 1 - np.sum((y_positive - p_log(log_x))**2) / np.sum((y_positive - np.mean(y_positive))**2)
            else:
                r2_log = 0
            
            # 5. Regresión exponencial (y = a * exp(b*x))
            try:
                from scipy.optimize import curve_fit
                def exp_func(x, a, b):
                    return a * np.exp(b * x)
                
                popt_exp, _ = curve_fit(exp_func, x, y, maxfev=10000)
                y_exp = exp_func(x_sorted, *popt_exp)
                r2_exp = 1 - np.sum((y - exp_func(x, *popt_exp))**2) / np.sum((y - np.mean(y))**2)
            except:
                r2_exp = 0
            
            # Determinar la mejor regresión basada en R²
            r2_scores = {
                'Lineal': r2_linear,
                'Cuadrática': r2_quad,
                'Cúbica': r2_cubic,
                'Logarítmica': r2_log,
                'Exponencial': r2_exp
            }
            
            best_model = max(r2_scores, key=r2_scores.get)
            best_r2 = r2_scores[best_model]
            
            # Dibujar todas las líneas de regresión
            ax1.plot(x_sorted, y_linear, '--', color='red', alpha=0.7, 
                    label=f'Lineal (R²={r2_linear:.3f})')
            ax1.plot(x_sorted, y_quad, '--', color='green', alpha=0.7, 
                    label=f'Cuadrática (R²={r2_quad:.3f})')
            ax1.plot(x_sorted, y_cubic, '--', color='orange', alpha=0.7, 
                    label=f'Cúbica (R²={r2_cubic:.3f})')
            
            if r2_log > 0:
                ax1.plot(x_sorted[x_sorted > 0], y_log, '--', color='purple', alpha=0.7, 
                        label=f'Logarítmica (R²={r2_log:.3f})')
            
            if r2_exp > 0:
                ax1.plot(x_sorted, y_exp, '--', color='brown', alpha=0.7, 
                        label=f'Exponencial (R²={r2_exp:.3f})')
            
            # Resaltar la mejor regresión
            if best_model == 'Lineal':
                ax1.plot(x_sorted, y_linear, '-', color='red', linewidth=3, 
                        label=f'MEJOR: {best_model} (R²={best_r2:.3f})')
            elif best_model == 'Cuadrática':
                ax1.plot(x_sorted, y_quad, '-', color='green', linewidth=3, 
                        label=f'MEJOR: {best_model} (R²={best_r2:.3f})')
            elif best_model == 'Cúbica':
                ax1.plot(x_sorted, y_cubic, '-', color='orange', linewidth=3, 
                        label=f'MEJOR: {best_model} (R²={best_r2:.3f})')
            elif best_model == 'Logarítmica':
                ax1.plot(x_sorted[x_sorted > 0], y_log, '-', color='purple', linewidth=3, 
                        label=f'MEJOR: {best_model} (R²={best_r2:.3f})')
            elif best_model == 'Exponencial':
                ax1.plot(x_sorted, y_exp, '-', color='brown', linewidth=3, 
                        label=f'MEJOR: {best_model} (R²={best_r2:.3f})')
            
            ax1.set_title('Regresión Curvilínea: Calificación Final vs Participación Total', 
                         fontsize=14, fontweight='bold')
            ax1.set_xlabel('Calificación Final', fontsize=12)
            ax1.set_ylabel('Puntuación de Participación Total', fontsize=12)
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Calcular y mostrar correlación
            correlation = df['Calificacion_Final'].corr(df['TotalEngagement'])
            ax1.text(0.05, 0.95, f'Correlación: r = {correlation:.3f}\nMejor modelo: {best_model}', 
                    transform=ax1.transAxes, fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
        
        # Gráfico 2: Regresión entre asistencia y cumplimiento de actividades
        if 'Porcentaje_Asistencia' in df.columns and 'Cumplimiento_Actividades' in df.columns:
            # Preparar datos
            x = df['Porcentaje_Asistencia'].values
            y = df['Cumplimiento_Actividades'].values
            
            # Crear scatter plot base
            ax2.scatter(x, y, alpha=0.6, s=50, color='blue', label='Datos observados')
            
            # Ordenar datos para las líneas de regresión
            sort_idx = np.argsort(x)
            x_sorted = x[sort_idx]
            y_sorted = y[sort_idx]
            
            # 1. Regresión lineal
            z_linear = np.polyfit(x, y, 1)
            p_linear = np.poly1d(z_linear)
            y_linear = p_linear(x_sorted)
            r2_linear = np.corrcoef(x, y)[0, 1]**2
            
            # 2. Regresión cuadrática
            z_quad = np.polyfit(x, y, 2)
            p_quad = np.poly1d(z_quad)
            y_quad = p_quad(x_sorted)
            r2_quad = 1 - np.sum((y - p_quad(x))**2) / np.sum((y - np.mean(y))**2)
            
            # 3. Regresión cúbica
            z_cubic = np.polyfit(x, y, 3)
            p_cubic = np.poly1d(z_cubic)
            y_cubic = p_cubic(x_sorted)
            r2_cubic = 1 - np.sum((y - p_cubic(x))**2) / np.sum((y - np.mean(y))**2)
            
            # Determinar la mejor regresión
            r2_scores = {
                'Lineal': r2_linear,
                'Cuadrática': r2_quad,
                'Cúbica': r2_cubic
            }
            
            best_model = max(r2_scores, key=r2_scores.get)
            best_r2 = r2_scores[best_model]
            
            # Dibujar líneas de regresión
            ax2.plot(x_sorted, y_linear, '--', color='red', alpha=0.7, 
                    label=f'Lineal (R²={r2_linear:.3f})')
            ax2.plot(x_sorted, y_quad, '--', color='green', alpha=0.7, 
                    label=f'Cuadrática (R²={r2_quad:.3f})')
            ax2.plot(x_sorted, y_cubic, '--', color='orange', alpha=0.7, 
                    label=f'Cúbica (R²={r2_cubic:.3f})')
            
            # Resaltar la mejor regresión
            if best_model == 'Lineal':
                ax2.plot(x_sorted, y_linear, '-', color='red', linewidth=3, 
                        label=f'MEJOR: {best_model} (R²={best_r2:.3f})')
            elif best_model == 'Cuadrática':
                ax2.plot(x_sorted, y_quad, '-', color='green', linewidth=3, 
                        label=f'MEJOR: {best_model} (R²={best_r2:.3f})')
            elif best_model == 'Cúbica':
                ax2.plot(x_sorted, y_cubic, '-', color='orange', linewidth=3, 
                        label=f'MEJOR: {best_model} (R²={best_r2:.3f})')
            
            ax2.set_title('Regresión: Asistencia vs Cumplimiento de Actividades', 
                         fontsize=14, fontweight='bold')
            ax2.set_xlabel('Porcentaje de Asistencia (%)', fontsize=12)
            ax2.set_ylabel('Cumplimiento de Actividades (%)', fontsize=12)
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Calcular y mostrar correlación
            correlation = df['Porcentaje_Asistencia'].corr(df['Cumplimiento_Actividades'])
            ax2.text(0.05, 0.95, f'Correlación: r = {correlation:.3f}\nMejor modelo: {best_model}', 
                    transform=ax2.transAxes, fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
        
        plt.tight_layout()
        
        # Guardar si se proporciona ruta
        if save_path:
            create_output_directory(Path(save_path).parent)
            plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
            logger.info(f"Gráfico de análisis de regresión guardado en {save_path}")
        
        logger.info("Gráfico de análisis de regresión creado exitosamente")
        return fig
        
    except Exception as e:
        logger.error(f"Error creando gráfico de análisis de regresión: {e}")
        return plt.figure()

def create_all_visualizations(df: pd.DataFrame, stats: Dict[str, Any], 
                            groups: Dict[str, Any], save_charts: bool = True) -> Dict[str, mfig.Figure]:
    """
    Crear todos los gráficos de visualización para el análisis académico.
    
    Args:
        df: DataFrame con datos de estudiantes limpios
        stats: Resultados del análisis estadístico
        groups: Resultados del análisis de agrupación
        save_charts: Si guardar gráficos en archivos
        
    Returns:
        Diccionario conteniendo todas las figuras creadas
    """
    try:
        log_analysis_step("Creando todos los gráficos de visualización")
        
        # Configurar estilo de gráficos
        setup_chart_style()
        
        charts = {}
        
        # Definir funciones de creación de gráficos y sus rutas de guardado
        chart_configs = [
            ('distribución_calificaciones', create_grade_distribution_chart, 'distribución_calificaciones.png'),
            ('comparación_materias', create_subject_comparison_bar, 'comparación_materias.png'),
            ('tendencias_semestrales', create_semester_trend_line, 'tendencias_semestrales.png'),
            ('dispersión_rendimiento', create_performance_scatter, 'dispersión_rendimiento.png'),
            ('mapa_calor_asistencia', create_attendance_heatmap, 'mapa_calor_asistencia.png'),
            ('análisis_demográfico', create_demographic_analysis_charts, 'análisis_demográfico.png'),
            ('matriz_correlación', create_engagement_correlation_matrix, 'matriz_correlación.png'),
            ('análisis_regresión', create_regression_analysis_chart, 'análisis_regresión.png')
        ]
        
        # Crear cada gráfico
        for chart_name, chart_function, filename in chart_configs:
            try:
                save_path = str(CHARTS_DIR / filename) if save_charts else None
                charts[chart_name] = chart_function(df, save_path)
                logger.info(f"Gráfico {chart_name} creado")
            except Exception as e:
                logger.error(f"Error creando gráfico {chart_name}: {e}")
                charts[chart_name] = plt.figure()  # Figura vacía como fallback
        
        logger.info(f"Todos los gráficos de visualización creados exitosamente: {len(charts)} gráficos")
        return charts
        
    except Exception as e:
        logger.error(f"Error creando todos los gráficos de visualización: {e}")
        return {}

if __name__ == "__main__":
    # Probar el módulo de visualización
    try:
        import sys
        import os
        
        # Agregar raíz del proyecto al path  
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        # Importar módulos - ignorar advertencias de análisis estático, funciona en tiempo de ejecución
        from src.data.data_loader import load_and_validate_data  # type: ignore
        from src.data.data_cleaner import clean_student_data  # type: ignore
        from src.analysis.statistics import calculate_all_statistics  # type: ignore
        from src.analysis.grouping import perform_all_groupings  # type: ignore
        
        # Cargar y preparar datos
        df = load_and_validate_data()
        df_clean = clean_student_data(df)
        stats = calculate_all_statistics(df_clean)
        groups = perform_all_groupings(df_clean)
        
        print("Datos preparados exitosamente!")
        
        # Crear visualizaciones
        charts = create_all_visualizations(df_clean, stats, groups, save_charts=True)
        print(f"Se crearon {len(charts)} gráficos de visualización")
        
        # Mostrar los gráficos
        print("Gráficos creados:")
        for chart_name in charts.keys():
            print(f"  - {chart_name}")
        
        # Mostrar un gráfico como ejemplo
        plt.show()
        
    except Exception as e:
        print(f"Error probando módulo de visualización: {e}")
