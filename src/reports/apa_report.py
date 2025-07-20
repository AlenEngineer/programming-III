"""
Generador de reportes estilo APA para el Sistema de Análisis de Datos Académicos.
Crea reportes académicos profesionales con gráficos integrados y análisis estadístico.
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import black, blue, red
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib import colors
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging
from datetime import datetime
import sys
import os

# Agregar el directorio padre al path para importar config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import REPORTS_DIR, CHARTS_DIR
from src.utils.helpers import log_analysis_step, create_output_directory

logger = logging.getLogger(__name__)

class APAReportGenerator:
    """Generar reportes académicos estilo APA con visualizaciones integradas y análisis."""
    
    def __init__(self, title: str = "Reporte de Análisis de Datos Académicos", 
                 author: str = "Sistema de Análisis de Datos Académicos",
                 institution: str = "Institución Educativa"):
        """
        Inicializar el generador de reportes APA.
        
        Args:
            title: Título del reporte
            author: Autor del reporte
            institution: Nombre de la institución
        """
        self.title = title
        self.author = author
        self.institution = institution
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
    def _setup_custom_styles(self):
        """Configurar estilos personalizados compatibles con APA."""
        # Estilo de Título APA
        self.styles.add(ParagraphStyle(
            name='APATitle',
            parent=self.styles['Title'],
            fontSize=16,
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        # Estilo de Autor APA
        self.styles.add(ParagraphStyle(
            name='APAAuthor',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=6,
            alignment=TA_CENTER,
            fontName='Helvetica'
        ))
        
        # Encabezado APA 1
        self.styles.add(ParagraphStyle(
            name='APAHeading1',
            parent=self.styles['Heading1'],
            fontSize=14,
            spaceAfter=12,
            spaceBefore=24,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        # Encabezado APA 2
        self.styles.add(ParagraphStyle(
            name='APAHeading2',
            parent=self.styles['Heading2'],
            fontSize=12,
            spaceAfter=6,
            spaceBefore=18,
            alignment=TA_LEFT,
            fontName='Helvetica-Bold'
        ))
        
        # Texto Principal APA
        self.styles.add(ParagraphStyle(
            name='APABody',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=12,
            alignment=TA_JUSTIFY,
            fontName='Helvetica',
            leading=14
        ))
        
        # Resumen APA
        self.styles.add(ParagraphStyle(
            name='APAAbstract',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=12,
            alignment=TA_JUSTIFY,
            fontName='Helvetica',
            leftIndent=0.5*inch,
            rightIndent=0.5*inch
        ))

    def generate_comprehensive_report(self, df: pd.DataFrame, stats: Dict[str, Any], 
                                    groups: Dict[str, Any], risk_report: Dict[str, Any],
                                    charts: Dict[str, Any]) -> str:
        """
        Generar un reporte integral estilo APA.
        
        Args:
            df: DataFrame limpio con datos de estudiantes
            stats: Resultados del análisis estadístico
            groups: Resultados del análisis de agrupación
            risk_report: Resultados del análisis de riesgos
            charts: Gráficos generados
            
        Returns:
            Ruta al reporte PDF generado
        """
        try:
            log_analysis_step("Generando reporte integral estilo APA")
            
            # Crear directorio de salida
            create_output_directory(REPORTS_DIR)
            
            # Generar nombre de archivo con timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = REPORTS_DIR / f"reporte_analisis_academico_{timestamp}.pdf"
            
            # Crear documento PDF
            doc = SimpleDocTemplate(
                str(report_path),
                pagesize=letter,
                rightMargin=1*inch,
                leftMargin=1*inch,
                topMargin=1*inch,
                bottomMargin=1*inch
            )
            
            # Construir contenido del reporte
            story = []
            
            # Página de título
            story.extend(self._create_title_page())
            story.append(PageBreak())
            
            # Resumen
            story.extend(self._create_abstract(df, stats, risk_report))
            story.append(PageBreak())
            
            # Introducción
            story.extend(self._create_introduction())
            
            # Métodos
            story.extend(self._create_methods_section(df))
            
            # Resultados
            story.extend(self._create_results_section(df, stats, groups, risk_report, charts))
            
            # Discusión
            story.extend(self._create_discussion_section(risk_report))
            
            # Conclusión
            story.extend(self._create_conclusion_section(risk_report))
            
            # Referencias
            story.extend(self._create_references_section())
            
            # Construir PDF
            doc.build(story)
            
            logger.info(f"Reporte APA generado exitosamente: {report_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Error generando reporte APA: {e}")
            return ""

    def _create_title_page(self) -> List:
        """Crear página de título estilo APA."""
        story = []
        
        # Título
        story.append(Spacer(1, 2*inch))
        story.append(Paragraph(self.title, self.styles['APATitle']))
        story.append(Spacer(1, 0.5*inch))
        
        # Autor
        story.append(Paragraph(self.author, self.styles['APAAuthor']))
        story.append(Spacer(1, 0.25*inch))
        
        # Institución
        story.append(Paragraph(self.institution, self.styles['APAAuthor']))
        story.append(Spacer(1, 0.25*inch))
        
        # Fecha
        current_date = datetime.now().strftime("%d de %B de %Y")
        story.append(Paragraph(f"Fecha: {current_date}", self.styles['APAAuthor']))
        
        return story

    def _create_abstract(self, df: pd.DataFrame, stats: Dict[str, Any], 
                        risk_report: Dict[str, Any]) -> List:
        """Crear resumen ejecutivo estilo APA."""
        story = []
        
        story.append(Paragraph("RESUMEN", self.styles['APAHeading1']))
        story.append(Spacer(1, 12))
        
        # Calcular estadísticas clave
        total_students = len(df)
        avg_grade = df['Calificacion_Final'].mean() if 'Calificacion_Final' in df.columns else 0
        avg_attendance = df['Porcentaje_Asistencia'].mean() if 'Porcentaje_Asistencia' in df.columns else 0
        avg_compliance = df['Cumplimiento_Actividades'].mean() if 'Cumplimiento_Actividades' in df.columns else 0
        
        # Contar estudiantes en riesgo
        at_risk_count = risk_report.get('total_at_risk_students', 0)
        risk_percentage = (at_risk_count / total_students * 100) if total_students > 0 else 0
        
        abstract_text = f"""
        Este estudio presenta un análisis integral del rendimiento académico de {total_students} estudiantes 
        de la Universidad Tecnológica de Panamá, utilizando datos de calificaciones finales, porcentaje de 
        asistencia y cumplimiento de actividades. Los resultados muestran un promedio de calificación de 
        {avg_grade:.1f} puntos, con una tasa de asistencia promedio del {avg_attendance:.1f}% y un cumplimiento 
        de actividades del {avg_compliance:.1f}%. Se identificaron {at_risk_count} estudiantes ({risk_percentage:.1f}%) 
        en situación de riesgo académico, definido por calificaciones bajas, asistencia deficiente o bajo 
        cumplimiento de actividades. El análisis incluye comparaciones por materia, semestre y carrera, 
        proporcionando insights valiosos para la toma de decisiones académicas y la implementación de 
        estrategias de intervención temprana.
        """
        
        story.append(Paragraph(abstract_text, self.styles['APAAbstract']))
        story.append(Spacer(1, 12))
        
        return story

    def _create_introduction(self) -> List:
        """Crear sección de introducción."""
        story = []
        
        story.append(Paragraph("INTRODUCCIÓN", self.styles['APAHeading1']))
        story.append(Spacer(1, 12))
        
        intro_text = """
        El análisis del rendimiento académico estudiantil es fundamental para el desarrollo de 
        estrategias educativas efectivas y la identificación temprana de estudiantes en riesgo. 
        La Universidad Tecnológica de Panamá, comprometida con la excelencia académica, requiere 
        herramientas analíticas robustas para evaluar el progreso de sus estudiantes y optimizar 
        los procesos de enseñanza-aprendizaje.
        
        Este estudio utiliza técnicas de análisis de datos para examinar múltiples dimensiones 
        del rendimiento académico, incluyendo calificaciones finales, patrones de asistencia y 
        cumplimiento de actividades académicas. La integración de estas métricas permite una 
        evaluación holística del desempeño estudiantil y la identificación de factores que 
        contribuyen al éxito o fracaso académico.
        
        Los objetivos específicos de este análisis incluyen: (1) evaluar el rendimiento general 
        de los estudiantes por materia y semestre, (2) identificar estudiantes en riesgo académico 
        basándose en múltiples indicadores, (3) analizar patrones de asistencia y participación, 
        y (4) proporcionar recomendaciones para intervenciones académicas efectivas.
        """
        
        story.append(Paragraph(intro_text, self.styles['APABody']))
        story.append(Spacer(1, 12))
        
        return story

    def _create_methods_section(self, df: pd.DataFrame) -> List:
        """Crear sección de métodos."""
        story = []
        
        story.append(Paragraph("MÉTODO", self.styles['APAHeading1']))
        story.append(Spacer(1, 12))
        
        # Información del dataset
        total_students = len(df)
        total_careers = df['Carrera'].nunique() if 'Carrera' in df.columns else 0
        total_subjects = df['Materia'].nunique() if 'Materia' in df.columns else 0
        total_semesters = df['Semestre'].nunique() if 'Semestre' in df.columns else 0
        
        methods_text = f"""
        <b>Participantes</b><br/>
        El estudio incluyó {total_students} estudiantes de la Universidad Tecnológica de Panamá, 
        distribuidos en {total_careers} carreras diferentes, {total_subjects} materias y {total_semesters} 
        semestres académicos. Los datos fueron recolectados durante el período académico actual 
        y representan una muestra diversa de la población estudiantil.
        
        <b>Variables de Medida</b><br/>
        Se analizaron tres variables principales de rendimiento académico:
        • <i>Calificación Final:</i> Puntuación numérica obtenida por el estudiante en cada materia (escala 0-100)
        • <i>Porcentaje de Asistencia:</i> Proporción de clases asistidas durante el semestre (escala 0-100%)
        • <i>Cumplimiento de Actividades:</i> Porcentaje de actividades académicas completadas (escala 0-100%)
        
        <b>Procedimiento de Análisis</b><br/>
        Los datos fueron procesados utilizando técnicas de limpieza y validación para asegurar 
        la calidad de la información. Se aplicaron análisis estadísticos descriptivos, análisis 
        de correlación y técnicas de agrupación para identificar patrones en el rendimiento 
        académico. Los estudiantes en riesgo fueron identificados utilizando umbrales específicos: 
        calificaciones menores a 60 puntos, asistencia menor al 75% y cumplimiento de actividades 
        menor al 70%.
        """
        
        story.append(Paragraph(methods_text, self.styles['APABody']))
        story.append(Spacer(1, 12))
        
        return story

    def _create_results_section(self, df: pd.DataFrame, stats: Dict[str, Any], 
                              groups: Dict[str, Any], risk_report: Dict[str, Any],
                              charts: Dict[str, Any]) -> List:
        """Crear sección de resultados."""
        story = []
        
        story.append(Paragraph("RESULTADOS", self.styles['APAHeading1']))
        story.append(Spacer(1, 12))
        
        # Estadísticas descriptivas generales
        story.append(Paragraph("Estadísticas Descriptivas Generales", self.styles['APAHeading2']))
        story.append(Spacer(1, 6))
        
        # Calcular estadísticas clave
        total_students = len(df)
        avg_grade = df['Calificacion_Final'].mean() if 'Calificacion_Final' in df.columns else 0
        std_grade = df['Calificacion_Final'].std() if 'Calificacion_Final' in df.columns else 0
        avg_attendance = df['Porcentaje_Asistencia'].mean() if 'Porcentaje_Asistencia' in df.columns else 0
        avg_compliance = df['Cumplimiento_Actividades'].mean() if 'Cumplimiento_Actividades' in df.columns else 0
        
        # Distribución de calificaciones
        high_performance = len(df[df['Calificacion_Final'] >= 85]) if 'Calificacion_Final' in df.columns else 0
        medium_performance = len(df[(df['Calificacion_Final'] >= 60) & (df['Calificacion_Final'] < 85)]) if 'Calificacion_Final' in df.columns else 0
        low_performance = len(df[df['Calificacion_Final'] < 60]) if 'Calificacion_Final' in df.columns else 0
        
        results_text = f"""
        <b>Rendimiento Académico General</b><br/>
        El análisis de {total_students} estudiantes reveló un promedio de calificación final de 
        {avg_grade:.1f} puntos (DE = {std_grade:.1f}). La distribución de rendimiento mostró 
        {high_performance} estudiantes ({high_performance/total_students*100:.1f}%) con rendimiento alto 
        (≥85 puntos), {medium_performance} estudiantes ({medium_performance/total_students*100:.1f}%) 
        con rendimiento medio (60-84 puntos), y {low_performance} estudiantes ({low_performance/total_students*100:.1f}%) 
        con rendimiento bajo (<60 puntos).
        
        <b>Patrones de Asistencia y Participación</b><br/>
        El porcentaje promedio de asistencia fue del {avg_attendance:.1f}%, indicando una buena 
        participación general en las actividades presenciales. El cumplimiento de actividades 
        académicas alcanzó un promedio del {avg_compliance:.1f}%, demostrando un compromiso 
        moderado con las tareas asignadas.
        """
        
        story.append(Paragraph(results_text, self.styles['APABody']))
        story.append(Spacer(1, 12))
        
        # Análisis por materia
        if 'Materia' in df.columns and 'Calificacion_Final' in df.columns:
            story.append(Paragraph("Análisis por Materia", self.styles['APAHeading2']))
            story.append(Spacer(1, 6))
            
            subject_stats = df.groupby('Materia', observed=True)['Calificacion_Final'].agg(['mean', 'count']).round(2)
            subject_stats = subject_stats.sort_values('mean', ascending=False)
            
            # Top 5 materias con mejor rendimiento
            top_subjects = subject_stats.head(5)
            top_subjects_text = "Las materias con mejor rendimiento promedio fueron: "
            for i, (subject, row) in enumerate(top_subjects.iterrows()):
                if i > 0:
                    top_subjects_text += ", "
                top_subjects_text += f"{subject} ({row['mean']:.1f} puntos, n={int(row['count'])})"
            top_subjects_text += "."
            
            story.append(Paragraph(top_subjects_text, self.styles['APABody']))
            story.append(Spacer(1, 12))
        
        # Análisis por semestre
        if 'Semestre' in df.columns and 'Calificacion_Final' in df.columns:
            story.append(Paragraph("Análisis por Semestre", self.styles['APAHeading2']))
            story.append(Spacer(1, 6))
            
            semester_stats = df.groupby('Semestre', observed=True)['Calificacion_Final'].agg(['mean', 'count']).round(2)
            semester_text = "El rendimiento por semestre mostró las siguientes tendencias: "
            for semester, row in semester_stats.iterrows():
                semester_text += f"Semestre {semester}: {row['mean']:.1f} puntos (n={int(row['count'])}), "
            semester_text = semester_text.rstrip(", ") + "."
            
            story.append(Paragraph(semester_text, self.styles['APABody']))
            story.append(Spacer(1, 12))
        
        # Análisis de riesgo
        story.append(Paragraph("Identificación de Estudiantes en Riesgo", self.styles['APAHeading2']))
        story.append(Spacer(1, 6))
        
        at_risk_count = risk_report.get('total_at_risk_students', 0)
        risk_percentage = (at_risk_count / total_students * 100) if total_students > 0 else 0
        
        risk_text = f"""
        Se identificaron {at_risk_count} estudiantes ({risk_percentage:.1f}%) en situación de riesgo 
        académico, definido por calificaciones bajas, asistencia deficiente o bajo cumplimiento 
        de actividades. Estos estudiantes requieren intervención académica inmediata para 
        mejorar su rendimiento y reducir el riesgo de fracaso académico.
        """
        
        story.append(Paragraph(risk_text, self.styles['APABody']))
        story.append(Spacer(1, 12))
        
        # Correlaciones
        if 'Calificacion_Final' in df.columns and 'Porcentaje_Asistencia' in df.columns and 'Cumplimiento_Actividades' in df.columns:
            story.append(Paragraph("Análisis de Correlaciones", self.styles['APAHeading2']))
            story.append(Spacer(1, 6))
            
            corr_grade_attendance = df['Calificacion_Final'].corr(df['Porcentaje_Asistencia'])
            corr_grade_compliance = df['Calificacion_Final'].corr(df['Cumplimiento_Actividades'])
            corr_attendance_compliance = df['Porcentaje_Asistencia'].corr(df['Cumplimiento_Actividades'])
            
            correlation_text = f"""
            El análisis de correlación reveló relaciones significativas entre las variables de 
            rendimiento: la correlación entre calificación final y asistencia fue r = {corr_grade_attendance:.3f}, 
            entre calificación final y cumplimiento de actividades fue r = {corr_grade_compliance:.3f}, 
            y entre asistencia y cumplimiento de actividades fue r = {corr_attendance_compliance:.3f}. 
            Estas correlaciones sugieren que la asistencia y el cumplimiento de actividades están 
            positivamente relacionados con el rendimiento académico.
            """
            
            story.append(Paragraph(correlation_text, self.styles['APABody']))
            story.append(Spacer(1, 12))
        
        # Agregar visualizaciones
        story.extend(self._create_visualizations_section(charts))
        
        return story

    def _create_visualizations_section(self, charts: Dict[str, Any]) -> List:
        """Crear sección de visualizaciones con las gráficas más importantes."""
        story = []
        
        story.append(Paragraph("Visualizaciones del Análisis", self.styles['APAHeading2']))
        story.append(Spacer(1, 6))
        
        intro_text = """
        Las siguientes visualizaciones proporcionan una representación gráfica de los hallazgos 
        principales del análisis, facilitando la comprensión de los patrones y tendencias 
        identificados en el rendimiento académico de los estudiantes.
        """
        story.append(Paragraph(intro_text, self.styles['APABody']))
        story.append(Spacer(1, 12))
        
        # Lista de gráficas a incluir en orden de importancia
        chart_files = [
            ("distribución_calificaciones.png", "Figura 1. Distribución de Calificaciones de Rendimiento"),
            ("comparación_materias.png", "Figura 2. Promedio de Participación por Materia"),
            ("tendencias_semestrales.png", "Figura 3. Análisis de Rendimiento por Semestre"),
            ("matriz_correlación.png", "Figura 4. Matriz de Correlación entre Variables"),
            ("análisis_regresión.png", "Figura 5. Análisis de Regresión: Relaciones entre Variables")
        ]
        
        for i, (filename, title) in enumerate(chart_files, 1):
            chart_path = CHARTS_DIR / filename
            
            if chart_path.exists():
                # Título de la figura
                story.append(Paragraph(title, self.styles['APAHeading2']))
                story.append(Spacer(1, 6))
                
                # Agregar la imagen
                img = Image(str(chart_path), width=6*inch, height=4*inch)
                img.hAlign = 'CENTER'
                story.append(img)
                story.append(Spacer(1, 12))
                
                # Espacio para análisis (líneas en blanco)
                story.append(Spacer(1, 24))
                
                logger.info(f"Gráfica {i} incluida en el reporte: {filename}")
            else:
                logger.warning(f"Archivo de gráfica no encontrado: {chart_path}")
        
        return story

    def _create_discussion_section(self, risk_report: Dict[str, Any]) -> List:
        """Crear sección de discusión."""
        story = []
        
        story.append(Paragraph("DISCUSIÓN", self.styles['APAHeading1']))
        story.append(Spacer(1, 12))
        
        at_risk_count = risk_report.get('total_at_risk_students', 0)
        
        discussion_text = f"""
        Los resultados de este análisis proporcionan insights valiosos sobre el rendimiento 
        académico de los estudiantes de la Universidad Tecnológica de Panamá. La identificación 
        de {at_risk_count} estudiantes en situación de riesgo académico destaca la importancia 
        de implementar sistemas de alerta temprana y estrategias de intervención proactivas.
        
        <b>Implicaciones para la Práctica Educativa</b><br/>
        Los hallazgos sugieren que la asistencia y el cumplimiento de actividades académicas 
        son predictores importantes del rendimiento académico. Esto refuerza la necesidad de 
        monitorear no solo las calificaciones finales, sino también los patrones de participación 
        y compromiso de los estudiantes a lo largo del semestre.
        
        <b>Estrategias de Intervención</b><br/>
        Para los estudiantes identificados en riesgo, se recomienda implementar programas de 
        tutoría académica, sesiones de refuerzo y seguimiento individualizado. La correlación 
        positiva entre asistencia y rendimiento sugiere que las estrategias que promueven la 
        participación regular en clases pueden tener un impacto significativo en el éxito académico.
        
        <b>Limitaciones del Estudio</b><br/>
        Es importante reconocer que este análisis se basa en datos de un período académico 
        específico y puede no reflejar tendencias a largo plazo. Además, el tamaño de la muestra 
        limita la generalización de los resultados a toda la población estudiantil de la universidad.
        """
        
        story.append(Paragraph(discussion_text, self.styles['APABody']))
        story.append(Spacer(1, 12))
        
        return story

    def _create_conclusion_section(self, risk_report: Dict[str, Any]) -> List:
        """Crear sección de conclusión."""
        story = []
        
        story.append(Paragraph("CONCLUSIÓN", self.styles['APAHeading1']))
        story.append(Spacer(1, 12))
        
        at_risk_count = risk_report.get('total_at_risk_students', 0)
        
        conclusion_text = f"""
        Este estudio demuestra la utilidad del análisis de datos académicos para identificar 
        estudiantes en riesgo y optimizar las estrategias educativas. La identificación de 
        {at_risk_count} estudiantes que requieren intervención inmediata subraya la importancia 
        de sistemas de monitoreo continuo del rendimiento académico.
        
        Los resultados apoyan la implementación de programas de intervención temprana que 
        aborden tanto los aspectos académicos como los patrones de participación de los 
        estudiantes. La Universidad Tecnológica de Panamá puede utilizar estos hallazgos 
        para desarrollar políticas educativas más efectivas y mejorar las tasas de retención 
        y éxito académico.
        
        Se recomienda continuar con análisis similares en períodos académicos futuros para 
        evaluar la efectividad de las intervenciones implementadas y ajustar las estrategias 
        según sea necesario.
        """
        
        story.append(Paragraph(conclusion_text, self.styles['APABody']))
        story.append(Spacer(1, 12))
        
        return story

    def _create_references_section(self) -> List:
        """Crear sección de referencias."""
        story = []
        
        story.append(Paragraph("Referencias", self.styles['APAHeading1']))
        story.append(Spacer(1, 12))
        
        references = [
            "American Psychological Association. (2020). Publication manual of the American Psychological Association (7th ed.).",
            "McKinney, W. (2017). Python for data analysis: Data wrangling with Pandas, NumPy, and IPython. O'Reilly Media.",
            "Wickham, H., & Grolemund, G. (2016). R for data science: Import, tidy, transform, visualize, and model data. O'Reilly Media.",
            "Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825-2830."
        ]
        
        for ref in references:
            story.append(Paragraph(ref, self.styles['APABody']))
            story.append(Spacer(1, 6))
        
        return story

def generate_apa_report(df: pd.DataFrame, stats: Dict[str, Any], 
                       groups: Dict[str, Any], risk_report: Dict[str, Any],
                       charts: Dict[str, Any]) -> str:
    """
    Función de conveniencia para generar reporte APA.
    
    Args:
        df: DataFrame con datos de estudiantes
        stats: Resultados del análisis estadístico
        groups: Resultados del análisis de agrupación
        risk_report: Resultados del análisis de riesgos
        charts: Gráficos generados
        
    Returns:
        Ruta al reporte PDF generado
    """
    try:
        generator = APAReportGenerator()
        return generator.generate_comprehensive_report(df, stats, groups, risk_report, charts)
    except Exception as e:
        logger.error(f"Error en función de conveniencia para reporte APA: {e}")
        return ""

if __name__ == "__main__":
    # Probar el generador de reportes
    try:
        print("Probando generador de reportes APA...")
        
        # Crear datos de ejemplo
        df = pd.DataFrame({
            'StudentID': range(1, 101),
            'Class': ['L', 'M', 'H'] * 33 + ['M'],
            'TotalEngagement': np.random.normal(50, 15, 100)
        })
        
        stats = {'overall_stats': {'mean': 50.5, 'std': 15.2}}
        groups = {'demographic': {}}
        risk_report = {'at_risk_students': [1, 2, 3]}
        charts = {}
        
        # Generar reporte
        report_path = generate_apa_report(df, stats, groups, risk_report, charts)
        print(f"Reporte generado: {report_path}")
        
    except Exception as e:
        print(f"Error probando generador de reportes: {e}")
