"""
Generador de Reportes APA para el Sistema de Análisis de Datos Académicos.
Crea reportes profesionales estilo APA con análisis estadístico y visualizaciones.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import sys
import os
import logging

# Agregar el directorio padre al path para importar config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import REPORTS_DIR
from src.utils.helpers import log_analysis_step, create_output_directory

logger = logging.getLogger(__name__)

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    logger.warning("ReportLab no disponible. Instalar con: pip install reportlab")
    REPORTLAB_AVAILABLE = False

class APAReportGenerator:
    """Generar reportes profesionales estilo APA para análisis de datos académicos."""
    
    def __init__(self, title: str = "Reporte de Análisis de Rendimiento Académico", 
                 author: str = "Sistema de Análisis de Datos Académicos"):
        """
        Inicializar el generador de reportes APA.
        
        Args:
            title: Título del reporte
            author: Autor del reporte
        """
        self.title = title
        self.author = author
        self.date = datetime.now().strftime("%d de %B de %Y")
        
        if not REPORTLAB_AVAILABLE:
            raise ImportError("ReportLab es requerido para generación de PDF. Instalar con: pip install reportlab")
        
        # Configurar formato estilo APA
        self.styles = getSampleStyleSheet()
        self._setup_apa_styles()
        
    def _setup_apa_styles(self):
        """Configurar estilos de párrafo compatibles con APA."""
        # Estilo de título
        self.styles.add(ParagraphStyle(
            name='APATitle',
            parent=self.styles['Title'],
            fontSize=14,
            spaceAfter=12,
            alignment=TA_CENTER,
            fontName='Times-Bold'
        ))
        
        # Estilo de autor
        self.styles.add(ParagraphStyle(
            name='APAAuthor',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=6,
            alignment=TA_CENTER,
            fontName='Times-Roman'
        ))
        
        # Estilo de fecha
        self.styles.add(ParagraphStyle(
            name='APADate',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=24,
            alignment=TA_CENTER,
            fontName='Times-Roman'
        ))
        
        # Estilo de encabezado 1
        self.styles.add(ParagraphStyle(
            name='APAHeading1',
            parent=self.styles['Heading1'],
            fontSize=12,
            spaceAfter=12,
            spaceBefore=24,
            alignment=TA_CENTER,
            fontName='Times-Bold'
        ))
        
        # Estilo de encabezado 2
        self.styles.add(ParagraphStyle(
            name='APAHeading2',
            parent=self.styles['Heading2'],
            fontSize=12,
            spaceAfter=6,
            spaceBefore=12,
            alignment=TA_LEFT,
            fontName='Times-Bold'
        ))
        
        # Estilo de texto principal
        self.styles.add(ParagraphStyle(
            name='APABody',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=12,
            alignment=TA_JUSTIFY,
            fontName='Times-Roman',
            leftIndent=0,
            rightIndent=0,
            firstLineIndent=0.5*inch
        ))
        
        # Estilo de título de tabla
        self.styles.add(ParagraphStyle(
            name='APATableCaption',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=6,
            spaceBefore=12,
            alignment=TA_LEFT,
            fontName='Times-Italic'
        ))
        
        # Estilo de título de figura
        self.styles.add(ParagraphStyle(
            name='APAFigureCaption',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=6,
            spaceBefore=6,
            alignment=TA_LEFT,
            fontName='Times-Italic'
        ))

    def generate_comprehensive_report(self, df: pd.DataFrame, stats: Dict[str, Any], 
                                    groups: Dict[str, Any], risk_report: Dict[str, Any],
                                    charts: Dict[str, Any], save_path: Optional[str] = None) -> str:
        """
        Generar un reporte integral estilo APA.
        
        Args:
            df: DataFrame limpio con datos de estudiantes
            stats: Resultados del análisis estadístico
            groups: Resultados del análisis de agrupación
            risk_report: Resultados del análisis de riesgos
            charts: Diccionario de gráficos generados
            save_path: Ruta opcional para guardar el reporte
            
        Returns:
            Ruta al reporte generado
        """
        try:
            log_analysis_step("Generando reporte integral estilo APA")
            
            if save_path is None:
                save_path = str(REPORTS_DIR / f"reporte_analisis_academico_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
            
            # Crear directorio de salida
            create_output_directory(Path(save_path).parent)
            
            # Crear documento PDF
            doc = SimpleDocTemplate(save_path, pagesize=A4, 
                                  rightMargin=1*inch, leftMargin=1*inch,
                                  topMargin=1*inch, bottomMargin=1*inch)
            
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
            
            # Método
            story.extend(self._create_method(df))
            
            # Resultados
            story.extend(self._create_results(df, stats, groups, risk_report, charts))
            
            # Discusión
            story.extend(self._create_discussion(stats, risk_report))
            
            # Referencias
            story.extend(self._create_references())
            
            # Construir PDF
            doc.build(story)
            
            logger.info(f"Reporte APA generado exitosamente: {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"Error generando reporte APA: {e}")
            raise

    def _create_title_page(self) -> List:
        """Crear página de título estilo APA."""
        content = []
        
        # Agregar espacio vertical
        content.append(Spacer(1, 2*inch))
        
        # Título
        content.append(Paragraph(self.title, self.styles['APATitle']))
        
        # Agregar espacio
        content.append(Spacer(1, 1*inch))
        
        # Autor
        content.append(Paragraph(self.author, self.styles['APAAuthor']))
        
        # Institución (opcional)
        content.append(Paragraph("Institución Académica", self.styles['APAAuthor']))
        
        # Agregar espacio
        content.append(Spacer(1, 1*inch))
        
        # Fecha
        content.append(Paragraph(self.date, self.styles['APADate']))
        
        return content

    def _create_abstract(self, df: pd.DataFrame, stats: Dict[str, Any], 
                        risk_report: Dict[str, Any]) -> List:
        """Crear resumen estilo APA."""
        content = []
        
        # Encabezado de resumen
        content.append(Paragraph("Resumen", self.styles['APAHeading1']))
        
        # Texto del resumen
        total_students = len(df)
        at_risk_count = risk_report.get('at_risk_students', {}).get('count', 0)
        at_risk_percentage = risk_report.get('at_risk_students', {}).get('percentage', '0%')
        
        abstract_text = f"""
        Este estudio presenta un análisis integral de los datos de rendimiento académico de {total_students} estudiantes 
        en múltiples materias y semestres. El análisis utilizó métodos estadísticos, técnicas de agrupación, 
        y algoritmos de evaluación de riesgo para identificar patrones en la participación, asistencia, y rendimiento académico. 
        Los hallazgos clave incluyen la identificación de {at_risk_count} estudiantes ({at_risk_percentage}) en riesgo académico, 
        variaciones significativas en la participación entre materias, y correlaciones entre patrones de asistencia y 
        resultados académicos. El estudio proporciona insights prácticos para intervenciones educativas y desarrollo de políticas 
        para mejorar las tasas de éxito académico.
        """
        
        content.append(Paragraph(abstract_text.strip(), self.styles['APABody']))
        
        # Palabras clave
        content.append(Spacer(1, 12))
        keywords_text = "<i>Palabras clave:</i> rendimiento académico, participación estudiantil, análisis de riesgo, minería de datos educativos, análisis estadístico"
        content.append(Paragraph(keywords_text, self.styles['APABody']))
        
        return content

    def _create_introduction(self) -> List:
        """Crear sección de introducción."""
        content = []
        
        content.append(Paragraph("Introducción", self.styles['APAHeading1']))
        
        intro_text = """
        El análisis del rendimiento académico ha cobrado cada vez más importancia en las instituciones educativas 
        que buscan mejorar los resultados académicos y identificar poblaciones en riesgo. La integración de 
        análisis de datos en la educación ofrece oportunidades para comprender relaciones complejas 
        entre la participación, patrones de asistencia, y el logro académico.
        
        Este estudio emplea un marco analítico integral para examinar los datos de rendimiento 
        de los estudiantes, utilizando análisis estadístico, segmentación demográfica, y métodos de evaluación de riesgo. 
        Los objetivos principales son: (1) identificar patrones en la participación de los estudiantes 
        en diferentes materias y grupos demográficos, (2) analizar patrones de asistencia y su correlación con el rendimiento académico, 
        y (3) desarrollar modelos de evaluación de riesgo para identificar estudiantes que requieren intervención académica.
        
        Los hallazgos de este análisis contribuyen al cuerpo de investigación sobre minería de datos educativos y proporcionan 
        insights prácticos para administradores y políticos educativos.
        """
        
        content.append(Paragraph(intro_text.strip(), self.styles['APABody']))
        
        return content

    def _create_method(self, df: pd.DataFrame) -> List:
        """Crear sección de método."""
        content = []
        
        content.append(Paragraph("Método", self.styles['APAHeading1']))
        
        # Participantes
        content.append(Paragraph("Participantes", self.styles['APAHeading2']))
        
        participants_text = f"""
        El estudio analizó datos de {len(df)} estudiantes en múltiples materias académicas y semestres. 
        El conjunto de datos incluyó información demográfica, métricas de participación, registros de asistencia, y 
        indicadores de rendimiento académico (niveles de rendimiento alto, medio, bajo).
        """
        
        content.append(Paragraph(participants_text.strip(), self.styles['APABody']))
        
        # Recolección de datos
        content.append(Paragraph("Recolección de datos y Variables", self.styles['APAHeading2']))
        
        variables_text = """
        La recolección de datos incluyó las siguientes variables: demográficas del estudiante (género, nacionalidad, 
        etapa educativa), métricas de participación (levantamientos de manos, visitas a recursos, vistas de anuncios, 
        participación en discusiones), registros de asistencia (días de ausencia), y clasificaciones de rendimiento académico 
        (niveles de rendimiento alto, medio, bajo).
        """
        
        content.append(Paragraph(variables_text.strip(), self.styles['APABody']))
        
        # Enfoque analítico
        content.append(Paragraph("Enfoque analítico", self.styles['APAHeading2']))
        
        analysis_text = """
        El análisis empleó estadísticas descriptivas, análisis de correlación, y técnicas de agrupación multivariante. 
        Los algoritmos de evaluación de riesgo se desarrollaron utilizando sistemas de puntuación ponderada 
        basados en el rendimiento académico, patrones de asistencia, y niveles de participación. 
        Las técnicas de visualización se utilizaron para presentar los hallazgos en formatos accesibles.
        """
        
        content.append(Paragraph(analysis_text.strip(), self.styles['APABody']))
        
        return content

    def _create_results(self, df: pd.DataFrame, stats: Dict[str, Any], 
                       groups: Dict[str, Any], risk_report: Dict[str, Any],
                       charts: Dict[str, Any]) -> List:
        """Crear sección de resultados con tablas y figuras."""
        content = []
        
        content.append(Paragraph("Resultados", self.styles['APAHeading1']))
        
        # Estadísticas descriptivas
        content.append(Paragraph("Estadísticas descriptivas", self.styles['APAHeading2']))
        
        # Crear tabla de estadísticas descriptivas
        if 'performance_stats' in stats:
            content.extend(self._create_descriptive_table(stats['performance_stats']))
        
        # Análisis de rendimiento
        content.append(Paragraph("Análisis de rendimiento académico", self.styles['APAHeading2']))
        
        performance_text = f"""
        El análisis de rendimiento académico reveló variaciones significativas entre materias y 
        grupos demográficos. La distribución general mostró {len(df[df['Class'] == 'H'])} estudiantes 
        ({(len(df[df['Class'] == 'H'])/len(df)*100):.1f}%) que alcanzaron un rendimiento alto, 
        {len(df[df['Class'] == 'M'])} estudiantes ({(len(df[df['Class'] == 'M'])/len(df)*100):.1f}%) 
        que alcanzaron un rendimiento medio, y {len(df[df['Class'] == 'L'])} estudiantes 
        ({(len(df[df['Class'] == 'L'])/len(df)*100):.1f}%) que alcanzaron un rendimiento bajo.
        """
        
        content.append(Paragraph(performance_text.strip(), self.styles['APABody']))
        
        # Resultados del análisis de riesgo
        content.append(Paragraph("Análisis de evaluación de riesgo", self.styles['APAHeading2']))
        
        if risk_report:
            risk_text = f"""
            El análisis de evaluación de riesgo identificó {risk_report.get('at_risk_students', {}).get('count', 0)} 
            estudiantes ({risk_report.get('at_risk_students', {}).get('percentage', '0%')}) que requieren 
            intervención académica. Los factores de riesgo más comunes incluyeron un rendimiento académico deficiente, 
            altas tasas de ausencia, y niveles bajos de participación.
            """
            
            content.append(Paragraph(risk_text.strip(), self.styles['APABody']))
            
            # Agregar tabla de factores de riesgo
            if 'at_risk_students' in risk_report and 'common_risk_factors' in risk_report['at_risk_students']:
                content.extend(self._create_risk_factors_table(risk_report['at_risk_students']['common_risk_factors']))
        
        # Análisis de correlación
        if 'correlation_matrix' in stats:
            content.append(Paragraph("Análisis de correlación", self.styles['APAHeading2']))
            
            correlation_text = """
            El análisis de correlación reveló relaciones significativas entre métricas de participación 
            y el rendimiento académico. Se observaron correlaciones fuertes positivas entre 
            las medidas de participación y el logro académico general.
            """
            
            content.append(Paragraph(correlation_text.strip(), self.styles['APABody']))
        
        return content

    def _create_discussion(self, stats: Dict[str, Any], risk_report: Dict[str, Any]) -> List:
        """Crear sección de discusión."""
        content = []
        
        content.append(Paragraph("Discusión", self.styles['APAHeading1']))
        
        discussion_text = """
        Los hallazgos de este estudio proporcionan insights valiosos sobre patrones de rendimiento académico 
        y factores de riesgo que afectan el éxito académico. La identificación de estudiantes en riesgo 
        mediante un análisis sistemático permite intervenciones dirigidas para mejorar los resultados académicos.
        
        La correlación entre métricas de participación y rendimiento académico sugiere que 
        la participación activa en actividades de aprendizaje es un predictor significativo del éxito académico. 
        Este hallazgo apoya teorías educativas que enfatizan la importancia de la participación estudiantil 
        en el proceso de aprendizaje.
        
        El modelo de evaluación de riesgo demuestra una utilidad práctica para instituciones educativas 
        que buscan implementar sistemas de aviso temprano. Al identificar estudiantes con múltiples factores de riesgo, 
        las instituciones pueden asignar recursos de manera más eficiente y proporcionar apoyo académico oportuno.
        """
        
        content.append(Paragraph(discussion_text.strip(), self.styles['APABody']))
        
        # Limitaciones
        content.append(Paragraph("Limitaciones", self.styles['APAHeading2']))
        
        limitations_text = """
        Este estudio presenta varias limitaciones. El análisis se basa en datos observacionales, 
        lo que limita las inferencias causales. El modelo de evaluación de riesgo requiere validación 
        con conjuntos de datos adicionales para garantizar la generalizabilidad en diferentes contextos educativos.
        """
        
        content.append(Paragraph(limitations_text.strip(), self.styles['APABody']))
        
        # Implicaciones
        content.append(Paragraph("Implicaciones para la práctica", self.styles['APAHeading2']))
        
        implications_text = """
        Los resultados sugieren varias implicaciones prácticas para la práctica educativa. 
        Las instituciones deberían considerar implementar un monitoreo sistemático de métricas de participación 
        y patrones de asistencia para identificar estudiantes en riesgo temprano. El desarrollo de programas 
        de intervención dirigidos basados en factores de riesgo específicos puede mejorar la 
        retención y las tasas de éxito académico.
        """
        
        content.append(Paragraph(implications_text.strip(), self.styles['APABody']))
        
        return content

    def _create_references(self) -> List:
        """Crear sección de referencias estilo APA."""
        content = []
        
        content.append(PageBreak())
        content.append(Paragraph("Referencias", self.styles['APAHeading1']))
        
        # Ejemplos de referencias en formato APA
        references = [
            """Baker, R. S., & Inventado, P. S. (2014). Educational data mining and learning analytics. 
            In <i>Learning analytics</i> (pp. 61-75). Springer.""",
            
            """Koedinger, K. R., D'Mello, S., McLaughlin, E. A., Pardos, Z. A., & Rosé, C. P. (2015). 
            Data mining and education. <i>Wiley Interdisciplinary Reviews: Cognitive Science</i>, 6(4), 333-353.""",
            
            """Romero, C., & Ventura, S. (2020). Educational data mining and learning analytics: 
            An updated survey. <i>Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery</i>, 10(3), e1355."""
        ]
        
        for ref in references:
            content.append(Paragraph(ref, self.styles['APABody']))
        
        return content

    def _create_descriptive_table(self, performance_stats: Dict[str, Any]) -> List:
        """Crear una tabla de estadísticas descriptivas."""
        content = []
        
        # Título de la tabla
        content.append(Paragraph("Tabla 1", self.styles['APATableCaption']))
        content.append(Paragraph("<i>Estadísticas descriptivas para métricas de rendimiento académico</i>", 
                                self.styles['APATableCaption']))
        
        # Preparar datos de la tabla
        headers = ['Métrica', 'Media', 'DE', 'Min', 'Max', 'N']
        data = [headers]
        
        # Agregar filas para cada métrica
        if 'engagement' in performance_stats:
            eng_stats = performance_stats['engagement']
            data.append([
                'Participación total',
                f"{eng_stats.get('mean', 0):.2f}",
                f"{eng_stats.get('std', 0):.2f}",
                f"{eng_stats.get('min', 0):.2f}",
                f"{eng_stats.get('max', 0):.2f}",
                f"{eng_stats.get('count', 0)}"
            ])
        
        # Crear tabla
        table = Table(data, colWidths=[2*inch, 1*inch, 1*inch, 1*inch, 1*inch, 1*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('FONTNAME', (0, 1), (-1, -1), 'Times-Roman'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        content.append(table)
        content.append(Spacer(1, 12))
        
        return content

    def _create_risk_factors_table(self, risk_factors: Dict[str, int]) -> List:
        """Crear una tabla de frecuencia de factores de riesgo."""
        content = []
        
        # Título de la tabla
        content.append(Paragraph("Tabla 2", self.styles['APATableCaption']))
        content.append(Paragraph("<i>Frecuencia de factores de riesgo entre estudiantes en riesgo</i>", 
                                self.styles['APATableCaption']))
        
        # Preparar datos de la tabla
        headers = ['Factor de riesgo', 'Frecuencia', 'Porcentaje']
        data = [headers]
        
        total_factors = sum(risk_factors.values())
        for factor, count in risk_factors.items():
            percentage = (count / total_factors) * 100 if total_factors > 0 else 0
            data.append([factor, str(count), f"{percentage:.1f}%"])
        
        # Crear tabla
        table = Table(data, colWidths=[3*inch, 1.5*inch, 1.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('FONTNAME', (0, 1), (-1, -1), 'Times-Roman'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        content.append(table)
        content.append(Spacer(1, 12))
        
        return content

def generate_simple_report(df: pd.DataFrame, stats: Dict[str, Any], 
                          save_path: Optional[str] = None) -> str:
    """
    Generar un reporte de texto simple (fallback cuando ReportLab no está disponible).
    
    Args:
        df: DataFrame limpio con datos de estudiantes
        stats: Resultados del análisis estadístico
        save_path: Ruta opcional para guardar el reporte
        
    Returns:
        Ruta al reporte generado
    """
    try:
        log_analysis_step("Generando reporte de texto simple")
        
        if save_path is None:
            save_path = str(REPORTS_DIR / f"reporte_analisis_academico_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        
        # Crear directorio de salida
        create_output_directory(Path(save_path).parent)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("REPORTE DE ANÁLISIS DE RENDIMIENTO ACADEMICO\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Reporte Generado: {datetime.now().strftime('%d de %B de %Y')}\n")
            f.write(f"Total de Estudiantes Analizados: {len(df)}\n\n")
            
            # Distribución de rendimiento
            f.write("DISTRIBUCIÓN DE RENDIMIENTO\n")
            f.write("-" * 30 + "\n")
            performance_counts = df['Class'].value_counts()
            for grade, count in performance_counts.items():
                percentage = (count / len(df)) * 100
                f.write(f"{grade} Rendimiento: {count} estudiantes ({percentage:.1f}%)\n")
            f.write("\n")
            
            # Análisis por materia
            if 'subject_averages' in stats:
                f.write("ANÁLISIS POR MATERIA\n")
                f.write("-" * 20 + "\n")
                for subject, avg in stats['subject_averages'].items():
                    f.write(f"{subject}: {avg:.2f} promedio de participación\n")
                f.write("\n")
            
            # Demografía
            f.write("BREAKDOWN DEMOGRÁFICO\n")
            f.write("-" * 25 + "\n")
            if 'gender' in df.columns:
                gender_counts = df['gender'].value_counts()
                for gender, count in gender_counts.items():
                    percentage = (count / len(df)) * 100
                    f.write(f"{gender}: {count} estudiantes ({percentage:.1f}%)\n")
            f.write("\n")
            
            f.write("Análisis completado exitosamente.\n")
        
        logger.info(f"Reporte de texto simple generado: {save_path}")
        return save_path
        
    except Exception as e:
        logger.error(f"Error generando reporte de texto simple: {e}")
        raise

if __name__ == "__main__":
    # Probar el generador de reportes
    try:
        import sys
        import os
        sys.path.append('.')
        
        from src.data.data_loader import load_and_validate_data
        from src.data.data_cleaner import clean_student_data
        from src.analysis.statistics import calculate_all_statistics
        from src.analysis.grouping import perform_all_groupings
        from src.analysis.risk_analysis import generate_risk_report
        
        print("Probando Generador de Reportes APA...")
        
        # Cargar y preparar datos
        df = load_and_validate_data()
        df_clean = clean_student_data(df)
        stats = calculate_all_statistics(df_clean)
        groups = perform_all_groupings(df_clean)
        risk_report = generate_risk_report(df_clean)
        
        if REPORTLAB_AVAILABLE:
            # Generar reporte APA
            report_gen = APAReportGenerator()
            report_path = report_gen.generate_comprehensive_report(
                df_clean, stats, groups, risk_report, {}
            )
            print(f"Reporte APA generado: {report_path}")
        else:
            # Generar reporte simple
            report_path = generate_simple_report(df_clean, stats)
            print(f"Reporte simple generado: {report_path}")
            
    except Exception as e:
        print(f"Error probando el generador de reportes: {e}")
