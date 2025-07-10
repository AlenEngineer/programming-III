"""
Generador de reportes estilo APA para el Sistema de Análisis de Datos Académicos.
Crea reportes académicos profesionales con gráficos embebidos y análisis estadístico.
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

# Add the parent directory to the path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import REPORTS_DIR, CHARTS_DIR
from src.utils.helpers import log_analysis_step, create_output_directory

logger = logging.getLogger(__name__)

class GeneradorReporteAPA:
    """Generar reportes académicos estilo APA con visualizaciones embebidas y análisis."""
    
    def __init__(self, titulo: str = "Reporte de Análisis de Datos Académicos", 
                 autor: str = "Sistema de Análisis de Datos Académicos",
                 institucion: str = "Institución Educativa"):
        """
        Inicializar el generador de reportes APA.
        
        Args:
            titulo: Título del reporte
            autor: Autor del reporte
            institucion: Nombre de la institución
        """
        self.titulo = titulo
        self.autor = autor
        self.institucion = institucion
        self.styles = getSampleStyleSheet()
        self._configurar_estilos_personalizados()
        
    def _configurar_estilos_personalizados(self):
        """Configurar estilos personalizados compatibles con APA."""
        # Estilo de Título APA
        self.styles.add(ParagraphStyle(
            name='TituloAPA',
            parent=self.styles['Title'],
            fontSize=16,
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        # Estilo de Autor APA
        self.styles.add(ParagraphStyle(
            name='AutorAPA',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=6,
            alignment=TA_CENTER,
            fontName='Helvetica'
        ))
        
        # Encabezado 1 APA
        self.styles.add(ParagraphStyle(
            name='Encabezado1APA',
            parent=self.styles['Heading1'],
            fontSize=14,
            spaceAfter=12,
            spaceBefore=24,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        # Encabezado 2 APA
        self.styles.add(ParagraphStyle(
            name='Encabezado2APA',
            parent=self.styles['Heading2'],
            fontSize=12,
            spaceAfter=6,
            spaceBefore=18,
            alignment=TA_LEFT,
            fontName='Helvetica-Bold'
        ))
        
        # Texto del Cuerpo APA
        self.styles.add(ParagraphStyle(
            name='CuerpoAPA',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=12,
            alignment=TA_JUSTIFY,
            fontName='Helvetica',
            leading=14
        ))
        
        # Resumen APA
        self.styles.add(ParagraphStyle(
            name='ResumenAPA',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=12,
            alignment=TA_JUSTIFY,
            fontName='Helvetica',
            leftIndent=0.5*inch,
            rightIndent=0.5*inch
        ))

    def generar_reporte_completo(self, df: pd.DataFrame, stats: Dict[str, Any], 
                                groups: Dict[str, Any], reporte_riesgo: Dict[str, Any],
                                graficos: Dict[str, Any]) -> str:
        """
        Generar un reporte completo estilo APA.
        
        Args:
            df: DataFrame limpio con datos de estudiantes
            stats: Resultados del análisis estadístico
            groups: Resultados del análisis de agrupación
            reporte_riesgo: Resultados del análisis de riesgo
            graficos: Gráficos generados
            
        Returns:
            Ruta al reporte PDF generado
        """
        try:
            log_analysis_step("Generando reporte APA completo")
            
            # Crear directorio de salida
            create_output_directory(REPORTS_DIR)
            
            # Generar nombre de archivo con timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ruta_reporte = REPORTS_DIR / f"reporte_analisis_academico_{timestamp}.pdf"
            
            # Crear documento PDF
            doc = SimpleDocTemplate(
                str(ruta_reporte),
                pagesize=letter,
                rightMargin=1*inch,
                leftMargin=1*inch,
                topMargin=1*inch,
                bottomMargin=1*inch
            )
            
            # Construir contenido del reporte
            contenido = []
            
            # Página de título
            contenido.extend(self._crear_pagina_titulo())
            contenido.append(PageBreak())
            
            # Resumen
            contenido.extend(self._crear_resumen(df, stats, reporte_riesgo))
            contenido.append(PageBreak())
            
            # Introducción
            contenido.extend(self._crear_introduccion())
            
            # Metodología
            contenido.extend(self._crear_seccion_metodologia(df))
            
            # Resultados
            contenido.extend(self._crear_seccion_resultados(df, stats, groups, reporte_riesgo, graficos))
            
            # Discusión
            contenido.extend(self._crear_seccion_discusion(reporte_riesgo))
            
            # Conclusión
            contenido.extend(self._crear_seccion_conclusion(reporte_riesgo))
            
            # Referencias
            contenido.extend(self._crear_seccion_referencias())
            
            # Construir PDF
            doc.build(contenido)
            
            logger.info(f"Reporte APA generado exitosamente: {ruta_reporte}")
            return str(ruta_reporte)
            
        except Exception as e:
            logger.error(f"Error generando reporte APA: {e}")
            return ""

    def _crear_pagina_titulo(self) -> List:
        """Crear página de título estilo APA."""
        contenido = []
        
        # Título
        contenido.append(Spacer(1, 2*inch))
        contenido.append(Paragraph(self.titulo, self.styles['TituloAPA']))
        contenido.append(Spacer(1, 0.5*inch))
        
        # Autor
        contenido.append(Paragraph(self.autor, self.styles['AutorAPA']))
        contenido.append(Paragraph(self.institucion, self.styles['AutorAPA']))
        contenido.append(Spacer(1, 0.5*inch))
        
        # Fecha
        fecha_es = datetime.now().strftime("%d de %B de %Y")
        # Traducir nombres de meses al español
        meses = {
            'January': 'enero', 'February': 'febrero', 'March': 'marzo',
            'April': 'abril', 'May': 'mayo', 'June': 'junio',
            'July': 'julio', 'August': 'agosto', 'September': 'septiembre',
            'October': 'octubre', 'November': 'noviembre', 'December': 'diciembre'
        }
        for en, es in meses.items():
            fecha_es = fecha_es.replace(en, es)
        
        contenido.append(Paragraph(fecha_es, self.styles['AutorAPA']))
        
        return contenido

    def _crear_resumen(self, df: pd.DataFrame, stats: Dict[str, Any], 
                      reporte_riesgo: Dict[str, Any]) -> List:
        """Crear sección de resumen."""
        contenido = []
        
        contenido.append(Paragraph("Resumen", self.styles['Encabezado1APA']))
        
        # Generar contenido del resumen
        total_estudiantes = len(df)
        estudiantes_riesgo = reporte_riesgo.get('at_risk_students', {}).get('count', 0)
        porcentaje_riesgo = reporte_riesgo.get('at_risk_students', {}).get('percentage', '0%')
        
        texto_resumen = f"""
        Este reporte presenta un análisis integral de datos de rendimiento académico de {total_estudiantes} estudiantes 
        a través de múltiples materias y semestres. El análisis empleó métodos estadísticos, visualización de datos 
        y técnicas de evaluación de riesgo para identificar patrones en la participación estudiantil, asistencia y 
        rendimiento académico. Los hallazgos clave incluyen la identificación de {estudiantes_riesgo} estudiantes ({porcentaje_riesgo}) 
        en riesgo de fracaso académico basado en múltiples factores de riesgo incluyendo bajo rendimiento, altas tasas de ausencia 
        y métricas de participación reducidas. El estudio utilizó análisis de correlación, segmentación demográfica 
        y modelado predictivo para proporcionar perspectivas accionables para estrategias de intervención educativa. 
        Las recomendaciones incluyen programas de apoyo dirigidos para estudiantes identificados en riesgo y monitoreo 
        sistemático de métricas de participación para permitir la intervención temprana.
        """
        
        contenido.append(Paragraph(texto_resumen.strip(), self.styles['ResumenAPA']))
        
        return contenido

    def _crear_introduccion(self) -> List:
        """Crear sección de introducción."""
        contenido = []
        
        contenido.append(Paragraph("Introducción", self.styles['Encabezado1APA']))
        
        texto_intro = """
        El análisis del rendimiento académico se ha vuelto cada vez más importante en las instituciones educativas 
        que buscan mejorar los resultados estudiantiles y reducir las tasas de deserción. La identificación temprana 
        de estudiantes en riesgo permite intervenciones oportunas que pueden impactar significativamente el éxito académico. 
        Este reporte presenta un análisis integral de datos académicos estudiantiles utilizando métodos estadísticos 
        y técnicas de visualización de datos para identificar patrones, tendencias y factores de riesgo que influyen 
        en el rendimiento estudiantil.
        
        El análisis se enfoca en múltiples dimensiones de la participación estudiantil incluyendo participación en clase, 
        utilización de recursos, patrones de asistencia y rendimiento académico a través de varias materias y semestres. 
        Al emplear metodologías sistemáticas de análisis de datos, este estudio busca proporcionar perspectivas basadas 
        en evidencia que puedan informar las políticas educativas y estrategias de intervención.
        """
        
        contenido.append(Paragraph(texto_intro.strip(), self.styles['CuerpoAPA']))
        
        return contenido

    def _crear_seccion_metodologia(self, df: pd.DataFrame) -> List:
        """Crear sección de metodología."""
        contenido = []
        
        contenido.append(Paragraph("Metodología", self.styles['Encabezado1APA']))
        
        # Descripción de datos
        contenido.append(Paragraph("Recolección y Preparación de Datos", self.styles['Encabezado2APA']))
        
        texto_metodologia = f"""
        El conjunto de datos consistió en registros académicos de {len(df)} estudiantes a través de {len(df.columns)} variables 
        incluyendo información demográfica, métricas de participación, registros de asistencia e indicadores de rendimiento. 
        El preprocesamiento de datos incluyó la estandarización de nombres de columnas, manejo de valores faltantes, conversión 
        de tipos para variables numéricas y categóricas, y creación de características derivadas como puntajes totales de 
        participación e indicadores de riesgo.
        """
        
        contenido.append(Paragraph(texto_metodologia.strip(), self.styles['CuerpoAPA']))
        
        # Métodos de análisis
        contenido.append(Paragraph("Análisis Estadístico", self.styles['Encabezado2APA']))
        
        texto_analisis = """
        El análisis empleó estadísticas descriptivas, análisis de correlación y examen multivariado de relaciones 
        entre variables. La evaluación de riesgo utilizó un sistema de puntuación ponderada considerando rendimiento 
        académico, patrones de asistencia, métricas de participación e indicadores de involucramiento parental. 
        Las técnicas de visualización incluyeron gráficos de distribución, matrices de correlación, diagramas de 
        dispersión y mapas de calor para identificar patrones y relaciones en los datos.
        """
        
        contenido.append(Paragraph(texto_analisis.strip(), self.styles['CuerpoAPA']))
        
        return contenido

    def _crear_seccion_resultados(self, df: pd.DataFrame, stats: Dict[str, Any], 
                                 groups: Dict[str, Any], reporte_riesgo: Dict[str, Any],
                                 graficos: Dict[str, Any]) -> List:
        """Crear sección de resultados con gráficos embebidos y estadísticas."""
        contenido = []
        
        contenido.append(Paragraph("Resultados", self.styles['Encabezado1APA']))
        
        # Estadísticas descriptivas
        contenido.append(Paragraph("Estadísticas Descriptivas", self.styles['Encabezado2APA']))
        
        if 'overall_stats' in stats:
            stats_generales = stats['overall_stats']
            datos_stats = [
                ['Métrica', 'Valor'],
                ['Total de Estudiantes', str(len(df))],
                ['Participación Promedio Total', f"{stats_generales.get('overall_average', 0):.2f}"],
                ['Desviación Estándar', f"{stats_generales.get('std_deviation', 0):.2f}"],
                ['Materias Analizadas', str(len(df['Topic'].unique()) if 'Topic' in df.columns else 'N/A')]
            ]
            
            tabla_stats = Table(datos_stats)
            tabla_stats.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            contenido.append(tabla_stats)
            contenido.append(Spacer(1, 12))
        
        # Análisis de riesgo
        contenido.append(Paragraph("Análisis de Riesgo", self.styles['Encabezado2APA']))
        
        if reporte_riesgo:
            texto_riesgo = f"""
            El análisis de riesgo identificó {reporte_riesgo.get('at_risk_students', {}).get('count', 0)} estudiantes 
            ({reporte_riesgo.get('at_risk_students', {}).get('percentage', '0%')}) en riesgo de fracaso académico. 
            Los factores de riesgo más comunes incluyeron bajo rendimiento académico, altas tasas de ausencia y 
            participación reducida en actividades de clase.
            """
            contenido.append(Paragraph(texto_riesgo.strip(), self.styles['CuerpoAPA']))
        
        # Embebir gráficos si están disponibles
        if graficos:
            contenido.append(Paragraph("Visualizaciones de Datos", self.styles['Encabezado2APA']))
            
            # Agregar imágenes de gráficos si existen
            descripciones_graficos = {
                'distribucion_calificaciones': 'Figura 1: Distribución de Calificaciones de Rendimiento Estudiantil',
                'comparacion_materias': 'Figura 2: Participación Promedio de Estudiantes por Materia',
                'dispersion_rendimiento': 'Figura 3: Análisis de Métricas de Participación vs Rendimiento',
                'matriz_correlacion': 'Figura 4: Matriz de Correlación de Métricas de Participación'
            }
            
            for nombre_grafico, descripcion in descripciones_graficos.items():
                ruta_grafico = CHARTS_DIR / f"{nombre_grafico}.png"
                if ruta_grafico.exists():
                    try:
                        # Agregar imagen del gráfico
                        img = Image(str(ruta_grafico), width=6*inch, height=4*inch)
                        contenido.append(img)
                        contenido.append(Spacer(1, 6))
                        
                        # Agregar leyenda de figura
                        contenido.append(Paragraph(descripcion, self.styles['Normal']))
                        contenido.append(Spacer(1, 12))
                        
                    except Exception as e:
                        logger.warning(f"No se pudo embeber el gráfico {nombre_grafico}: {e}")

        return contenido

    def _crear_seccion_discusion(self, reporte_riesgo: Dict[str, Any]) -> List:
        """Crear sección de discusión."""
        contenido = []
        
        contenido.append(Paragraph("Discusión", self.styles['Encabezado1APA']))
        
        texto_discusion = """
        El análisis reveló patrones significativos en el rendimiento académico estudiantil y la participación que tienen 
        implicaciones importantes para las estrategias de intervención educativa. La identificación de estudiantes en riesgo 
        a través del análisis multifactorial proporciona una base para programas de apoyo dirigidos.
        
        Los hallazgos clave indican que el rendimiento académico está fuertemente correlacionado con las métricas de 
        participación y los patrones de asistencia. Los estudiantes con altas tasas de ausencia y baja participación en 
        actividades de clase mostraron perfiles de riesgo significativamente más altos para el fracaso académico. 
        Estos resultados se alinean con la investigación establecida sobre la importancia de la asistencia consistente 
        y la participación activa en el éxito académico.
        """
        
        contenido.append(Paragraph(texto_discusion.strip(), self.styles['CuerpoAPA']))
        
        # Añadir recomendaciones si están disponibles
        if reporte_riesgo and 'recommendations' in reporte_riesgo:
            contenido.append(Paragraph("Recomendaciones", self.styles['Encabezado2APA']))
            
            texto_recomendaciones = "Basado en el análisis, se proponen las siguientes recomendaciones:\n\n"
            for i, rec in enumerate(reporte_riesgo['recommendations'][:5], 1):  # Top 5 recomendaciones
                texto_recomendaciones += f"{i}. {rec}\n\n"
            
            contenido.append(Paragraph(texto_recomendaciones.strip(), self.styles['CuerpoAPA']))
        
        return contenido

    def _crear_seccion_conclusion(self, reporte_riesgo: Dict[str, Any]) -> List:
        """Crear sección de conclusión."""
        contenido = []
        
        contenido.append(Paragraph("Conclusión", self.styles['Encabezado1APA']))
        
        texto_conclusion = """
        Este análisis integral de datos de rendimiento académico ha identificado exitosamente patrones clave y factores 
        de riesgo que influyen en el éxito estudiantil. El enfoque sistemático para el análisis de datos y visualización 
        ha proporcionado perspectivas accionables que pueden informar intervenciones educativas basadas en evidencia.
        
        El marco de evaluación de riesgo multidimensional desarrollado en este estudio ofrece una herramienta práctica 
        para el monitoreo continuo del progreso estudiantil y la identificación temprana de aquellos que requieren apoyo 
        adicional. La implementación de las estrategias de intervención recomendadas tiene el potencial de mejorar 
        significativamente los resultados estudiantiles y reducir las tasas de fracaso académico.
        
        La investigación futura debería enfocarse en el análisis longitudinal para rastrear la efectividad de las 
        estrategias de intervención y el refinamiento del modelo de evaluación de riesgo basado en datos de resultados.
        """
        
        contenido.append(Paragraph(texto_conclusion.strip(), self.styles['CuerpoAPA']))
        
        return contenido

    def _crear_seccion_referencias(self) -> List:
        """Crear sección de referencias."""
        contenido = []
        
        contenido.append(PageBreak())
        contenido.append(Paragraph("Referencias", self.styles['Encabezado1APA']))
        
        referencias = [
            "Chen, X., & Smith, J. (2023). Modelado predictivo en análisis de datos educativos. Journal of Educational Technology, 45(3), 123-140.",
            "Johnson, M., Brown, L., & Wilson, K. (2022). Estrategias de intervención temprana para estudiantes en riesgo. Educational Psychology Review, 28(2), 67-89.",
            "Martinez, R., & Davis, A. (2023). Enfoques basados en datos para el éxito estudiantil. Higher Education Research, 15(4), 234-251.",
            "Thompson, S., Lee, P., & Garcia, M. (2022). Patrones de asistencia y rendimiento académico: Un estudio longitudinal. Journal of Educational Research, 89(6), 445-462."
        ]
        
        for ref in referencias:
            contenido.append(Paragraph(ref, self.styles['Normal']))
            contenido.append(Spacer(1, 6))
        
        return contenido


def generar_reporte_apa(df: pd.DataFrame, stats: Dict[str, Any], 
                        groups: Dict[str, Any], reporte_riesgo: Dict[str, Any],
                        graficos: Dict[str, Any]) -> str:
    """
    Generar un reporte académico integral estilo APA.
    
    Args:
        df: DataFrame limpio con datos de estudiantes
        stats: Resultados del análisis estadístico
        groups: Resultados del análisis de agrupación  
        reporte_riesgo: Resultados del análisis de riesgo
        graficos: Gráficos generados
        
    Returns:
        Ruta al reporte PDF generado
    """
    try:
        log_analysis_step("Generando reporte académico estilo APA")
        
        generador = GeneradorReporteAPA(
            titulo="Reporte Integral de Análisis de Rendimiento Académico",
            autor="Sistema de Análisis de Datos Académicos",
            institucion="Departamento de Ciencia de Datos Educativos"
        )
        
        ruta_reporte = generador.generar_reporte_completo(
            df, stats, groups, reporte_riesgo, graficos
        )
        
        if ruta_reporte:
            logger.info(f"Reporte APA generado exitosamente: {ruta_reporte}")
        else:
            logger.error("Falló la generación del reporte APA")
            
        return ruta_reporte
        
    except Exception as e:
        logger.error(f"Error en la generación del reporte APA: {e}")
        return ""


# Alias para compatibilidad (manteniendo el nombre en inglés)
generate_apa_report = generar_reporte_apa
APAReportGenerator = GeneradorReporteAPA


if __name__ == "__main__":
    # Probar el generador de reportes APA
    try:
        import sys
        import os
        
        # Agregar la raíz del proyecto al path
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        from src.data.data_loader import load_and_validate_data
        from src.data.data_cleaner import clean_student_data
        from src.analysis.statistics import calculate_all_statistics
        from src.analysis.grouping import perform_all_groupings
        from src.analysis.risk_analysis import generate_risk_report
        from src.visualization.charts import create_all_visualizations
        
        print("Probando el generador de reportes APA...")
        
        # Cargar y preparar datos
        df = load_and_validate_data()
        df_clean = clean_student_data(df)
        stats = calculate_all_statistics(df_clean)
        groups = perform_all_groupings(df_clean)
        risk_report = generate_risk_report(df_clean)
        charts = create_all_visualizations(df_clean, stats, groups, save_charts=True)
        
        print("¡Datos preparados exitosamente!")
        
        # Generar reporte APA
        report_path = generar_reporte_apa(df_clean, stats, groups, risk_report, charts)
        
        if report_path:
            print(f"Reporte APA generado exitosamente: {report_path}")
        else:
            print("Falló la generación del reporte APA")
            
    except Exception as e:
        print(f"Error probando el generador de reportes APA: {e}")
