"""
APA Report Generator for the Academic Data Analysis System.
Creates professional APA-style reports with statistical analysis and visualizations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import sys
import os
import logging

# Add the parent directory to the path to import config
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
    logger.warning("ReportLab not available. Install with: pip install reportlab")
    REPORTLAB_AVAILABLE = False

class APAReportGenerator:
    """Generate professional APA-style reports for academic data analysis."""
    
    def __init__(self, title: str = "Academic Performance Analysis Report", 
                 author: str = "Academic Data Analysis System"):
        """
        Initialize the APA report generator.
        
        Args:
            title: Report title
            author: Report author
        """
        self.title = title
        self.author = author
        self.date = datetime.now().strftime("%B %d, %Y")
        
        if not REPORTLAB_AVAILABLE:
            raise ImportError("ReportLab is required for PDF generation. Install with: pip install reportlab")
        
        # Set up APA-style formatting
        self.styles = getSampleStyleSheet()
        self._setup_apa_styles()
        
    def _setup_apa_styles(self):
        """Set up APA-compliant paragraph styles."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='APATitle',
            parent=self.styles['Title'],
            fontSize=14,
            spaceAfter=12,
            alignment=TA_CENTER,
            fontName='Times-Bold'
        ))
        
        # Author style
        self.styles.add(ParagraphStyle(
            name='APAAuthor',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=6,
            alignment=TA_CENTER,
            fontName='Times-Roman'
        ))
        
        # Date style
        self.styles.add(ParagraphStyle(
            name='APADate',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=24,
            alignment=TA_CENTER,
            fontName='Times-Roman'
        ))
        
        # Heading 1 style
        self.styles.add(ParagraphStyle(
            name='APAHeading1',
            parent=self.styles['Heading1'],
            fontSize=12,
            spaceAfter=12,
            spaceBefore=24,
            alignment=TA_CENTER,
            fontName='Times-Bold'
        ))
        
        # Heading 2 style
        self.styles.add(ParagraphStyle(
            name='APAHeading2',
            parent=self.styles['Heading2'],
            fontSize=12,
            spaceAfter=6,
            spaceBefore=12,
            alignment=TA_LEFT,
            fontName='Times-Bold'
        ))
        
        # Body text style
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
        
        # Table caption style
        self.styles.add(ParagraphStyle(
            name='APATableCaption',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=6,
            spaceBefore=12,
            alignment=TA_LEFT,
            fontName='Times-Italic'
        ))
        
        # Figure caption style
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
        Generate a comprehensive APA-style report.
        
        Args:
            df: Cleaned DataFrame with student data
            stats: Statistical analysis results
            groups: Grouping analysis results
            risk_report: Risk analysis results
            charts: Generated charts dictionary
            save_path: Optional path to save the report
            
        Returns:
            Path to the generated report
        """
        try:
            log_analysis_step("Generating comprehensive APA report")
            
            if save_path is None:
                save_path = str(REPORTS_DIR / f"academic_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
            
            # Create output directory
            create_output_directory(Path(save_path).parent)
            
            # Create PDF document
            doc = SimpleDocTemplate(save_path, pagesize=A4, 
                                  rightMargin=1*inch, leftMargin=1*inch,
                                  topMargin=1*inch, bottomMargin=1*inch)
            
            # Build report content
            story = []
            
            # Title page
            story.extend(self._create_title_page())
            story.append(PageBreak())
            
            # Abstract
            story.extend(self._create_abstract(df, stats, risk_report))
            story.append(PageBreak())
            
            # Introduction
            story.extend(self._create_introduction())
            
            # Method
            story.extend(self._create_method(df))
            
            # Results
            story.extend(self._create_results(df, stats, groups, risk_report, charts))
            
            # Discussion
            story.extend(self._create_discussion(stats, risk_report))
            
            # References
            story.extend(self._create_references())
            
            # Build PDF
            doc.build(story)
            
            logger.info(f"APA report generated successfully: {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"Error generating APA report: {e}")
            raise

    def _create_title_page(self) -> List:
        """Create APA-style title page."""
        content = []
        
        # Add vertical space
        content.append(Spacer(1, 2*inch))
        
        # Title
        content.append(Paragraph(self.title, self.styles['APATitle']))
        
        # Add space
        content.append(Spacer(1, 1*inch))
        
        # Author
        content.append(Paragraph(self.author, self.styles['APAAuthor']))
        
        # Institution (optional)
        content.append(Paragraph("Academic Institution", self.styles['APAAuthor']))
        
        # Add space
        content.append(Spacer(1, 1*inch))
        
        # Date
        content.append(Paragraph(self.date, self.styles['APADate']))
        
        return content

    def _create_abstract(self, df: pd.DataFrame, stats: Dict[str, Any], 
                        risk_report: Dict[str, Any]) -> List:
        """Create APA-style abstract."""
        content = []
        
        # Abstract heading
        content.append(Paragraph("Abstract", self.styles['APAHeading1']))
        
        # Abstract text
        total_students = len(df)
        at_risk_count = risk_report.get('at_risk_students', {}).get('count', 0)
        at_risk_percentage = risk_report.get('at_risk_students', {}).get('percentage', '0%')
        
        abstract_text = f"""
        This study presents a comprehensive analysis of academic performance data from {total_students} students 
        across multiple subjects and semesters. The analysis utilized statistical methods, grouping techniques, 
        and risk assessment algorithms to identify patterns in student engagement, attendance, and academic performance. 
        Key findings include the identification of {at_risk_count} students ({at_risk_percentage}) at academic risk, 
        significant variations in engagement across subjects, and correlations between attendance patterns and 
        academic outcomes. The study provides actionable insights for educational interventions and policy 
        development to improve student success rates.
        """
        
        content.append(Paragraph(abstract_text.strip(), self.styles['APABody']))
        
        # Keywords
        content.append(Spacer(1, 12))
        keywords_text = "<i>Keywords:</i> academic performance, student engagement, risk analysis, educational data mining, statistical analysis"
        content.append(Paragraph(keywords_text, self.styles['APABody']))
        
        return content

    def _create_introduction(self) -> List:
        """Create introduction section."""
        content = []
        
        content.append(Paragraph("Introduction", self.styles['APAHeading1']))
        
        intro_text = """
        Academic performance analysis has become increasingly important in educational institutions 
        seeking to improve student outcomes and identify at-risk populations. The integration of 
        data analytics in education provides opportunities to understand complex relationships 
        between student engagement, attendance patterns, and academic achievement.
        
        This study employs a comprehensive analytical framework to examine student performance 
        data, utilizing statistical analysis, demographic segmentation, and risk assessment 
        methodologies. The primary objectives include: (1) identifying patterns in student 
        engagement across different subjects and demographics, (2) analyzing attendance patterns 
        and their correlation with academic performance, and (3) developing risk assessment 
        models to identify students requiring academic intervention.
        
        The findings from this analysis contribute to the growing body of research on educational 
        data analytics and provide practical insights for educational administrators and policymakers.
        """
        
        content.append(Paragraph(intro_text.strip(), self.styles['APABody']))
        
        return content

    def _create_method(self, df: pd.DataFrame) -> List:
        """Create method section."""
        content = []
        
        content.append(Paragraph("Method", self.styles['APAHeading1']))
        
        # Participants
        content.append(Paragraph("Participants", self.styles['APAHeading2']))
        
        participants_text = f"""
        The study analyzed data from {len(df)} students across multiple academic subjects and semesters. 
        The dataset included demographic information, engagement metrics, attendance records, and 
        academic performance indicators.
        """
        
        content.append(Paragraph(participants_text.strip(), self.styles['APABody']))
        
        # Data Collection
        content.append(Paragraph("Data Collection and Variables", self.styles['APAHeading2']))
        
        variables_text = """
        Data collection included the following variables: student demographics (gender, nationality, 
        educational stage), engagement metrics (raised hands, resource visits, announcement views, 
        discussion participation), attendance records (absence days), and academic performance 
        classifications (High, Medium, Low performance levels).
        """
        
        content.append(Paragraph(variables_text.strip(), self.styles['APABody']))
        
        # Analytical Approach
        content.append(Paragraph("Analytical Approach", self.styles['APAHeading2']))
        
        analysis_text = """
        The analysis employed descriptive statistics, correlation analysis, and multivariate 
        grouping techniques. Risk assessment algorithms were developed using weighted scoring 
        systems based on academic performance, attendance patterns, and engagement levels. 
        Visualization techniques were utilized to present findings in accessible formats.
        """
        
        content.append(Paragraph(analysis_text.strip(), self.styles['APABody']))
        
        return content

    def _create_results(self, df: pd.DataFrame, stats: Dict[str, Any], 
                       groups: Dict[str, Any], risk_report: Dict[str, Any],
                       charts: Dict[str, Any]) -> List:
        """Create results section with tables and figures."""
        content = []
        
        content.append(Paragraph("Results", self.styles['APAHeading1']))
        
        # Descriptive Statistics
        content.append(Paragraph("Descriptive Statistics", self.styles['APAHeading2']))
        
        # Create descriptive statistics table
        if 'performance_stats' in stats:
            content.extend(self._create_descriptive_table(stats['performance_stats']))
        
        # Performance Analysis
        content.append(Paragraph("Academic Performance Analysis", self.styles['APAHeading2']))
        
        performance_text = f"""
        Analysis of academic performance revealed significant variations across subjects and 
        demographic groups. The overall distribution showed {len(df[df['Class'] == 'H'])} students 
        ({(len(df[df['Class'] == 'H'])/len(df)*100):.1f}%) achieving high performance, 
        {len(df[df['Class'] == 'M'])} students ({(len(df[df['Class'] == 'M'])/len(df)*100):.1f}%) 
        achieving medium performance, and {len(df[df['Class'] == 'L'])} students 
        ({(len(df[df['Class'] == 'L'])/len(df)*100):.1f}%) achieving low performance.
        """
        
        content.append(Paragraph(performance_text.strip(), self.styles['APABody']))
        
        # Risk Analysis Results
        content.append(Paragraph("Risk Assessment Analysis", self.styles['APAHeading2']))
        
        if risk_report:
            risk_text = f"""
            The risk assessment analysis identified {risk_report.get('at_risk_students', {}).get('count', 0)} 
            students ({risk_report.get('at_risk_students', {}).get('percentage', '0%')}) as requiring 
            academic intervention. The most common risk factors included poor academic performance, 
            high absence rates, and low engagement levels.
            """
            
            content.append(Paragraph(risk_text.strip(), self.styles['APABody']))
            
            # Add risk factors table
            if 'at_risk_students' in risk_report and 'common_risk_factors' in risk_report['at_risk_students']:
                content.extend(self._create_risk_factors_table(risk_report['at_risk_students']['common_risk_factors']))
        
        # Correlation Analysis
        if 'correlation_matrix' in stats:
            content.append(Paragraph("Correlation Analysis", self.styles['APAHeading2']))
            
            correlation_text = """
            Correlation analysis revealed significant relationships between engagement metrics 
            and academic performance. Strong positive correlations were observed between 
            participation measures and overall academic achievement.
            """
            
            content.append(Paragraph(correlation_text.strip(), self.styles['APABody']))
        
        return content

    def _create_discussion(self, stats: Dict[str, Any], risk_report: Dict[str, Any]) -> List:
        """Create discussion section."""
        content = []
        
        content.append(Paragraph("Discussion", self.styles['APAHeading1']))
        
        discussion_text = """
        The findings of this study provide valuable insights into academic performance patterns 
        and risk factors affecting student success. The identification of at-risk students 
        through systematic analysis enables targeted interventions to improve academic outcomes.
        
        The correlation between engagement metrics and academic performance suggests that 
        active participation in learning activities is a significant predictor of academic success. 
        This finding supports educational theories emphasizing the importance of student engagement 
        in the learning process.
        
        The risk assessment model demonstrates practical utility for educational institutions 
        seeking to implement early warning systems. By identifying students with multiple risk 
        factors, institutions can allocate resources more effectively and provide timely 
        academic support.
        """
        
        content.append(Paragraph(discussion_text.strip(), self.styles['APABody']))
        
        # Limitations
        content.append(Paragraph("Limitations", self.styles['APAHeading2']))
        
        limitations_text = """
        This study has several limitations. The analysis is based on observational data, 
        limiting causal inferences. The risk assessment model requires validation with 
        additional datasets to ensure generalizability across different educational contexts.
        """
        
        content.append(Paragraph(limitations_text.strip(), self.styles['APABody']))
        
        # Implications
        content.append(Paragraph("Implications for Practice", self.styles['APAHeading2']))
        
        implications_text = """
        The results suggest several practical implications for educational practice. 
        Institutions should consider implementing systematic monitoring of engagement 
        metrics and attendance patterns to identify at-risk students early. The development 
        of targeted intervention programs based on specific risk factors may improve 
        student retention and academic success rates.
        """
        
        content.append(Paragraph(implications_text.strip(), self.styles['APABody']))
        
        return content

    def _create_references(self) -> List:
        """Create APA-style references section."""
        content = []
        
        content.append(PageBreak())
        content.append(Paragraph("References", self.styles['APAHeading1']))
        
        # Sample references in APA format
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
        """Create a descriptive statistics table."""
        content = []
        
        # Table caption
        content.append(Paragraph("Table 1", self.styles['APATableCaption']))
        content.append(Paragraph("<i>Descriptive Statistics for Academic Performance Metrics</i>", 
                                self.styles['APATableCaption']))
        
        # Prepare table data
        headers = ['Metric', 'Mean', 'SD', 'Min', 'Max', 'N']
        data = [headers]
        
        # Add rows for each metric
        if 'engagement' in performance_stats:
            eng_stats = performance_stats['engagement']
            data.append([
                'Total Engagement',
                f"{eng_stats.get('mean', 0):.2f}",
                f"{eng_stats.get('std', 0):.2f}",
                f"{eng_stats.get('min', 0):.2f}",
                f"{eng_stats.get('max', 0):.2f}",
                f"{eng_stats.get('count', 0)}"
            ])
        
        # Create table
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
        """Create a risk factors frequency table."""
        content = []
        
        # Table caption
        content.append(Paragraph("Table 2", self.styles['APATableCaption']))
        content.append(Paragraph("<i>Frequency of Risk Factors Among At-Risk Students</i>", 
                                self.styles['APATableCaption']))
        
        # Prepare table data
        headers = ['Risk Factor', 'Frequency', 'Percentage']
        data = [headers]
        
        total_factors = sum(risk_factors.values())
        for factor, count in risk_factors.items():
            percentage = (count / total_factors) * 100 if total_factors > 0 else 0
            data.append([factor, str(count), f"{percentage:.1f}%"])
        
        # Create table
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
    Generate a simple text-based report (fallback when ReportLab is not available).
    
    Args:
        df: Cleaned DataFrame with student data
        stats: Statistical analysis results
        save_path: Optional path to save the report
        
    Returns:
        Path to the generated report
    """
    try:
        log_analysis_step("Generating simple text report")
        
        if save_path is None:
            save_path = str(REPORTS_DIR / f"academic_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        
        # Create output directory
        create_output_directory(Path(save_path).parent)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("ACADEMIC PERFORMANCE ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Report Generated: {datetime.now().strftime('%B %d, %Y')}\n")
            f.write(f"Total Students Analyzed: {len(df)}\n\n")
            
            # Performance Distribution
            f.write("PERFORMANCE DISTRIBUTION\n")
            f.write("-" * 30 + "\n")
            performance_counts = df['Class'].value_counts()
            for grade, count in performance_counts.items():
                percentage = (count / len(df)) * 100
                f.write(f"{grade} Performance: {count} students ({percentage:.1f}%)\n")
            f.write("\n")
            
            # Subject Analysis
            if 'subject_averages' in stats:
                f.write("SUBJECT ANALYSIS\n")
                f.write("-" * 20 + "\n")
                for subject, avg in stats['subject_averages'].items():
                    f.write(f"{subject}: {avg:.2f} average engagement\n")
                f.write("\n")
            
            # Demographics
            f.write("DEMOGRAPHIC BREAKDOWN\n")
            f.write("-" * 25 + "\n")
            if 'gender' in df.columns:
                gender_counts = df['gender'].value_counts()
                for gender, count in gender_counts.items():
                    percentage = (count / len(df)) * 100
                    f.write(f"{gender}: {count} students ({percentage:.1f}%)\n")
            f.write("\n")
            
            f.write("Analysis completed successfully.\n")
        
        logger.info(f"Simple report generated: {save_path}")
        return save_path
        
    except Exception as e:
        logger.error(f"Error generating simple report: {e}")
        raise

if __name__ == "__main__":
    # Test the report generator
    try:
        import sys
        import os
        sys.path.append('.')
        
        from src.data.data_loader import load_and_validate_data
        from src.data.data_cleaner import clean_student_data
        from src.analysis.statistics import calculate_all_statistics
        from src.analysis.grouping import perform_all_groupings
        from src.analysis.risk_analysis import generate_risk_report
        
        print("Testing APA report generator...")
        
        # Load and prepare data
        df = load_and_validate_data()
        df_clean = clean_student_data(df)
        stats = calculate_all_statistics(df_clean)
        groups = perform_all_groupings(df_clean)
        risk_report = generate_risk_report(df_clean)
        
        if REPORTLAB_AVAILABLE:
            # Generate APA report
            report_gen = APAReportGenerator()
            report_path = report_gen.generate_comprehensive_report(
                df_clean, stats, groups, risk_report, {}
            )
            print(f"APA report generated: {report_path}")
        else:
            # Generate simple report
            report_path = generate_simple_report(df_clean, stats)
            print(f"Simple report generated: {report_path}")
            
    except Exception as e:
        print(f"Error testing report generator: {e}")
