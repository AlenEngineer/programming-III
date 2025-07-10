"""
Academic Data Analysis System - Main Orchestrator

This module coordinates the complete academic data analysis pipeline:
1. Data loading and validation
2. Data cleaning and preprocessing
3. Statistical analysis and grouping
4. Risk identification and assessment
5. Visualization generation
6. APA-style report generation

The system provides a modular, team-friendly approach to analyzing student
performance data with comprehensive reporting capabilities.
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

# Import configuration
from config import (
    DATA_FILE_PATH, OUTPUT_DIR, CHARTS_DIR, REPORTS_DIR,
    REPORT_TITLE, REPORT_AUTHOR, INSTITUTION
)

# Import data modules
from src.data.data_loader import load_and_validate_data, get_data_info
from src.data.data_cleaner import clean_student_data, get_data_quality_report

# Import analysis modules
from src.analysis.statistics import calculate_all_statistics
from src.analysis.grouping import group_by_demographics, perform_all_groupings
from src.analysis.risk_analysis import generate_risk_report, identify_intervention_priorities

# Import visualization modules
from src.visualization.charts import create_all_visualizations

# Import reporting modules
from src.reports.apa_report import generate_apa_report

# Import utilities
from src.utils.helpers import setup_logging, log_analysis_step


def run_complete_analysis(data_file: Optional[str] = None, 
                         generate_charts: bool = True,
                         generate_report: bool = True,
                         verbose: bool = True) -> Dict[str, Any]:
    """
    Run the complete academic data analysis pipeline.
    
    Args:
        data_file: Path to the data file. If None, uses default from config.
        generate_charts: Whether to generate visualization charts.
        generate_report: Whether to generate the APA-style PDF report.
        verbose: Whether to enable verbose logging output.
    
    Returns:
        Dict containing all analysis results and file paths.
    """
    # Setup logging
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
        # Step 1: Load and validate data
        log_analysis_step("Loading and validating data")
        df = load_and_validate_data(data_file)
        data_info = get_data_info(df)
        results['data_info'] = data_info
        logger.info(f"Loaded {len(df)} student records with {len(df.columns)} features")
        
        # Step 2: Clean and preprocess data
        log_analysis_step("Cleaning and preprocessing data")
        df_clean = clean_student_data(df)
        quality_report = get_data_quality_report(df_clean)
        logger.info(f"Data cleaning completed. Quality score: {quality_report.get('overall_quality_score', 'N/A')}")
        
        # Step 3: Statistical analysis
        log_analysis_step("Performing statistical analysis")
        statistics = calculate_all_statistics(df_clean)
        results['statistics'] = statistics
        logger.info("Statistical analysis completed")
        
        # Step 4: Demographic and engagement analysis
        log_analysis_step("Analyzing performance by demographics")
        demographic_analysis = group_by_demographics(df_clean)
        results['demographic_analysis'] = demographic_analysis
        
        log_analysis_step("Analyzing engagement patterns")
        engagement_analysis = perform_all_groupings(df_clean)
        results['engagement_analysis'] = engagement_analysis
        logger.info("Demographic and engagement analysis completed")
        
        # Step 5: Risk analysis
        log_analysis_step("Generating risk assessment report")
        risk_report = generate_risk_report(df_clean)
        results['risk_report'] = risk_report
        
        log_analysis_step("Identifying intervention priorities")
        intervention_priorities = identify_intervention_priorities(df_clean)
        results['intervention_priorities'] = intervention_priorities
        logger.info(f"Risk analysis completed. {len(risk_report['at_risk_students'])} at-risk students identified")
        
        # Step 6: Generate visualizations
        if generate_charts:
            log_analysis_step("Generating visualization charts")
            chart_files = create_all_visualizations(df_clean, statistics, risk_report)
            results['chart_files'] = chart_files
            logger.info(f"Generated {len(chart_files)} visualization charts")
        
        # Step 7: Generate APA-style report
        if generate_report:
            log_analysis_step("Generating APA-style PDF report")
            report_file = generate_apa_report(
                df_clean, statistics, demographic_analysis, 
                engagement_analysis, risk_report
            )
            results['report_file'] = report_file
            logger.info(f"APA-style report generated: {report_file}")
        
        results['success'] = True
        logger.info("Complete analysis pipeline executed successfully")
        
    except Exception as e:
        error_msg = f"Error in analysis pipeline: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        results['errors'].append(error_msg)
        results['success'] = False
    
    return results


def print_analysis_summary(results: Dict[str, Any]) -> None:
    """Print a summary of the analysis results."""
    print("\n" + "="*80)
    print("ACADEMIC DATA ANALYSIS - EXECUTION SUMMARY")
    print("="*80)
    
    if not results['success']:
        print("❌ Analysis failed!")
        for error in results['errors']:
            print(f"   Error: {error}")
        return
    
    print("✅ Analysis completed successfully!")
    
    # Data information
    if results['data_info']:
        data_info = results['data_info']
        print(f"\n📊 Dataset Information:")
        print(f"   • Students: {data_info['shape'][0]:,}")
        print(f"   • Features: {data_info['shape'][1]}")
        print(f"   • Data quality: {data_info.get('quality_score', 'N/A')}")
    
    # Statistical summary
    if results['statistics']:
        stats = results['statistics']
        print(f"\n📈 Statistical Summary:")
        if 'overall_stats' in stats:
            overall = stats['overall_stats']
            print(f"   • Average engagement: {overall.get('mean', 0):.2f}")
            print(f"   • Standard deviation: {overall.get('std', 0):.2f}")
    
    # Risk analysis summary
    if results['risk_report']:
        risk = results['risk_report']
        print(f"\n⚠️  Risk Analysis:")
        print(f"   • At-risk students: {len(risk.get('at_risk_students', []))}")
        print(f"   • High absence students: {len(risk.get('high_absence_students', []))}")
        print(f"   • Low participation students: {len(risk.get('low_participation_students', []))}")
    
    # Output files
    print(f"\n📁 Generated Files:")
    if results['chart_files']:
        print(f"   • Charts: {len(results['chart_files'])} files in {CHARTS_DIR}")
    if results['report_file']:
        print(f"   • Report: {results['report_file']}")
    
    print(f"\n📍 Output directory: {OUTPUT_DIR}")
    print("="*80)


def main():
    """
    Main entry point for the Academic Data Analysis System.
    
    Executes the complete analysis pipeline and displays results summary.
    """
    print("🎓 Academic Data Analysis System")
    print(f"📊 Analyzing data from: {DATA_FILE_PATH}")
    print(f"🏫 Institution: {INSTITUTION}")
    print(f"👥 Authors: {REPORT_AUTHOR}")
    
    # Check if data file exists
    if not DATA_FILE_PATH.exists():
        print(f"\n❌ Error: Data file not found at {DATA_FILE_PATH}")
        print("Please ensure the data file is in the correct location.")
        return
    
    # Run the complete analysis
    try:
        results = run_complete_analysis(
            data_file=None,  # Use default from config
            generate_charts=True,
            generate_report=True,
            verbose=True
        )
        
        # Print summary
        print_analysis_summary(results)
        
        if results['success']:
            print(f"\n🎉 Analysis complete! Check the output directory: {OUTPUT_DIR}")
        else:
            print(f"\n💥 Analysis failed. Check the logs for details.")
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Analysis interrupted by user.")
    except Exception as e:
        print(f"\n💥 Unexpected error: {str(e)}")
        print("Check the logs for detailed error information.")


if __name__ == "__main__":
    main()
