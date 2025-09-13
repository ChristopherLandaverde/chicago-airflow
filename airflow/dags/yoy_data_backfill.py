from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import sys

# Add src to path for imports
sys.path.append('/opt/airflow/dags')

default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'yoy_data_backfill',
    default_args=default_args,
    description='Backfill multi-year data for YoY analysis',
    schedule_interval=None,  # Manual trigger only
    catchup=False,
    tags=['crime-data', 'yoy-analysis', 'backfill'],
)

def extract_year_data(**context):
    """Extract a full year of data for YoY analysis"""
    from src.extractors.dot_api import DOTAPIExtractor
    
    # Get year from DAG run config
    year = context['dag_run'].conf.get('year', 2023)
    
    extractor = DOTAPIExtractor()
    
    # Extract data for the entire year
    params = {
        '$limit': 50000,  # High limit for full year
        '$where': f"date >= '{year}-01-01T00:00:00.000' AND date < '{year+1}-01-01T00:00:00.000'"
    }
    
    import requests
    try:
        print(f"Extracting full year {year} data...")
        response = requests.get(extractor.base_url, params=params, timeout=300)
        
        if response.status_code == 200:
            data = response.json()
            print(f"Successfully extracted {len(data)} records for year {year}")
            return data
        else:
            print(f"Error extracting year {year}: {response.status_code}")
            return []
            
    except Exception as e:
        print(f"Exception extracting year {year}: {e}")
        return []

def load_year_data(**context):
    """Load full year data into database"""
    from src.loaders.postgres_loader import PostgresLoader
    
    # Get year from DAG run config
    year = context['dag_run'].conf.get('year', 2023)
    
    # Get data from previous task
    data = extract_year_data(**context)
    
    if not data:
        return f"No data to load for year {year}"
    
    # Load data
    loader = PostgresLoader()
    result = loader.load_batch_data(data, f"year_{year}")
    
    return result

def create_yoy_analytics(**context):
    """Create comprehensive YoY analytics after loading data"""
    from src.transformers.sql_transformer import SQLTransformer
    
    transformer = SQLTransformer()
    
    # Run staging transformation
    print("Creating staging tables...")
    staging_result = transformer.create_staging_tables()
    print(f"Staging: {staging_result}")
    
    # Run analytics transformation (includes YoY analysis)
    print("Creating analytics tables...")
    analytics_result = transformer.create_analytics_tables()
    print(f"Analytics: {analytics_result}")
    
    # Get summary
    summary = transformer.get_analytics_summary()
    print(f"Summary: {summary}")
    
    return f"YoY analytics completed. {analytics_result}"

def generate_yoy_report(**context):
    """Generate YoY analysis report"""
    from src.transformers.sql_transformer import SQLTransformer
    
    transformer = SQLTransformer()
    
    # Get YoY insights
    yoy_insights_sql = """
    -- Top YoY changes by crime type
    SELECT 
        'Top Growing Crime Types' as analysis_type,
        primary_type,
        yoy_change_percent,
        current_year_incidents,
        prev_year_incidents
    FROM analytics.yoy_crime_type_analysis 
    WHERE yoy_change_percent IS NOT NULL
    ORDER BY yoy_change_percent DESC 
    LIMIT 5
    
    UNION ALL
    
    -- Top declining crime types
    SELECT 
        'Top Declining Crime Types' as analysis_type,
        primary_type,
        yoy_change_percent,
        current_year_incidents,
        prev_year_incidents
    FROM analytics.yoy_crime_type_analysis 
    WHERE yoy_change_percent IS NOT NULL
    ORDER BY yoy_change_percent ASC 
    LIMIT 5;
    """
    
    try:
        with transformer.engine.connect() as conn:
            result = conn.execute(text(yoy_insights_sql))
            insights = result.fetchall()
        
        report = "YoY Analysis Report:\n"
        for row in insights:
            report += f"- {row.analysis_type}: {row.primary_type} ({row.yoy_change_percent}% change)\n"
        
        print(report)
        return report
        
    except Exception as e:
        print(f"Error generating YoY report: {e}")
        return f"Error: {str(e)}"

# Tasks
extract_task = PythonOperator(
    task_id='extract_year_data',
    python_callable=extract_year_data,
    dag=dag,
)

load_task = PythonOperator(
    task_id='load_year_data',
    python_callable=load_year_data,
    dag=dag,
)

analytics_task = PythonOperator(
    task_id='create_yoy_analytics',
    python_callable=create_yoy_analytics,
    dag=dag,
)

report_task = PythonOperator(
    task_id='generate_yoy_report',
    python_callable=generate_yoy_report,
    dag=dag,
)

# Dependencies
extract_task >> load_task >> analytics_task >> report_task