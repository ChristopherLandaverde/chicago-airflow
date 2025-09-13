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
    'crime_data_backfill',
    default_args=default_args,
    description='Backfill historical crime data',
    schedule_interval=None,  # Manual trigger only
    catchup=False,
    tags=['crime-data', 'backfill'],
)

def extract_monthly_batch(**context):
    """Extract a month of historical data"""
    from src.extractors.dot_api import DOTAPIExtractor
    
    # Get parameters from DAG run config
    year = context['dag_run'].conf.get('year', 2024)
    month = context['dag_run'].conf.get('month', 1)
    
    extractor = DOTAPIExtractor()
    data = extractor.extract_monthly_data(year, month)
    
    return f"Extracted {len(data)} records for {year}-{month:02d}"

def load_monthly_batch(**context):
    """Load monthly batch into database"""
    from src.loaders.postgres_loader import PostgresLoader
    from src.extractors.dot_api import DOTAPIExtractor
    
    # Get parameters from DAG run config
    year = context['dag_run'].conf.get('year', 2024)
    month = context['dag_run'].conf.get('month', 1)
    
    # Extract data
    extractor = DOTAPIExtractor()
    data = extractor.extract_monthly_data(year, month)
    
    if not data:
        return f"No data to load for {year}-{month:02d}"
    
    # Load data
    loader = PostgresLoader()
    result = loader.load_batch_data(data, f"{year}-{month:02d}")
    
    return result

def transform_after_backfill(**context):
    """Run transformations after backfill"""
    from src.transformers.sql_transformer import SQLTransformer
    
    transformer = SQLTransformer()
    
    # Run staging transformation
    staging_result = transformer.create_staging_tables()
    print(f"Staging: {staging_result}")
    
    # Run analytics transformation
    analytics_result = transformer.create_analytics_tables()
    print(f"Analytics: {analytics_result}")
    
    # Get summary
    summary = transformer.get_analytics_summary()
    
    return f"Transformations completed. Summary: {summary}"

# Tasks
extract_task = PythonOperator(
    task_id='extract_monthly_batch',
    python_callable=extract_monthly_batch,
    dag=dag,
)

load_task = PythonOperator(
    task_id='load_monthly_batch',
    python_callable=load_monthly_batch,
    dag=dag,
)

transform_task = PythonOperator(
    task_id='transform_after_backfill',
    python_callable=transform_after_backfill,
    dag=dag,
)

# Dependencies
extract_task >> load_task >> transform_task