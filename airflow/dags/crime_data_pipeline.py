from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
import sys
import os

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
    'crime_data_pipeline',
    default_args=default_args,
    description='Chicago crime data pipeline with analytics',
    schedule_interval='@daily',
    catchup=False,
    tags=['crime-data', 'analytics'],
)

def extract_crime_data(**context):
    """Extract crime data from Chicago API"""
    from src.extractors.dot_api import DOTAPIExtractor
    
    extractor = DOTAPIExtractor()
    execution_date = context['execution_date']
    
    # Extract data for the execution date
    data = extractor.extract_daily_data(execution_date)
    
    return f"Extracted {len(data)} records for {execution_date.date()}"

def load_raw_data(**context):
    """Load raw data into PostgreSQL"""
    from src.loaders.postgres_loader import PostgresLoader
    
    loader = PostgresLoader()
    execution_date = context['execution_date']
    
    # Load data from extraction
    result = loader.load_daily_data(execution_date)
    
    return f"Loaded data for {execution_date.date()}: {result}"

def transform_staging_data(**context):
    """Transform raw data into staging tables"""
    from src.transformers.sql_transformer import SQLTransformer
    
    transformer = SQLTransformer()
    result = transformer.create_staging_tables()
    
    return f"Staging transformation completed: {result}"

def transform_analytics_data(**context):
    """Transform staging data into analytics tables"""
    from src.transformers.sql_transformer import SQLTransformer
    
    transformer = SQLTransformer()
    result = transformer.create_analytics_tables()
    
    return f"Analytics transformation completed: {result}"

# Extract task
extract_task = PythonOperator(
    task_id='extract_crime_data',
    python_callable=extract_crime_data,
    dag=dag,
)

# Load task
load_task = PythonOperator(
    task_id='load_raw_data',
    python_callable=load_raw_data,
    dag=dag,
)

# SQL transformation tasks
transform_staging = PythonOperator(
    task_id='transform_staging',
    python_callable=transform_staging_data,
    dag=dag,
)

transform_analytics = PythonOperator(
    task_id='transform_analytics', 
    python_callable=transform_analytics_data,
    dag=dag,
)

# Set dependencies
extract_task >> load_task >> transform_staging >> transform_analytics