from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import sys
import os

# Add the src directory to Python path
sys.path.append('/opt/airflow/dags/src')

from ml.feature_engineering import CrimeFeatureEngineer
from ml.time_series_models import CrimeTimeSeriesForecaster
from ml.spatial_clustering import CrimeSpatialAnalyzer

# Default arguments
default_args = {
    'owner': 'crime-analytics',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Create DAG
dag = DAG(
    'ml_pipeline',
    default_args=default_args,
    description='Machine Learning Pipeline for Crime Analytics',
    schedule_interval='@weekly',  # Run weekly to retrain models
    catchup=False,
    tags=['ml', 'crime-analytics', 'forecasting', 'clustering']
)

def run_feature_engineering(**context):
    """Run feature engineering pipeline"""
    print("=== Starting Feature Engineering ===")
    
    engineer = CrimeFeatureEngineer()
    
    # Create all features
    features_df = engineer.create_all_features()
    
    # Save to database
    result = engineer.save_features_to_db(features_df)
    
    print(f"Feature engineering completed: {result}")
    return result

def run_time_series_forecasting(**context):
    """Run time series forecasting for overall crime trends"""
    print("=== Starting Time Series Forecasting ===")
    
    forecaster = CrimeTimeSeriesForecaster()
    
    # Run forecasting for overall crime trends
    results = forecaster.run_full_forecasting_pipeline(
        crime_type=None,  # All crimes
        days_ahead=30
    )
    
    print("Time series forecasting completed")
    return results

def run_violent_crime_forecasting(**context):
    """Run time series forecasting specifically for violent crimes"""
    print("=== Starting Violent Crime Forecasting ===")
    
    forecaster = CrimeTimeSeriesForecaster()
    
    # Get violent crime types
    violent_crimes = ['HOMICIDE', 'CRIMINAL SEXUAL ASSAULT', 'ROBBERY', 'ASSAULT', 'BATTERY']
    
    results = {}
    for crime_type in violent_crimes:
        try:
            print(f"Forecasting for {crime_type}...")
            crime_results = forecaster.run_full_forecasting_pipeline(
                crime_type=crime_type,
                days_ahead=30
            )
            results[crime_type] = crime_results
        except Exception as e:
            print(f"Error forecasting {crime_type}: {e}")
            results[crime_type] = {'error': str(e)}
    
    print("Violent crime forecasting completed")
    return results

def run_spatial_clustering_analysis(**context):
    """Run spatial clustering analysis for hotspot identification"""
    print("=== Starting Spatial Clustering Analysis ===")
    
    analyzer = CrimeSpatialAnalyzer()
    
    # Run clustering for all crimes (recent year)
    results = analyzer.run_full_spatial_analysis(
        crime_type=None,
        time_period='recent_year'
    )
    
    print("Spatial clustering analysis completed")
    return results

def run_violent_crime_clustering(**context):
    """Run spatial clustering specifically for violent crimes"""
    print("=== Starting Violent Crime Clustering ===")
    
    analyzer = CrimeSpatialAnalyzer()
    
    # Run clustering for violent crimes only
    results = analyzer.run_full_spatial_analysis(
        crime_type=None,  # We'll filter by violent crimes in the query
        time_period='recent_year'
    )
    
    print("Violent crime clustering completed")
    return results

def generate_model_performance_report(**context):
    """Generate performance report for all ML models"""
    print("=== Generating ML Model Performance Report ===")
    
    from sqlalchemy import create_engine, text
    import pandas as pd
    
    # Database connection
    db_config = {
        'host': os.getenv('DB_HOST', 'postgres'),
        'port': os.getenv('DB_PORT', '5432'),
        'database': os.getenv('DB_NAME', 'crime_analytics'),
        'user': os.getenv('DB_USER', 'airflow'),
        'password': os.getenv('DB_PASSWORD', 'airflow')
    }
    
    conn_string = (
        f"postgresql://{db_config['user']}:{db_config['password']}"
        f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    )
    engine = create_engine(conn_string)
    
    report = {
        'generated_at': datetime.now(),
        'feature_count': 0,
        'forecast_count': 0,
        'cluster_count': 0,
        'hotspot_count': 0
    }
    
    try:
        # Count features
        feature_count = pd.read_sql(
            "SELECT COUNT(*) as count FROM ml_features.crime_features", 
            engine
        ).iloc[0]['count']
        report['feature_count'] = feature_count
        
        # Count forecasts
        forecast_count = pd.read_sql(
            "SELECT COUNT(*) as count FROM ml_features.crime_forecasts", 
            engine
        ).iloc[0]['count']
        report['forecast_count'] = forecast_count
        
        # Count clusters
        cluster_count = pd.read_sql(
            "SELECT COUNT(DISTINCT cluster_id) as count FROM ml_features.crime_clusters", 
            engine
        ).iloc[0]['count']
        report['cluster_count'] = cluster_count
        
        # Count hotspots
        hotspot_count = pd.read_sql(
            "SELECT COUNT(*) as count FROM ml_features.crime_hotspots", 
            engine
        ).iloc[0]['count']
        report['hotspot_count'] = hotspot_count
        
    except Exception as e:
        print(f"Error generating report: {e}")
        report['error'] = str(e)
    
    print(f"ML Pipeline Report:")
    print(f"  Features: {report['feature_count']}")
    print(f"  Forecasts: {report['forecast_count']}")
    print(f"  Clusters: {report['cluster_count']}")
    print(f"  Hotspots: {report['hotspot_count']}")
    
    return report

# Define tasks
feature_engineering_task = PythonOperator(
    task_id='feature_engineering',
    python_callable=run_feature_engineering,
    dag=dag,
)

time_series_forecasting_task = PythonOperator(
    task_id='time_series_forecasting',
    python_callable=run_time_series_forecasting,
    dag=dag,
)

violent_crime_forecasting_task = PythonOperator(
    task_id='violent_crime_forecasting',
    python_callable=run_violent_crime_forecasting,
    dag=dag,
)

spatial_clustering_task = PythonOperator(
    task_id='spatial_clustering_analysis',
    python_callable=run_spatial_clustering_analysis,
    dag=dag,
)

violent_crime_clustering_task = PythonOperator(
    task_id='violent_crime_clustering',
    python_callable=run_violent_crime_clustering,
    dag=dag,
)

performance_report_task = PythonOperator(
    task_id='generate_performance_report',
    python_callable=generate_model_performance_report,
    dag=dag,
)

# Define task dependencies
feature_engineering_task >> [time_series_forecasting_task, spatial_clustering_task]
time_series_forecasting_task >> violent_crime_forecasting_task
spatial_clustering_task >> violent_crime_clustering_task
[violent_crime_forecasting_task, violent_crime_clustering_task] >> performance_report_task