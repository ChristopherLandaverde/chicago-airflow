#!/usr/bin/env python3
"""
Check the complete ML pipeline status and results
"""

import sys
import pandas as pd
from sqlalchemy import create_engine
import os

def check_pipeline_status():
    """Check status of all ML pipeline components"""
    print("🔍 CHICAGO CRIME ML PIPELINE STATUS CHECK")
    print("=" * 60)
    
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
    
    # Check each pipeline component
    components = {
        'Raw Data': 'raw_crime_data',
        'Feature Engineering': 'ml_features.crime_features',
        'Spatial Clustering': 'ml_features.crime_clusters',
        'Hotspot Analysis': 'ml_features.crime_hotspots',
        'Time Series Forecasts': 'ml_features.crime_forecasts'
    }
    
    print("\n📊 PIPELINE COMPONENT STATUS:")
    print("-" * 40)
    
    for component, table in components.items():
        try:
            result = pd.read_sql(f"SELECT COUNT(*) as count FROM {table}", engine)
            count = result.iloc[0]['count']
            status = "✅ COMPLETE" if count > 0 else "❌ EMPTY"
            print(f"{component:<25} {status:<12} ({count:,} records)")
        except Exception as e:
            print(f"{component:<25} ❌ ERROR      (Table not found)")
    
    # Detailed analysis of each component
    print("\n🔬 DETAILED COMPONENT ANALYSIS:")
    print("-" * 40)
    
    # 1. Feature Engineering Analysis
    try:
        features_query = """
        SELECT 
            COUNT(*) as total_records,
            COUNT(DISTINCT primary_type) as crime_types,
            MIN(date) as start_date,
            MAX(date) as end_date,
            SUM(CASE WHEN has_coordinates THEN 1 ELSE 0 END) as with_coords
        FROM ml_features.crime_features
        """
        features_stats = pd.read_sql(features_query, engine).iloc[0]
        
        print(f"\n📈 Feature Engineering:")
        print(f"   Records: {features_stats['total_records']:,}")
        print(f"   Crime Types: {features_stats['crime_types']}")
        print(f"   Date Range: {features_stats['start_date']} to {features_stats['end_date']}")
        print(f"   With Coordinates: {features_stats['with_coords']:,} ({features_stats['with_coords']/features_stats['total_records']:.1%})")
        
    except Exception as e:
        print(f"\n📈 Feature Engineering: ❌ Error - {e}")
    
    # 2. Spatial Clustering Analysis
    try:
        clustering_query = """
        SELECT 
            model_type,
            COUNT(DISTINCT cluster_id) as clusters,
            COUNT(*) as total_records
        FROM ml_features.crime_clusters
        GROUP BY model_type
        """
        clustering_stats = pd.read_sql(clustering_query, engine)
        
        print(f"\n🗺️  Spatial Clustering:")
        for _, row in clustering_stats.iterrows():
            print(f"   {row['model_type']}: {row['clusters']} clusters, {row['total_records']:,} records")
        
        # Top hotspots
        hotspots_query = """
        SELECT model_type, risk_score, cluster_size, top_crime_type
        FROM ml_features.crime_hotspots
        ORDER BY risk_score DESC
        LIMIT 3
        """
        hotspots = pd.read_sql(hotspots_query, engine)
        
        print(f"   Top Hotspots:")
        for _, hotspot in hotspots.iterrows():
            print(f"     {hotspot['model_type']}: Risk {hotspot['risk_score']}, {hotspot['cluster_size']} crimes ({hotspot['top_crime_type']})")
        
    except Exception as e:
        print(f"\n🗺️  Spatial Clustering: ❌ Error - {e}")
    
    # 3. Time Series Forecasting Analysis
    try:
        forecasts_query = """
        SELECT 
            crime_type,
            model_type,
            COUNT(*) as forecast_days,
            MIN(date) as start_date,
            MAX(date) as end_date,
            ROUND(AVG(forecast)::numeric, 1) as avg_forecast
        FROM ml_features.crime_forecasts
        GROUP BY crime_type, model_type
        ORDER BY crime_type, model_type
        """
        forecasts_stats = pd.read_sql(forecasts_query, engine)
        
        print(f"\n📊 Time Series Forecasting:")
        for _, forecast in forecasts_stats.iterrows():
            print(f"   {forecast['crime_type']} ({forecast['model_type']}): {forecast['forecast_days']} days")
            print(f"     Forecast: {forecast['start_date']} to {forecast['end_date']}")
            print(f"     Avg Daily: {forecast['avg_forecast']} crimes")
        
    except Exception as e:
        print(f"\n📊 Time Series Forecasting: ❌ Error - {e}")
    
    # 4. Overall Pipeline Health
    print(f"\n🎯 PIPELINE HEALTH SUMMARY:")
    print("-" * 40)
    
    try:
        # Check data freshness
        freshness_query = """
        SELECT 
            'Raw Data' as component,
            MAX(loaded_at) as last_update
        FROM raw_crime_data
        UNION ALL
        SELECT 
            'Features' as component,
            MAX(created_at) as last_update
        FROM ml_features.crime_features
        UNION ALL
        SELECT 
            'Forecasts' as component,
            MAX(created_at) as last_update
        FROM ml_features.crime_forecasts
        """
        
        freshness = pd.read_sql(freshness_query, engine)
        
        print("Data Freshness:")
        for _, row in freshness.iterrows():
            print(f"   {row['component']}: {row['last_update']}")
        
        # Overall status
        raw_count = pd.read_sql("SELECT COUNT(*) as c FROM raw_crime_data", engine).iloc[0]['c']
        features_count = pd.read_sql("SELECT COUNT(*) as c FROM ml_features.crime_features", engine).iloc[0]['c']
        clusters_count = pd.read_sql("SELECT COUNT(*) as c FROM ml_features.crime_clusters", engine).iloc[0]['c']
        forecasts_count = pd.read_sql("SELECT COUNT(*) as c FROM ml_features.crime_forecasts", engine).iloc[0]['c']
        
        print(f"\nPipeline Completeness:")
        print(f"   ✅ Data Ingestion: {raw_count:,} records")
        print(f"   ✅ Feature Engineering: {features_count:,} records")
        print(f"   ✅ Spatial Clustering: {clusters_count:,} assignments")
        print(f"   ✅ Time Series Forecasting: {forecasts_count:,} predictions")
        
        if all([raw_count > 0, features_count > 0, clusters_count > 0, forecasts_count > 0]):
            print(f"\n🎉 PIPELINE STATUS: FULLY OPERATIONAL")
            print(f"   All components working and producing results!")
        else:
            print(f"\n⚠️  PIPELINE STATUS: PARTIALLY COMPLETE")
            print(f"   Some components may need attention")
        
    except Exception as e:
        print(f"❌ Error checking pipeline health: {e}")

if __name__ == "__main__":
    check_pipeline_status()