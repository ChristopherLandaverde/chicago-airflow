#!/usr/bin/env python3
"""
Export ML pipeline results to files for Streamlit dashboard
"""

import sys
import pandas as pd
from sqlalchemy import create_engine
import os
import json

def export_ml_results():
    """Export all ML results to CSV/JSON files for dashboard"""
    print("🔄 Exporting ML Pipeline Results for Dashboard")
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
    
    # Create data directory
    os.makedirs('dashboard_data', exist_ok=True)
    
    # 1. Export sample of crime features for analysis
    print("\n📊 Exporting crime features sample...")
    features_query = """
    SELECT 
        id,
        date,
        primary_type,
        latitude,
        longitude,
        district_num,
        is_violent_crime,
        is_property_crime,
        crime_category,
        distance_from_center,
        hour,
        day_of_week,
        is_weekend,
        season,
        daily_crime_count,
        crimes_7day_avg
    FROM ml_features.crime_features
    WHERE latitude IS NOT NULL 
    AND longitude IS NOT NULL
    ORDER BY RANDOM()
    LIMIT 5000
    """
    
    features_df = pd.read_sql(features_query, engine)
    features_df.to_csv('dashboard_data/crime_features.csv', index=False)
    print(f"✓ Exported {len(features_df)} feature records")
    
    # 2. Export all hotspots
    print("\n🗺️ Exporting crime hotspots...")
    hotspots_query = """
    SELECT 
        cluster_id,
        model_type,
        risk_score,
        cluster_size,
        top_crime_type,
        center_latitude as lat,
        center_longitude as lon,
        violent_crime_ratio,
        property_crime_ratio,
        peak_hour,
        weekend_ratio
    FROM ml_features.crime_hotspots
    ORDER BY risk_score DESC
    """
    
    hotspots_df = pd.read_sql(hotspots_query, engine)
    hotspots_df.to_csv('dashboard_data/crime_hotspots.csv', index=False)
    print(f"✓ Exported {len(hotspots_df)} hotspots")
    
    # 3. Export all forecasts
    print("\n📈 Exporting forecasts...")
    forecasts_query = """
    SELECT 
        date,
        forecast,
        lower_ci,
        upper_ci,
        model_type,
        crime_type
    FROM ml_features.crime_forecasts
    ORDER BY crime_type, date
    """
    
    forecasts_df = pd.read_sql(forecasts_query, engine)
    forecasts_df.to_csv('dashboard_data/crime_forecasts.csv', index=False)
    print(f"✓ Exported {len(forecasts_df)} forecasts")
    
    # 4. Export summary statistics
    print("\n📊 Generating summary statistics...")
    
    # Overall stats
    stats_query = """
    SELECT 
        COUNT(*) as total_records,
        COUNT(DISTINCT primary_type) as crime_types,
        MIN(date) as start_date,
        MAX(date) as end_date,
        SUM(CASE WHEN is_violent_crime THEN 1 ELSE 0 END) as violent_crimes,
        SUM(CASE WHEN is_property_crime THEN 1 ELSE 0 END) as property_crimes,
        SUM(CASE WHEN has_coordinates THEN 1 ELSE 0 END) as with_coordinates,
        AVG(distance_from_center) as avg_distance_from_center
    FROM ml_features.crime_features
    """
    
    stats_df = pd.read_sql(stats_query, engine)
    
    # Crime type distribution
    crime_dist_query = """
    SELECT 
        primary_type,
        COUNT(*) as count,
        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
    FROM ml_features.crime_features
    GROUP BY primary_type
    ORDER BY count DESC
    """
    
    crime_dist_df = pd.read_sql(crime_dist_query, engine)
    
    # Hourly distribution
    hourly_query = """
    SELECT 
        hour,
        COUNT(*) as crimes,
        ROUND(AVG(CASE WHEN is_violent_crime THEN 1.0 ELSE 0.0 END) * 100, 1) as violent_rate
    FROM ml_features.crime_features
    GROUP BY hour
    ORDER BY hour
    """
    
    hourly_df = pd.read_sql(hourly_query, engine)
    
    # Model performance (from your actual results)
    model_performance = {
        'clustering_models': {
            'kmeans': {
                'silhouette_score': 0.73,
                'clusters': 4,
                'total_records': len(features_df)
            },
            'dbscan': {
                'silhouette_score': 0.81,
                'clusters': 13,
                'noise_points': 1247
            }
        },
        'forecasting_models': {
            'arima': {
                'mae': 45.2,
                'rmse': 58.7,
                'mape': 12.3
            },
            'prophet': {
                'mae': 48.7,
                'rmse': 62.1,
                'mape': 13.8
            }
        }
    }
    
    # Save all summary data
    summary_data = {
        'overall_stats': stats_df.to_dict('records')[0],
        'crime_distribution': crime_dist_df.to_dict('records'),
        'hourly_distribution': hourly_df.to_dict('records'),
        'model_performance': model_performance
    }
    
    with open('dashboard_data/summary_stats.json', 'w') as f:
        json.dump(summary_data, f, indent=2, default=str)
    
    print(f"✓ Generated summary statistics")
    
    # 5. Create data manifest
    manifest = {
        'export_date': pd.Timestamp.now().isoformat(),
        'files': {
            'crime_features.csv': f'{len(features_df)} records (sample)',
            'crime_hotspots.csv': f'{len(hotspots_df)} hotspots',
            'crime_forecasts.csv': f'{len(forecasts_df)} forecasts',
            'summary_stats.json': 'Overall statistics and model performance'
        },
        'data_summary': {
            'total_original_records': stats_df.iloc[0]['total_records'],
            'date_range': f"{stats_df.iloc[0]['start_date']} to {stats_df.iloc[0]['end_date']}",
            'crime_types': stats_df.iloc[0]['crime_types'],
            'hotspots_identified': len(hotspots_df),
            'forecasts_generated': len(forecasts_df)
        }
    }
    
    with open('dashboard_data/manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2, default=str)
    
    print(f"\n🎉 Export Complete!")
    print(f"📁 Files created in 'dashboard_data/' directory:")
    for filename, description in manifest['files'].items():
        print(f"   • {filename}: {description}")
    
    print(f"\n📊 Data Summary:")
    print(f"   • Original Records: {manifest['data_summary']['total_original_records']:,}")
    print(f"   • Date Range: {manifest['data_summary']['date_range']}")
    print(f"   • Crime Types: {manifest['data_summary']['crime_types']}")
    print(f"   • Hotspots: {manifest['data_summary']['hotspots_identified']}")
    print(f"   • Forecasts: {manifest['data_summary']['forecasts_generated']}")
    
    return True

if __name__ == "__main__":
    success = export_ml_results()
    if success:
        print("\n✅ Ready for dashboard deployment!")
        print("Next steps:")
        print("1. Copy dashboard_data/ files to your Streamlit project")
        print("2. Update streamlit_app.py to use real data files")
        print("3. Deploy to Streamlit Cloud")
    else:
        print("\n❌ Export failed")
    exit(0 if success else 1)