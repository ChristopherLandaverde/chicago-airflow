#!/usr/bin/env python3
"""
Test time series data preparation without heavy ML dependencies
"""

import sys
sys.path.append('/opt/airflow/dags/src')

def test_time_series_data_prep():
    """Test time series data preparation"""
    print("=== Testing Time Series Data Preparation ===")
    
    try:
        from ml.time_series_models import CrimeTimeSeriesForecaster
        
        # Initialize forecaster
        forecaster = CrimeTimeSeriesForecaster()
        print("✓ CrimeTimeSeriesForecaster initialized")
        
        # Test database connection
        with forecaster.engine.connect() as conn:
            result = conn.execute("SELECT COUNT(*) FROM ml_features.crime_features")
            count = result.fetchone()[0]
            print(f"✓ Connected to features database: {count} records available")
        
        # Load time series data
        print("Loading time series data...")
        ts_data = forecaster.load_time_series_data()
        print(f"✓ Loaded time series data: {len(ts_data)} daily observations")
        print(f"  Date range: {ts_data['date'].min()} to {ts_data['date'].max()}")
        print(f"  Crime count range: {ts_data['crime_count'].min()} to {ts_data['crime_count'].max()}")
        print(f"  Average daily crimes: {ts_data['crime_count'].mean():.1f}")
        
        # Test specific crime type
        print("\nTesting violent crime time series...")
        violent_ts = forecaster.load_time_series_data(crime_type='BATTERY')
        print(f"✓ Loaded BATTERY time series: {len(violent_ts)} daily observations")
        print(f"  Average daily BATTERY crimes: {violent_ts['crime_count'].mean():.1f}")
        
        # Show sample data
        print(f"\nSample time series data:")
        print(ts_data.head(10).to_string())
        
        return True
        
    except Exception as e:
        print(f"✗ Time series data preparation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_spatial_data_prep():
    """Test spatial data preparation"""
    print("\n=== Testing Spatial Data Preparation ===")
    
    try:
        from ml.spatial_clustering import CrimeSpatialAnalyzer
        
        # Initialize analyzer
        analyzer = CrimeSpatialAnalyzer()
        print("✓ CrimeSpatialAnalyzer initialized")
        
        # Load spatial data
        print("Loading spatial data...")
        spatial_data = analyzer.load_spatial_data()
        print(f"✓ Loaded spatial data: {len(spatial_data)} records with coordinates")
        
        # Show coordinate bounds
        print(f"  Latitude range: {spatial_data['latitude'].min():.4f} to {spatial_data['latitude'].max():.4f}")
        print(f"  Longitude range: {spatial_data['longitude'].min():.4f} to {spatial_data['longitude'].max():.4f}")
        
        # Test crime type filtering
        print("\nTesting violent crime spatial data...")
        violent_spatial = analyzer.load_spatial_data(crime_type='BATTERY')
        print(f"✓ Loaded BATTERY spatial data: {len(violent_spatial)} records")
        
        # Test time period filtering
        print("\nTesting recent data filtering...")
        recent_spatial = analyzer.load_spatial_data(time_period='recent_month')
        print(f"✓ Loaded recent month data: {len(recent_spatial)} records")
        
        return True
        
    except Exception as e:
        print(f"✗ Spatial data preparation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing ML Pipeline Data Preparation")
    print("=" * 50)
    
    tests = [
        test_time_series_data_prep,
        test_spatial_data_prep
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 50)
    print(f"=== Test Summary ===")
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {sum(results)}")
    print(f"Failed: {len(tests) - sum(results)}")
    
    if all(results):
        print("🎉 All ML data preparation tests passed!")
        print("\nData is ready for:")
        print("- Time series forecasting (ARIMA/Prophet)")
        print("- Spatial clustering (K-means/DBSCAN)")
        print("- Interactive dashboard development")
    else:
        print("❌ Some tests failed")
    
    exit(0 if all(results) else 1)