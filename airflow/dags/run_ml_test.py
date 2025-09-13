#!/usr/bin/env python3
"""
Simple ML pipeline test script to run inside Docker
"""

import sys
sys.path.append('/opt/airflow/dags/src')

def test_feature_engineering():
    """Test feature engineering with actual data"""
    print("=== Testing Feature Engineering ===")
    
    try:
        from ml.feature_engineering import CrimeFeatureEngineer
        
        # Initialize feature engineer
        engineer = CrimeFeatureEngineer()
        print("✓ CrimeFeatureEngineer initialized")
        
        # Test database connection
        with engineer.engine.connect() as conn:
            result = conn.execute("SELECT COUNT(*) FROM raw_crime_data")
            count = result.fetchone()[0]
            print(f"✓ Connected to database: {count} records available")
        
        # Extract a small sample for testing
        print("Extracting sample data...")
        sample_df = engineer.extract_raw_data()
        print(f"✓ Extracted {len(sample_df)} records")
        
        if len(sample_df) > 0:
            print(f"Sample data columns: {list(sample_df.columns)}")
            print(f"Date range: {sample_df['date'].min()} to {sample_df['date'].max()}")
            
            # Test feature creation on small sample
            print("Creating temporal features...")
            sample_with_temporal = engineer.create_temporal_features(sample_df.head(100))
            print(f"✓ Created temporal features: {len(sample_with_temporal.columns)} total columns")
            
            print("Creating spatial features...")
            sample_with_spatial = engineer.create_spatial_features(sample_with_temporal)
            print(f"✓ Created spatial features: {len(sample_with_spatial.columns)} total columns")
            
            print("Creating crime features...")
            sample_with_crime = engineer.create_crime_features(sample_with_spatial)
            print(f"✓ Created crime features: {len(sample_with_crime.columns)} total columns")
            
            print("\n=== Feature Engineering Test Results ===")
            print(f"Final feature count: {len(sample_with_crime.columns)}")
            print("Sample features created:")
            for col in sample_with_crime.columns:
                if col not in ['id', 'date', 'case_number', 'primary_type', 'description']:
                    print(f"  - {col}")
            
            return True
        else:
            print("✗ No data extracted")
            return False
            
    except Exception as e:
        print(f"✗ Feature engineering test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_feature_engineering()
    if success:
        print("\n🎉 ML Pipeline test completed successfully!")
    else:
        print("\n❌ ML Pipeline test failed")
    exit(0 if success else 1)