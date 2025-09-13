#!/usr/bin/env python3
"""
Run full feature engineering pipeline on all Chicago crime data
"""

import sys
sys.path.append('/opt/airflow/dags/src')

def run_full_feature_engineering():
    """Run complete feature engineering on all data"""
    print("=== Running Full Feature Engineering Pipeline ===")
    
    try:
        from ml.feature_engineering import CrimeFeatureEngineer
        
        # Initialize feature engineer
        engineer = CrimeFeatureEngineer()
        print("✓ CrimeFeatureEngineer initialized")
        
        # Check data availability
        with engineer.engine.connect() as conn:
            result = conn.execute("SELECT COUNT(*) FROM raw_crime_data")
            count = result.fetchone()[0]
            print(f"✓ Found {count} records to process")
        
        # Run complete feature engineering pipeline
        print("Starting feature engineering on all data...")
        features_df = engineer.create_all_features()
        
        print(f"✓ Feature engineering completed")
        print(f"  - Total records processed: {len(features_df)}")
        print(f"  - Total features created: {len(features_df.columns)}")
        print(f"  - Date range: {features_df['date'].min()} to {features_df['date'].max()}")
        
        # Save features to database
        print("Saving features to database...")
        save_result = engineer.save_features_to_db(features_df)
        print(f"✓ {save_result}")
        
        # Show sample of created features
        print("\n=== Sample Feature Summary ===")
        print(f"Crime types: {features_df['primary_type'].nunique()} unique types")
        print(f"Violent crimes: {features_df['is_violent_crime'].sum()} ({features_df['is_violent_crime'].mean():.1%})")
        print(f"Property crimes: {features_df['is_property_crime'].sum()} ({features_df['is_property_crime'].mean():.1%})")
        print(f"Records with coordinates: {features_df['has_coordinates'].sum()} ({features_df['has_coordinates'].mean():.1%})")
        print(f"Weekend crimes: {features_df['is_weekend'].sum()} ({features_df['is_weekend'].mean():.1%})")
        print(f"Night crimes: {features_df['is_night'].sum()} ({features_df['is_night'].mean():.1%})")
        
        # Show temporal distribution
        print(f"\nTemporal Distribution:")
        print(f"  - Hours: {features_df['hour'].min()}-{features_df['hour'].max()}")
        print(f"  - Months: {features_df['month'].min()}-{features_df['month'].max()}")
        print(f"  - Seasons: {features_df['season'].value_counts().to_dict()}")
        
        # Show spatial distribution
        print(f"\nSpatial Distribution:")
        print(f"  - Districts: {features_df['district_num'].nunique()} unique")
        print(f"  - Wards: {features_df['ward_num'].nunique()} unique")
        print(f"  - Community areas: {features_df['community_area_num'].nunique()} unique")
        print(f"  - Distance from center: {features_df['distance_from_center'].min():.1f} - {features_df['distance_from_center'].max():.1f} miles")
        
        return True
        
    except Exception as e:
        print(f"✗ Feature engineering failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_full_feature_engineering()
    if success:
        print("\n🎉 Full Feature Engineering Pipeline completed successfully!")
        print("\nNext steps:")
        print("1. Run time series forecasting models")
        print("2. Run spatial clustering analysis")
        print("3. Generate ML model performance reports")
    else:
        print("\n❌ Feature Engineering Pipeline failed")
    exit(0 if success else 1)