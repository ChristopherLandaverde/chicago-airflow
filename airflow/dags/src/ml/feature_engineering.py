import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
import os
from typing import Dict, List, Tuple
import json

class CrimeFeatureEngineer:
    """Feature engineering pipeline for Chicago crime data"""
    
    def __init__(self):
        self.db_config = {
            'host': os.getenv('DB_HOST', 'postgres'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME', 'crime_analytics'),
            'user': os.getenv('DB_USER', 'airflow'),
            'password': os.getenv('DB_PASSWORD', 'airflow')
        }
        
        self.engine = self._create_engine()
    
    def _create_engine(self):
        """Create SQLAlchemy engine"""
        conn_string = (
            f"postgresql://{self.db_config['user']}:{self.db_config['password']}"
            f"@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
        )
        return create_engine(conn_string)
    
    def extract_raw_data(self) -> pd.DataFrame:
        """Extract and parse raw crime data for feature engineering"""
        
        query = """
        SELECT 
            id,
            date,
            record_data,
            loaded_at,
            source
        FROM raw_crime_data
        ORDER BY date
        """
        
        print("Extracting raw crime data...")
        df = pd.read_sql(query, self.engine)
        
        # Parse record data (handle both JSON strings and dict objects)
        print("Parsing records...")
        parsed_records = []
        
        for _, row in df.iterrows():
            try:
                # Handle both JSON string and dict formats
                if isinstance(row['record_data'], str):
                    record = json.loads(row['record_data'])
                else:
                    record = row['record_data']  # Already a dict
                
                parsed_record = {
                    'id': row['id'],
                    'date': pd.to_datetime(row['date']),
                    'case_number': record.get('case_number'),
                    'primary_type': record.get('primary_type'),
                    'description': record.get('description'),
                    'location_description': record.get('location_description'),
                    'arrest': record.get('arrest') == True or record.get('arrest') == 'true',
                    'domestic': record.get('domestic') == True or record.get('domestic') == 'true',
                    'beat': record.get('beat'),
                    'district': record.get('district'),
                    'ward': record.get('ward'),
                    'community_area': record.get('community_area'),
                    'latitude': self._safe_float(record.get('latitude')),
                    'longitude': self._safe_float(record.get('longitude')),
                    'year': self._safe_int(record.get('year')),
                    'loaded_at': row['loaded_at'],
                    'source': row['source']
                }
                parsed_records.append(parsed_record)
                
            except Exception as e:
                print(f"Error parsing record {row['id']}: {e}")
                continue
        
        result_df = pd.DataFrame(parsed_records)
        print(f"Parsed {len(result_df)} records successfully")
        
        return result_df
    
    def _safe_float(self, value) -> float:
        """Safely convert to float"""
        try:
            return float(value) if value is not None else None
        except (ValueError, TypeError):
            return None
    
    def _safe_int(self, value) -> int:
        """Safely convert to int"""
        try:
            return int(value) if value is not None else None
        except (ValueError, TypeError):
            return None
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        
        print("Creating temporal features...")
        
        # Ensure date is datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Basic time features
        df['hour'] = df['date'].dt.hour
        df['day_of_week'] = df['date'].dt.dayofweek  # 0=Monday, 6=Sunday
        df['day_of_month'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['week_of_year'] = df['date'].dt.isocalendar().week
        
        # Categorical time features
        df['is_weekend'] = df['day_of_week'].isin([5, 6])  # Saturday, Sunday
        df['is_night'] = df['hour'].between(22, 5)  # 10 PM to 5 AM
        df['is_rush_hour'] = (
            df['hour'].between(7, 9) |  # Morning rush
            df['hour'].between(17, 19)  # Evening rush
        )
        
        # Season mapping
        season_map = {12: 'Winter', 1: 'Winter', 2: 'Winter',
                     3: 'Spring', 4: 'Spring', 5: 'Spring',
                     6: 'Summer', 7: 'Summer', 8: 'Summer',
                     9: 'Fall', 10: 'Fall', 11: 'Fall'}
        df['season'] = df['month'].map(season_map)
        
        # Time period categories
        df['time_period'] = pd.cut(df['hour'], 
                                  bins=[0, 6, 12, 18, 24], 
                                  labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                                  include_lowest=True)
        
        print(f"Created temporal features for {len(df)} records")
        return df
    
    def create_spatial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create location-based features"""
        
        print("Creating spatial features...")
        
        # Basic geographic features
        df['has_coordinates'] = ~(df['latitude'].isna() | df['longitude'].isna())
        
        # District and beat as categorical
        df['district_num'] = pd.to_numeric(df['district'], errors='coerce')
        df['beat_num'] = pd.to_numeric(df['beat'], errors='coerce')
        df['ward_num'] = pd.to_numeric(df['ward'], errors='coerce')
        df['community_area_num'] = pd.to_numeric(df['community_area'], errors='coerce')
        
        # Location type categories
        indoor_locations = ['RESIDENCE', 'APARTMENT', 'HOUSE', 'HOTEL/MOTEL', 'RESTAURANT']
        outdoor_locations = ['STREET', 'SIDEWALK', 'PARKING LOT/GARAGE(NON.RESID.)', 'ALLEY']
        commercial_locations = ['RETAIL STORE', 'GROCERY FOOD STORE', 'COMMERCIAL / BUSINESS OFFICE']
        
        df['location_type'] = 'Other'
        df.loc[df['location_description'].isin(indoor_locations), 'location_type'] = 'Indoor'
        df.loc[df['location_description'].isin(outdoor_locations), 'location_type'] = 'Outdoor'
        df.loc[df['location_description'].isin(commercial_locations), 'location_type'] = 'Commercial'
        
        # Calculate distance from city center (approximate Chicago downtown)
        chicago_center_lat, chicago_center_lon = 41.8781, -87.6298
        
        def haversine_distance(lat1, lon1, lat2, lon2):
            """Calculate distance between two points on Earth"""
            if pd.isna(lat1) or pd.isna(lon1):
                return None
            
            R = 3959  # Earth's radius in miles
            
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            
            return R * c
        
        df['distance_from_center'] = df.apply(
            lambda row: haversine_distance(
                row['latitude'], row['longitude'],
                chicago_center_lat, chicago_center_lon
            ), axis=1
        )
        
        print(f"Created spatial features for {len(df)} records")
        return df
    
    def create_crime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create crime-specific features"""
        
        print("Creating crime-specific features...")
        
        # Crime severity mapping (based on typical law enforcement classifications)
        violent_crimes = ['HOMICIDE', 'CRIMINAL SEXUAL ASSAULT', 'ROBBERY', 'ASSAULT', 'BATTERY']
        property_crimes = ['BURGLARY', 'THEFT', 'MOTOR VEHICLE THEFT', 'ARSON', 'CRIMINAL DAMAGE']
        drug_crimes = ['NARCOTICS', 'OTHER NARCOTIC VIOLATION']
        
        df['crime_category'] = 'Other'
        df.loc[df['primary_type'].isin(violent_crimes), 'crime_category'] = 'Violent'
        df.loc[df['primary_type'].isin(property_crimes), 'crime_category'] = 'Property'
        df.loc[df['primary_type'].isin(drug_crimes), 'crime_category'] = 'Drug'
        
        # Binary flags for major crime types
        df['is_violent_crime'] = df['primary_type'].isin(violent_crimes)
        df['is_property_crime'] = df['primary_type'].isin(property_crimes)
        df['is_drug_crime'] = df['primary_type'].isin(drug_crimes)
        
        # Arrest likelihood features (based on crime type)
        arrest_rates = df.groupby('primary_type')['arrest'].mean()
        df['crime_type_arrest_rate'] = df['primary_type'].map(arrest_rates)
        
        print(f"Created crime-specific features for {len(df)} records")
        return df
    
    def create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lagged and rolling window features"""
        
        print("Creating lag and rolling features...")
        
        # Sort by date for time series features
        df = df.sort_values('date').reset_index(drop=True)
        
        # Daily crime counts
        daily_counts = df.groupby(df['date'].dt.date).size().reset_index()
        daily_counts.columns = ['date', 'daily_crime_count']
        daily_counts['date'] = pd.to_datetime(daily_counts['date'])
        
        # Create rolling averages
        daily_counts['crimes_7day_avg'] = daily_counts['daily_crime_count'].rolling(7, min_periods=1).mean()
        daily_counts['crimes_30day_avg'] = daily_counts['daily_crime_count'].rolling(30, min_periods=1).mean()
        
        # Merge back to main dataframe
        df['date_only'] = df['date'].dt.date
        df['date_only'] = pd.to_datetime(df['date_only'])
        
        df = df.merge(daily_counts[['date', 'daily_crime_count', 'crimes_7day_avg', 'crimes_30day_avg']], 
                     left_on='date_only', right_on='date', how='left', suffixes=('', '_daily'))
        
        # Clean up
        df = df.drop(['date_daily', 'date_only'], axis=1)
        
        print(f"Created lag features for {len(df)} records")
        return df
    
    def create_all_features(self) -> pd.DataFrame:
        """Create complete feature set"""
        
        print("=== Starting Feature Engineering Pipeline ===")
        
        # Extract raw data
        df = self.extract_raw_data()
        
        # Create feature sets
        df = self.create_temporal_features(df)
        df = self.create_spatial_features(df)
        df = self.create_crime_features(df)
        df = self.create_lag_features(df)
        
        print(f"=== Feature Engineering Complete ===")
        print(f"Final dataset: {len(df)} records with {len(df.columns)} features")
        
        return df
    
    def save_features_to_db(self, df: pd.DataFrame) -> str:
        """Save engineered features to database"""
        
        print("Saving features to database...")
        
        # Create features table
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS ml_features.crime_features (
            id TEXT,
            date TIMESTAMP,
            case_number TEXT,
            primary_type TEXT,
            description TEXT,
            location_description TEXT,
            arrest BOOLEAN,
            domestic BOOLEAN,
            beat TEXT,
            district TEXT,
            ward TEXT,
            community_area TEXT,
            latitude DECIMAL(10,8),
            longitude DECIMAL(11,8),
            year INTEGER,
            
            -- Temporal features
            hour INTEGER,
            day_of_week INTEGER,
            day_of_month INTEGER,
            month INTEGER,
            quarter INTEGER,
            week_of_year INTEGER,
            is_weekend BOOLEAN,
            is_night BOOLEAN,
            is_rush_hour BOOLEAN,
            season TEXT,
            time_period TEXT,
            
            -- Spatial features
            has_coordinates BOOLEAN,
            district_num INTEGER,
            beat_num INTEGER,
            ward_num INTEGER,
            community_area_num INTEGER,
            location_type TEXT,
            distance_from_center DECIMAL(8,2),
            
            -- Crime features
            crime_category TEXT,
            is_violent_crime BOOLEAN,
            is_property_crime BOOLEAN,
            is_drug_crime BOOLEAN,
            crime_type_arrest_rate DECIMAL(5,4),
            
            -- Lag features
            daily_crime_count INTEGER,
            crimes_7day_avg DECIMAL(8,2),
            crimes_30day_avg DECIMAL(8,2),
            
            -- Metadata
            loaded_at TIMESTAMP,
            source TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        # Create schema if not exists
        with self.engine.begin() as conn:
            conn.execute(text("CREATE SCHEMA IF NOT EXISTS ml_features"))
            conn.execute(text("DROP TABLE IF EXISTS ml_features.crime_features"))
            conn.execute(text(create_table_sql))
        
        # Save features
        df.to_sql(
            'crime_features',
            self.engine,
            schema='ml_features',
            if_exists='append',
            index=False,
            method='multi'
        )
        
        result = f"Saved {len(df)} feature records to ml_features.crime_features"
        print(result)
        return result