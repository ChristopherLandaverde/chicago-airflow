import pandas as pd
import psycopg2
from sqlalchemy import create_engine, text
import os
import json
from datetime import datetime
from typing import List, Dict

class PostgresLoader:
    """Loader for PostgreSQL database"""
    
    def __init__(self):
        self.db_config = {
            'host': os.getenv('DB_HOST', 'postgres'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME', 'flight_analytics'),
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
    
    def load_daily_data(self, date: datetime) -> str:
        """Load daily crime data into raw table"""
        
        # This would typically read from a staging area or temp file
        # For now, we'll simulate loading from the extractor
        from src.extractors.dot_api import DOTAPIExtractor
        
        extractor = DOTAPIExtractor()
        data = extractor.extract_daily_data(date)
        
        if not data:
            print("No data extracted, skipping load")
            return "No data to load - extraction returned empty dataset"
        
        # Convert to DataFrame with flexible schema
        processed_data = []
        for record in data:
            processed_data.append({
                'id': record.get('id', ''),
                'date': record.get('date'),
                'record_data': record,  # Store full record as JSON
                'loaded_at': datetime.now(),
                'source': 'crime_api'
            })
        
        df = pd.DataFrame(processed_data)
        
        # Load to raw table
        table_name = 'raw_crime_data'
        
        try:
            # Create table if not exists
            self._create_raw_table()
            
            # Convert record_data to JSON strings for PostgreSQL
            df['record_data'] = df['record_data'].apply(json.dumps)
            
            # Load data
            df.to_sql(
                table_name,
                self.engine,
                if_exists='append',
                index=False,
                method='multi'
            )
            
            print(f"Successfully loaded {len(df)} records to {table_name}")
            return f"Loaded {len(df)} records to {table_name}"
            
        except Exception as e:
            print(f"Error loading data: {e}")
            # Don't raise the exception, just return error message
            return f"Error loading data: {str(e)}"
    
    def load_batch_data(self, data: List[Dict], batch_id: str) -> str:
        """Load batch data (e.g., monthly) into raw table"""
        
        if not data:
            print("No data in batch, skipping load")
            return f"No data to load for batch {batch_id}"
        
        # Convert to DataFrame with flexible schema
        processed_data = []
        for record in data:
            processed_data.append({
                'id': record.get('id', ''),
                'date': record.get('date'),
                'record_data': record,  # Store full record as JSON
                'loaded_at': datetime.now(),
                'source': f'batch_api_{batch_id}'
            })
        
        df = pd.DataFrame(processed_data)
        
        # Load to raw table
        table_name = 'raw_crime_data'
        
        try:
            # Create table if not exists
            self._create_raw_table()
            
            # Convert record_data to JSON strings for PostgreSQL
            df['record_data'] = df['record_data'].apply(json.dumps)
            
            # Load data
            df.to_sql(
                table_name,
                self.engine,
                if_exists='append',
                index=False,
                method='multi'
            )
            
            print(f"Successfully loaded {len(df)} records to {table_name} for batch {batch_id}")
            return f"Loaded {len(df)} records to {table_name} for batch {batch_id}"
            
        except Exception as e:
            print(f"Error loading batch data: {e}")
            return f"Error loading batch data: {str(e)}"
    
    def _create_raw_table(self):
        """Create raw data table if not exists (flexible schema for demo)"""
        
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS raw_crime_data (
            id TEXT,
            date TIMESTAMP,
            record_data JSONB,
            loaded_at TIMESTAMP,
            source TEXT
        );
        """
        
        with self.engine.begin() as conn:
            conn.execute(text(create_table_sql))