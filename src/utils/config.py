import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration settings"""
    
    # DOT API
    DOT_API_KEY = os.getenv('DOT_API_KEY')
    DOT_API_BASE_URL = os.getenv('DOT_API_BASE_URL', 'https://data.transportation.gov/resource/s6ew-h6mp.json')
    
    # Database
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = os.getenv('DB_PORT', '5432')
    DB_NAME = os.getenv('DB_NAME', 'flight_analytics')
    DB_USER = os.getenv('DB_USER', 'airflow')
    DB_PASSWORD = os.getenv('DB_PASSWORD', 'airflow')
    
    @property
    def database_url(self):
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"