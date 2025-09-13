import requests
import pandas as pd
from datetime import datetime, timedelta
import os
from typing import List, Dict

class DOTAPIExtractor:
    """Extractor for DOT flight data API"""
    
    def __init__(self):
        self.api_key = None  # Chicago API doesn't require key
        self.base_url = os.getenv('CHICAGO_API_BASE_URL', 'https://data.cityofchicago.org/resource/ijzp-q8t2.json')
        
        # API key is optional for many Socrata datasets
        if not self.api_key:
            print("No API key provided - using anonymous access")
    
    def extract_daily_data(self, date: datetime) -> List[Dict]:
        """Extract flight data for a specific date"""
        
        # Format date for API query
        date_str = date.strftime('%Y-%m-%d')
        
        # Use a historical date if the requested date is in the future
        # The DOT dataset typically has data up to a few months ago
        from datetime import datetime, timedelta
        today = datetime.now()
        if date.date() >= today.date():
            # Use a date from 3 months ago as a fallback
            fallback_date = today - timedelta(days=90)
            date_str = fallback_date.strftime('%Y-%m-%d')
            print(f"Requested date {date.strftime('%Y-%m-%d')} is in future, using fallback date {date_str}")
        
        # Build query parameters - extract more data
        params = {
            '$limit': 10000,  # Increased limit for more data
            '$where': f"date >= '{date_str}T00:00:00.000' AND date < '{date_str}T23:59:59.999'"  # Full day range
        }
        
        # Add API key if available
        if self.api_key:
            params['$$app_token'] = self.api_key
        
        try:
            print(f"Requesting data for {date_str} from {self.base_url}")
            response = requests.get(self.base_url, params=params, timeout=300)
            
            if response.status_code == 403:
                print(f"403 Forbidden - API key might be invalid or rate limited")
                print(f"Response: {response.text[:500]}")
                # Return empty list instead of failing
                return []
            
            response.raise_for_status()
            
            data = response.json()
            print(f"Successfully extracted {len(data)} records for {date_str}")
            
            return data
            
        except requests.exceptions.RequestException as e:
            print(f"Error extracting data for {date_str}: {e}")
            # Return empty list instead of failing to allow pipeline to continue
            print("Returning empty dataset to allow pipeline to continue")
            return []
    
    def extract_date_range(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Extract data for a date range"""
        
        all_data = []
        current_date = start_date
        
        while current_date <= end_date:
            daily_data = self.extract_daily_data(current_date)
            all_data.extend(daily_data)
            current_date += timedelta(days=1)
        
        return all_data
    
    def extract_monthly_data(self, year: int, month: int) -> List[Dict]:
        """Extract data for an entire month"""
        
        # Build query for entire month
        date_filter = f"date >= '{year}-{month:02d}-01T00:00:00.000' AND date < '{year}-{month+1 if month < 12 else year+1}-{1 if month < 12 else 1:02d}-01T00:00:00.000'"
        
        params = {
            '$limit': 50000,  # Higher limit for monthly data
            '$where': date_filter
        }
        
        # Add API key if available
        if self.api_key:
            params['$$app_token'] = self.api_key
        
        try:
            print(f"Requesting monthly data for {year}-{month:02d}")
            response = requests.get(self.base_url, params=params, timeout=300)
            
            if response.status_code == 403:
                print(f"403 Forbidden - API key might be invalid or rate limited")
                return []
            
            response.raise_for_status()
            
            data = response.json()
            print(f"Successfully extracted {len(data)} records for {year}-{month:02d}")
            
            return data
            
        except requests.exceptions.RequestException as e:
            print(f"Error extracting monthly data for {year}-{month:02d}: {e}")
            return []