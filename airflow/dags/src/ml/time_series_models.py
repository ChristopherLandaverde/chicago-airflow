import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Time series libraries
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import prophet
from prophet import Prophet

# Plotting and metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error

class CrimeTimeSeriesForecaster:
    """Time series forecasting models for Chicago crime data"""
    
    def __init__(self):
        self.db_config = {
            'host': os.getenv('DB_HOST', 'postgres'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME', 'crime_analytics'),
            'user': os.getenv('DB_USER', 'airflow'),
            'password': os.getenv('DB_PASSWORD', 'airflow')
        }
        
        self.engine = self._create_engine()
        self.models = {}
        self.forecasts = {}
    
    def _create_engine(self):
        """Create SQLAlchemy engine"""
        conn_string = (
            f"postgresql://{self.db_config['user']}:{self.db_config['password']}"
            f"@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
        )
        return create_engine(conn_string)
    
    def load_time_series_data(self, crime_type: Optional[str] = None) -> pd.DataFrame:
        """Load and prepare time series data for modeling"""
        
        print(f"Loading time series data for crime type: {crime_type or 'ALL'}")
        
        # Base query
        base_query = """
        SELECT 
            DATE(date) as date,
            primary_type,
            COUNT(*) as crime_count
        FROM ml_features.crime_features
        """
        
        if crime_type:
            query = base_query + f" WHERE primary_type = '{crime_type}'"
        else:
            query = base_query
            
        query += " GROUP BY DATE(date), primary_type ORDER BY date"
        
        df = pd.read_sql(query, self.engine)
        df['date'] = pd.to_datetime(df['date'])
        
        if crime_type:
            # Single crime type time series
            ts_data = df.groupby('date')['crime_count'].sum().reset_index()
        else:
            # Total crime time series
            ts_data = df.groupby('date')['crime_count'].sum().reset_index()
        
        # Fill missing dates with 0
        date_range = pd.date_range(start=ts_data['date'].min(), 
                                  end=ts_data['date'].max(), 
                                  freq='D')
        
        complete_ts = pd.DataFrame({'date': date_range})
        ts_data = complete_ts.merge(ts_data, on='date', how='left')
        ts_data['crime_count'] = ts_data['crime_count'].fillna(0)
        
        print(f"Loaded {len(ts_data)} daily observations from {ts_data['date'].min()} to {ts_data['date'].max()}")
        
        return ts_data
    
    def check_stationarity(self, series: pd.Series) -> Dict:
        """Check if time series is stationary using Augmented Dickey-Fuller test"""
        
        result = adfuller(series.dropna())
        
        stationarity_result = {
            'adf_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'is_stationary': result[1] < 0.05
        }
        
        print(f"ADF Statistic: {result[0]:.6f}")
        print(f"p-value: {result[1]:.6f}")
        print(f"Is Stationary: {stationarity_result['is_stationary']}")
        
        return stationarity_result
    
    def seasonal_decomposition(self, ts_data: pd.DataFrame, period: int = 365) -> Dict:
        """Perform seasonal decomposition of time series"""
        
        print(f"Performing seasonal decomposition with period={period}")
        
        # Set date as index
        ts_indexed = ts_data.set_index('date')['crime_count']
        
        # Perform decomposition
        decomposition = seasonal_decompose(ts_indexed, 
                                         model='additive', 
                                         period=period)
        
        decomp_results = {
            'trend': decomposition.trend,
            'seasonal': decomposition.seasonal,
            'residual': decomposition.resid,
            'original': ts_indexed
        }
        
        print("Seasonal decomposition completed")
        
        return decomp_results
    
    def fit_arima_model(self, ts_data: pd.DataFrame, order: Tuple[int, int, int] = (1, 1, 1)) -> Dict:
        """Fit ARIMA model to time series data"""
        
        print(f"Fitting ARIMA{order} model...")
        
        # Prepare data
        ts_series = ts_data.set_index('date')['crime_count']
        
        # Split into train/test (80/20)
        split_point = int(len(ts_series) * 0.8)
        train_data = ts_series[:split_point]
        test_data = ts_series[split_point:]
        
        try:
            # Fit ARIMA model
            model = ARIMA(train_data, order=order)
            fitted_model = model.fit()
            
            # Generate forecasts
            forecast_steps = len(test_data)
            forecast = fitted_model.forecast(steps=forecast_steps)
            forecast_ci = fitted_model.get_forecast(steps=forecast_steps).conf_int()
            
            # Calculate metrics
            mae = mean_absolute_error(test_data, forecast)
            rmse = np.sqrt(mean_squared_error(test_data, forecast))
            mape = np.mean(np.abs((test_data - forecast) / test_data)) * 100
            
            results = {
                'model': fitted_model,
                'train_data': train_data,
                'test_data': test_data,
                'forecast': forecast,
                'forecast_ci': forecast_ci,
                'metrics': {
                    'mae': mae,
                    'rmse': rmse,
                    'mape': mape,
                    'aic': fitted_model.aic,
                    'bic': fitted_model.bic
                },
                'model_summary': fitted_model.summary()
            }
            
            print(f"ARIMA model fitted successfully")
            print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
            
            return results
            
        except Exception as e:
            print(f"Error fitting ARIMA model: {e}")
            return {'error': str(e)}
    
    def fit_prophet_model(self, ts_data: pd.DataFrame) -> Dict:
        """Fit Prophet model to time series data"""
        
        print("Fitting Prophet model...")
        
        # Prepare data for Prophet (requires 'ds' and 'y' columns)
        prophet_data = ts_data.copy()
        prophet_data.columns = ['ds', 'y']
        
        # Split into train/test (80/20)
        split_point = int(len(prophet_data) * 0.8)
        train_data = prophet_data[:split_point]
        test_data = prophet_data[split_point:]
        
        try:
            # Initialize and fit Prophet model
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0
            )
            
            model.fit(train_data)
            
            # Generate forecasts
            future_dates = model.make_future_dataframe(periods=len(test_data))
            forecast = model.predict(future_dates)
            
            # Extract test period forecasts
            test_forecast = forecast.tail(len(test_data))
            
            # Calculate metrics
            mae = mean_absolute_error(test_data['y'], test_forecast['yhat'])
            rmse = np.sqrt(mean_squared_error(test_data['y'], test_forecast['yhat']))
            mape = np.mean(np.abs((test_data['y'] - test_forecast['yhat']) / test_data['y'])) * 100
            
            results = {
                'model': model,
                'train_data': train_data,
                'test_data': test_data,
                'forecast': forecast,
                'test_forecast': test_forecast,
                'metrics': {
                    'mae': mae,
                    'rmse': rmse,
                    'mape': mape
                }
            }
            
            print(f"Prophet model fitted successfully")
            print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
            
            return results
            
        except Exception as e:
            print(f"Error fitting Prophet model: {e}")
            return {'error': str(e)}
    
    def generate_future_forecasts(self, model_results: Dict, model_type: str, days_ahead: int = 30) -> pd.DataFrame:
        """Generate future forecasts beyond training data"""
        
        print(f"Generating {days_ahead}-day forecast using {model_type} model...")
        
        if model_type.lower() == 'arima':
            model = model_results['model']
            forecast = model.forecast(steps=days_ahead)
            forecast_ci = model.get_forecast(steps=days_ahead).conf_int()
            
            # Create forecast dataframe
            last_date = model_results['test_data'].index[-1]
            future_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                       periods=days_ahead, freq='D')
            
            forecast_df = pd.DataFrame({
                'date': future_dates,
                'forecast': forecast.values,
                'lower_ci': forecast_ci.iloc[:, 0].values,
                'upper_ci': forecast_ci.iloc[:, 1].values,
                'model_type': 'ARIMA'
            })
            
        elif model_type.lower() == 'prophet':
            model = model_results['model']
            
            # Create future dataframe
            last_date = model_results['test_data']['ds'].max()
            future_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                       periods=days_ahead, freq='D')
            
            future_df = pd.DataFrame({'ds': future_dates})
            forecast = model.predict(future_df)
            
            forecast_df = pd.DataFrame({
                'date': future_dates,
                'forecast': forecast['yhat'].values,
                'lower_ci': forecast['yhat_lower'].values,
                'upper_ci': forecast['yhat_upper'].values,
                'model_type': 'Prophet'
            })
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        print(f"Generated forecasts from {forecast_df['date'].min()} to {forecast_df['date'].max()}")
        
        return forecast_df
    
    def compare_models(self, ts_data: pd.DataFrame) -> Dict:
        """Compare ARIMA and Prophet models performance"""
        
        print("=== Comparing ARIMA vs Prophet Models ===")
        
        # Fit both models
        arima_results = self.fit_arima_model(ts_data)
        prophet_results = self.fit_prophet_model(ts_data)
        
        if 'error' in arima_results or 'error' in prophet_results:
            print("Error in model fitting")
            return {'arima': arima_results, 'prophet': prophet_results}
        
        # Compare metrics
        comparison = {
            'arima': {
                'mae': arima_results['metrics']['mae'],
                'rmse': arima_results['metrics']['rmse'],
                'mape': arima_results['metrics']['mape']
            },
            'prophet': {
                'mae': prophet_results['metrics']['mae'],
                'rmse': prophet_results['metrics']['rmse'],
                'mape': prophet_results['metrics']['mape']
            }
        }
        
        # Determine best model
        arima_score = (arima_results['metrics']['mae'] + arima_results['metrics']['rmse']) / 2
        prophet_score = (prophet_results['metrics']['mae'] + prophet_results['metrics']['rmse']) / 2
        
        best_model = 'ARIMA' if arima_score < prophet_score else 'Prophet'
        
        print(f"\n=== Model Comparison Results ===")
        print(f"ARIMA - MAE: {comparison['arima']['mae']:.2f}, RMSE: {comparison['arima']['rmse']:.2f}, MAPE: {comparison['arima']['mape']:.2f}%")
        print(f"Prophet - MAE: {comparison['prophet']['mae']:.2f}, RMSE: {comparison['prophet']['rmse']:.2f}, MAPE: {comparison['prophet']['mape']:.2f}%")
        print(f"Best Model: {best_model}")
        
        return {
            'arima': arima_results,
            'prophet': prophet_results,
            'comparison': comparison,
            'best_model': best_model
        }
    
    def save_forecasts_to_db(self, forecasts: pd.DataFrame, crime_type: str = 'ALL') -> str:
        """Save forecasts to database"""
        
        print(f"Saving forecasts to database for crime type: {crime_type}")
        
        # Add metadata
        forecasts['crime_type'] = crime_type
        forecasts['created_at'] = datetime.now()
        
        # Create forecasts table
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS ml_features.crime_forecasts (
            date DATE,
            forecast DECIMAL(10,2),
            lower_ci DECIMAL(10,2),
            upper_ci DECIMAL(10,2),
            model_type TEXT,
            crime_type TEXT,
            created_at TIMESTAMP,
            PRIMARY KEY (date, model_type, crime_type)
        );
        """
        
        with self.engine.begin() as conn:
            conn.execute(text("CREATE SCHEMA IF NOT EXISTS ml_features"))
            conn.execute(text(create_table_sql))
        
        # Save forecasts
        forecasts.to_sql(
            'crime_forecasts',
            self.engine,
            schema='ml_features',
            if_exists='append',
            index=False,
            method='multi'
        )
        
        result = f"Saved {len(forecasts)} forecast records to ml_features.crime_forecasts"
        print(result)
        return result
    
    def run_full_forecasting_pipeline(self, crime_type: Optional[str] = None, days_ahead: int = 30) -> Dict:
        """Run complete time series forecasting pipeline"""
        
        print(f"=== Starting Time Series Forecasting Pipeline ===")
        print(f"Crime Type: {crime_type or 'ALL'}")
        print(f"Forecast Horizon: {days_ahead} days")
        
        # Load data
        ts_data = self.load_time_series_data(crime_type)
        
        # Check stationarity
        stationarity = self.check_stationarity(ts_data['crime_count'])
        
        # Seasonal decomposition
        decomposition = self.seasonal_decomposition(ts_data)
        
        # Compare models
        model_comparison = self.compare_models(ts_data)
        
        # Generate future forecasts with best model
        if model_comparison['best_model'] == 'ARIMA':
            future_forecasts = self.generate_future_forecasts(
                model_comparison['arima'], 'arima', days_ahead
            )
        else:
            future_forecasts = self.generate_future_forecasts(
                model_comparison['prophet'], 'prophet', days_ahead
            )
        
        # Save forecasts
        save_result = self.save_forecasts_to_db(future_forecasts, crime_type or 'ALL')
        
        results = {
            'data_summary': {
                'total_observations': len(ts_data),
                'date_range': (ts_data['date'].min(), ts_data['date'].max()),
                'mean_daily_crimes': ts_data['crime_count'].mean(),
                'std_daily_crimes': ts_data['crime_count'].std()
            },
            'stationarity': stationarity,
            'model_comparison': model_comparison,
            'future_forecasts': future_forecasts,
            'save_result': save_result
        }
        
        print(f"=== Forecasting Pipeline Complete ===")
        
        return results