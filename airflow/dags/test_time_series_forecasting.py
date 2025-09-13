#!/usr/bin/env python3
"""
Test time series forecasting with actual ML libraries
"""

import sys
sys.path.append('/opt/airflow/dags/src')

def test_time_series_forecasting():
    """Test time series forecasting pipeline"""
    print("=== Testing Time Series Forecasting Pipeline ===")
    
    try:
        from ml.time_series_models import CrimeTimeSeriesForecaster
        
        # Initialize forecaster
        forecaster = CrimeTimeSeriesForecaster()
        print("✓ CrimeTimeSeriesForecaster initialized with ML libraries")
        
        # Check data availability
        with forecaster.engine.connect() as conn:
            result = conn.execute("SELECT COUNT(*) FROM ml_features.crime_features")
            count = result.fetchone()[0]
            print(f"✓ Found {count} feature records for forecasting")
        
        # Load time series data for overall crime trends
        print("\n=== Loading Time Series Data ===")
        ts_data = forecaster.load_time_series_data()
        print(f"✓ Loaded time series: {len(ts_data)} daily observations")
        print(f"  Date range: {ts_data['date'].min()} to {ts_data['date'].max()}")
        print(f"  Crime count range: {ts_data['crime_count'].min()} to {ts_data['crime_count'].max()}")
        
        # Test stationarity check
        print("\n=== Testing Stationarity ===")
        stationarity = forecaster.check_stationarity(ts_data['crime_count'])
        print(f"✓ Stationarity test completed")
        
        # Test seasonal decomposition
        print("\n=== Testing Seasonal Decomposition ===")
        decomposition = forecaster.seasonal_decomposition(ts_data, period=7)  # Weekly seasonality
        print(f"✓ Seasonal decomposition completed")
        
        # Test ARIMA model (simple parameters for demo)
        print("\n=== Testing ARIMA Model ===")
        arima_results = forecaster.fit_arima_model(ts_data, order=(1, 1, 1))
        
        if 'error' not in arima_results:
            print(f"✓ ARIMA model fitted successfully")
            print(f"  MAE: {arima_results['metrics']['mae']:.2f}")
            print(f"  RMSE: {arima_results['metrics']['rmse']:.2f}")
            print(f"  MAPE: {arima_results['metrics']['mape']:.2f}%")
            print(f"  AIC: {arima_results['metrics']['aic']:.2f}")
        else:
            print(f"✗ ARIMA model failed: {arima_results['error']}")
            return False
        
        # Test Prophet model
        print("\n=== Testing Prophet Model ===")
        prophet_results = forecaster.fit_prophet_model(ts_data)
        
        if 'error' not in prophet_results:
            print(f"✓ Prophet model fitted successfully")
            print(f"  MAE: {prophet_results['metrics']['mae']:.2f}")
            print(f"  RMSE: {prophet_results['metrics']['rmse']:.2f}")
            print(f"  MAPE: {prophet_results['metrics']['mape']:.2f}%")
        else:
            print(f"✗ Prophet model failed: {prophet_results['error']}")
            return False
        
        # Compare models
        print("\n=== Model Comparison ===")
        arima_score = (arima_results['metrics']['mae'] + arima_results['metrics']['rmse']) / 2
        prophet_score = (prophet_results['metrics']['mae'] + prophet_results['metrics']['rmse']) / 2
        
        best_model = 'ARIMA' if arima_score < prophet_score else 'Prophet'
        print(f"ARIMA Average Score: {arima_score:.2f}")
        print(f"Prophet Average Score: {prophet_score:.2f}")
        print(f"✓ Best Model: {best_model}")
        
        # Generate future forecasts
        print("\n=== Generating Future Forecasts ===")
        if best_model == 'ARIMA':
            future_forecasts = forecaster.generate_future_forecasts(arima_results, 'arima', days_ahead=14)
        else:
            future_forecasts = forecaster.generate_future_forecasts(prophet_results, 'prophet', days_ahead=14)
        
        print(f"✓ Generated 14-day forecast")
        print(f"  Forecast range: {future_forecasts['forecast'].min():.1f} to {future_forecasts['forecast'].max():.1f} crimes/day")
        
        # Save forecasts to database
        print("\n=== Saving Forecasts ===")
        save_result = forecaster.save_forecasts_to_db(future_forecasts, 'ALL_CRIMES')
        print(f"✓ {save_result}")
        
        # Test specific crime type forecasting
        print("\n=== Testing Crime-Specific Forecasting ===")
        theft_ts = forecaster.load_time_series_data(crime_type='THEFT')
        if len(theft_ts) > 30:  # Need enough data
            print(f"✓ Loaded THEFT time series: {len(theft_ts)} days")
            
            # Quick Prophet model for theft
            theft_prophet = forecaster.fit_prophet_model(theft_ts)
            if 'error' not in theft_prophet:
                theft_forecast = forecaster.generate_future_forecasts(theft_prophet, 'prophet', days_ahead=7)
                theft_save = forecaster.save_forecasts_to_db(theft_forecast, 'THEFT')
                print(f"✓ THEFT forecasting completed: {theft_save}")
            else:
                print(f"⚠️  THEFT forecasting skipped: insufficient data")
        
        return True
        
    except Exception as e:
        print(f"✗ Time series forecasting test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_time_series_forecasting()
    if success:
        print("\n🎉 Time Series Forecasting Pipeline Test Completed Successfully!")
        print("\n📊 Results:")
        print("  • ARIMA and Prophet models working")
        print("  • Future forecasts generated")
        print("  • Results saved to ml_features.crime_forecasts")
        print("  • Ready for production pipeline integration")
    else:
        print("\n❌ Time Series Forecasting Test Failed")
    exit(0 if success else 1)