# Machine Learning Pipeline for Chicago Crime Analytics

This document describes the ML pipeline components built for the Chicago Crime Analytics portfolio project.

## Overview

The ML pipeline consists of three main components:

1. **Feature Engineering** - Transforms raw crime data into ML-ready features
2. **Time Series Forecasting** - Predicts future crime trends using ARIMA and Prophet models
3. **Spatial Clustering** - Identifies crime hotspots using K-means and DBSCAN algorithms

## Components

### 1. Feature Engineering (`airflow/dags/src/ml/feature_engineering.py`)

**Purpose**: Transform raw JSON crime data into structured features for ML models.

**Features Created**:
- **Temporal Features**: hour, day_of_week, month, season, is_weekend, is_night, is_rush_hour
- **Spatial Features**: distance_from_center, location_type, geographic clustering
- **Crime Features**: crime_category, severity flags, arrest likelihood
- **Lag Features**: rolling averages, daily crime counts

**Key Methods**:
- `create_all_features()` - Main pipeline to create all feature sets
- `save_features_to_db()` - Saves engineered features to `ml_features.crime_features` table

### 2. Time Series Forecasting (`airflow/dags/src/ml/time_series_models.py`)

**Purpose**: Forecast future crime trends using statistical and ML models.

**Models Implemented**:
- **ARIMA**: Statistical time series model with trend and seasonality
- **Prophet**: Facebook's forecasting model with holiday effects and changepoints

**Key Features**:
- Automatic model comparison and selection
- Seasonal decomposition analysis
- Confidence intervals for predictions
- Performance metrics (MAE, RMSE, MAPE)

**Key Methods**:
- `run_full_forecasting_pipeline()` - Complete forecasting workflow
- `compare_models()` - Compare ARIMA vs Prophet performance
- `generate_future_forecasts()` - Create future predictions

**Output**: Saves forecasts to `ml_features.crime_forecasts` table

### 3. Spatial Clustering (`airflow/dags/src/ml/spatial_clustering.py`)

**Purpose**: Identify crime hotspots and spatial patterns using clustering algorithms.

**Models Implemented**:
- **K-means**: Partitional clustering with optimal K selection
- **DBSCAN**: Density-based clustering for irregular hotspot shapes

**Key Features**:
- Automatic parameter tuning (optimal K, eps, min_samples)
- Risk score calculation for each cluster
- Geographic feature engineering
- Hotspot ranking and analysis

**Key Methods**:
- `run_full_spatial_analysis()` - Complete spatial analysis workflow
- `analyze_clusters()` - Generate hotspot profiles and risk scores
- `save_hotspot_analysis_to_db()` - Save results to database

**Output**: 
- Cluster assignments: `ml_features.crime_clusters` table
- Hotspot analysis: `ml_features.crime_hotspots` table

## Airflow DAG (`airflow/dags/ml_pipeline.py`)

**Schedule**: Weekly (`@weekly`)

**Tasks**:
1. `feature_engineering` - Create ML features from raw data
2. `time_series_forecasting` - Generate overall crime forecasts
3. `violent_crime_forecasting` - Specific forecasts for violent crimes
4. `spatial_clustering_analysis` - Identify crime hotspots
5. `violent_crime_clustering` - Hotspot analysis for violent crimes
6. `generate_performance_report` - Create model performance summary

**Dependencies**:
```
feature_engineering → [time_series_forecasting, spatial_clustering_analysis]
time_series_forecasting → violent_crime_forecasting
spatial_clustering_analysis → violent_crime_clustering
[violent_crime_forecasting, violent_crime_clustering] → generate_performance_report
```

## Database Schema

### ml_features.crime_features
Stores engineered features for ML models.

**Key Columns**:
- Temporal: `hour`, `day_of_week`, `season`, `is_weekend`
- Spatial: `latitude`, `longitude`, `distance_from_center`, `location_type`
- Crime: `crime_category`, `is_violent_crime`, `crime_type_arrest_rate`
- Lag: `daily_crime_count`, `crimes_7day_avg`, `crimes_30day_avg`

### ml_features.crime_forecasts
Stores time series predictions.

**Key Columns**:
- `date` - Forecast date
- `forecast` - Predicted crime count
- `lower_ci`, `upper_ci` - Confidence intervals
- `model_type` - ARIMA or Prophet
- `crime_type` - Specific crime type or ALL

### ml_features.crime_clusters
Stores cluster assignments for each crime record.

**Key Columns**:
- `cluster_id` - Assigned cluster number
- `model_type` - K-means or DBSCAN
- `latitude`, `longitude` - Crime location

### ml_features.crime_hotspots
Stores hotspot analysis results.

**Key Columns**:
- `cluster_id` - Hotspot identifier
- `risk_score` - Calculated risk score (0-100)
- `center_latitude`, `center_longitude` - Hotspot center
- `cluster_size` - Number of crimes in hotspot
- `top_crime_type` - Most common crime in hotspot

## Usage

### Running the Full Pipeline

```bash
# Trigger the ML pipeline DAG
docker-compose exec airflow-webserver airflow dags trigger ml_pipeline
```

### Running Individual Components

```python
# Feature Engineering
from ml.feature_engineering import CrimeFeatureEngineer
engineer = CrimeFeatureEngineer()
features = engineer.create_all_features()
engineer.save_features_to_db(features)

# Time Series Forecasting
from ml.time_series_models import CrimeTimeSeriesForecaster
forecaster = CrimeTimeSeriesForecaster()
results = forecaster.run_full_forecasting_pipeline(days_ahead=30)

# Spatial Clustering
from ml.spatial_clustering import CrimeSpatialAnalyzer
analyzer = CrimeSpatialAnalyzer()
results = analyzer.run_full_spatial_analysis(time_period='recent_year')
```

## Model Performance

### Time Series Models
- **ARIMA**: Good for stationary time series with clear trends
- **Prophet**: Better for handling seasonality and holiday effects
- **Metrics**: MAE, RMSE, MAPE for forecast accuracy

### Clustering Models
- **K-means**: Creates balanced, spherical clusters
- **DBSCAN**: Identifies irregular hotspots and filters noise
- **Metrics**: Silhouette score, Calinski-Harabasz index

## Dependencies

Key Python packages required:
- `pandas`, `numpy` - Data manipulation
- `scikit-learn` - ML algorithms
- `statsmodels` - ARIMA modeling
- `prophet` - Facebook Prophet forecasting
- `geopandas`, `shapely` - Geospatial analysis
- `sqlalchemy`, `psycopg2` - Database connectivity

## Future Enhancements

1. **Model Retraining**: Implement automated model retraining with performance monitoring
2. **Real-time Predictions**: Add streaming predictions for live crime data
3. **Advanced Models**: Implement LSTM/GRU for deep learning forecasting
4. **Ensemble Methods**: Combine multiple models for better accuracy
5. **Feature Selection**: Automated feature importance and selection
6. **Model Drift Detection**: Monitor model performance degradation over time

## Monitoring and Alerts

The pipeline includes basic performance reporting. For production use, consider adding:
- Model accuracy monitoring
- Data drift detection
- Automated retraining triggers
- Performance degradation alerts
- Resource usage monitoring