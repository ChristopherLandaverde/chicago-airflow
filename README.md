# Chicago Crime Analytics Dashboard

A comprehensive machine learning pipeline and interactive dashboard for analyzing Chicago crime data using advanced clustering algorithms and time series forecasting.

## 🚀 Features

- **Interactive Streamlit Dashboard** with 4 main views:
  - 📊 Executive Dashboard with filtering capabilities
  - 🗺️ Crime Hotspot Analysis with risk scoring
  - 📈 ML Forecasting with ARIMA models
  - 🧠 Model Performance Analytics

- **Machine Learning Pipeline**:
  - K-means and DBSCAN clustering for hotspot identification
  - ARIMA time series forecasting
  - Feature engineering and risk scoring algorithms

- **Data Processing**:
  - 50,000+ Chicago crime records
  - Real-time data filtering and visualization
  - Automated pipeline with Airflow orchestration

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly
- **Machine Learning**: Scikit-learn, ARIMA
- **Orchestration**: Apache Airflow
- **Data Storage**: CSV files with JSON metadata

## 📊 Dashboard Views

### Executive Dashboard
- Crime type distribution analysis
- Temporal pattern identification
- Interactive filtering by date, crime type, and time period
- Key performance metrics

### Crime Hotspots
- Geographic clustering analysis
- Risk scoring (1-100 scale)
- Interactive map visualization
- Model comparison (K-means vs DBSCAN)

### ML Forecasting
- 14-day crime trend predictions
- Confidence intervals
- Model performance metrics
- Historical vs predicted data visualization

### Model Performance
- Clustering model comparison
- Feature importance analysis
- Time series model metrics
- Algorithm effectiveness evaluation

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd flowdbt
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the dashboard**
   ```bash
   streamlit run streamlit_app.py
   ```

5. **Access the dashboard**
   - Open your browser to `http://localhost:8501`

## 📁 Project Structure

```
flowdbt/
├── streamlit_app.py          # Main dashboard application
├── dashboard_data/           # Processed data files
│   ├── crime_features.csv    # Crime data with features
│   ├── crime_forecasts.csv   # Forecasting results
│   ├── crime_hotspots.csv    # Hotspot analysis
│   └── summary_stats.json    # Model performance metrics
├── airflow/                  # Data pipeline orchestration
│   └── dags/                 # Airflow DAGs
├── src/                      # Source code modules
├── notebooks/                # Jupyter notebooks
└── requirements.txt          # Python dependencies
```

## 🔧 Key Components

### Data Pipeline
- **Extraction**: Crime data from Chicago Data Portal
- **Transformation**: Feature engineering and data cleaning
- **Loading**: Processed data storage and dashboard integration

### Machine Learning Models
- **Clustering**: K-means and DBSCAN for hotspot identification
- **Forecasting**: ARIMA for time series prediction
- **Risk Scoring**: Multi-factor algorithm for hotspot prioritization

### Dashboard Features
- **Real-time Filtering**: Interactive controls for data exploration
- **Responsive Design**: Professional UI with Plotly visualizations
- **Multi-page Navigation**: Organized analysis views

## 📈 Performance Metrics

- **Clustering Accuracy**: 0.73 silhouette score (K-means)
- **Forecasting Accuracy**: 85% MAPE with ARIMA
- **Data Processing**: 50,000+ records analyzed
- **Hotspot Detection**: 17 high-risk zones identified

## 🎯 Use Cases

- **Law Enforcement**: Resource allocation and patrol optimization
- **City Planning**: Crime prevention and community safety
- **Research**: Crime pattern analysis and trend identification
- **Policy Making**: Data-driven decision support

## 🤝 Contributing

This is a portfolio project demonstrating end-to-end data science capabilities including:
- Data engineering and pipeline development
- Machine learning model development
- Interactive dashboard creation
- Production-ready deployment considerations

## 📄 License

This project is for portfolio demonstration purposes.