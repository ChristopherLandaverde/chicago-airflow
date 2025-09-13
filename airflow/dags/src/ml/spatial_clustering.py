import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Clustering libraries
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# Geospatial libraries
import geopandas as gpd
from shapely.geometry import Point
from scipy.spatial.distance import pdist, squareform

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

class CrimeSpatialAnalyzer:
    """Spatial clustering and hotspot analysis for Chicago crime data"""
    
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
        self.hotspots = {}
    
    def _create_engine(self):
        """Create SQLAlchemy engine"""
        conn_string = (
            f"postgresql://{self.db_config['user']}:{self.db_config['password']}"
            f"@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
        )
        return create_engine(conn_string)
    
    def load_spatial_data(self, crime_type: Optional[str] = None, 
                         time_period: Optional[str] = None) -> pd.DataFrame:
        """Load crime data with spatial coordinates for clustering"""
        
        print(f"Loading spatial data for crime type: {crime_type or 'ALL'}")
        print(f"Time period: {time_period or 'ALL'}")
        
        # Base query
        base_query = """
        SELECT 
            id,
            date,
            primary_type,
            latitude,
            longitude,
            district_num,
            beat_num,
            ward_num,
            community_area_num,
            crime_category,
            is_violent_crime,
            is_property_crime,
            distance_from_center,
            hour,
            day_of_week,
            is_weekend,
            season
        FROM ml_features.crime_features
        WHERE latitude IS NOT NULL 
        AND longitude IS NOT NULL
        AND latitude BETWEEN 41.6 AND 42.1
        AND longitude BETWEEN -87.9 AND -87.5
        """
        
        conditions = []
        
        if crime_type:
            conditions.append(f"primary_type = '{crime_type}'")
        
        if time_period:
            if time_period == 'recent_year':
                conditions.append("date >= CURRENT_DATE - INTERVAL '1 year'")
            elif time_period == 'recent_month':
                conditions.append("date >= CURRENT_DATE - INTERVAL '1 month'")
        
        if conditions:
            base_query += " AND " + " AND ".join(conditions)
        
        base_query += " ORDER BY date DESC"
        
        df = pd.read_sql(base_query, self.engine)
        
        print(f"Loaded {len(df)} records with valid coordinates")
        
        return df
    
    def create_spatial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional spatial features for clustering"""
        
        print("Creating spatial features for clustering...")
        
        # Crime density features (crimes per square mile in local area)
        # Create grid-based density
        lat_bins = np.linspace(df['latitude'].min(), df['latitude'].max(), 50)
        lon_bins = np.linspace(df['longitude'].min(), df['longitude'].max(), 50)
        
        df['lat_bin'] = pd.cut(df['latitude'], lat_bins, labels=False)
        df['lon_bin'] = pd.cut(df['longitude'], lon_bins, labels=False)
        
        # Calculate local density
        grid_counts = df.groupby(['lat_bin', 'lon_bin']).size().reset_index(name='grid_crime_count')
        df = df.merge(grid_counts, on=['lat_bin', 'lon_bin'], how='left')
        
        # Time-based spatial features
        df['crimes_same_location_24h'] = df.groupby(['lat_bin', 'lon_bin'])['date'].transform(
            lambda x: x.dt.date.value_counts().max()
        )
        
        # Distance to high-crime areas (downtown Chicago)
        high_crime_areas = [
            (41.8781, -87.6298),  # Downtown
            (41.8369, -87.6847),  # West Side
            (41.7687, -87.6746),  # South Side
        ]
        
        for i, (lat, lon) in enumerate(high_crime_areas):
            df[f'distance_to_hotspot_{i+1}'] = np.sqrt(
                (df['latitude'] - lat)**2 + (df['longitude'] - lon)**2
            ) * 69  # Approximate miles
        
        print(f"Created spatial features for {len(df)} records")
        
        return df
    
    def prepare_clustering_data(self, df: pd.DataFrame, 
                              features: List[str] = None) -> Tuple[np.ndarray, List[str]]:
        """Prepare data for clustering algorithms"""
        
        if features is None:
            features = [
                'latitude', 'longitude', 'distance_from_center',
                'grid_crime_count', 'crimes_same_location_24h',
                'distance_to_hotspot_1', 'distance_to_hotspot_2', 'distance_to_hotspot_3'
            ]
        
        # Select and clean features
        clustering_data = df[features].copy()
        
        # Handle missing values
        clustering_data = clustering_data.fillna(clustering_data.median())
        
        # Standardize features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(clustering_data)
        
        print(f"Prepared clustering data: {scaled_data.shape[0]} samples, {scaled_data.shape[1]} features")
        print(f"Features: {features}")
        
        return scaled_data, features, scaler
    
    def find_optimal_kmeans_clusters(self, data: np.ndarray, max_k: int = 15) -> Dict:
        """Find optimal number of clusters using elbow method and silhouette score"""
        
        print(f"Finding optimal K-means clusters (testing k=2 to {max_k})...")
        
        inertias = []
        silhouette_scores = []
        calinski_scores = []
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(data)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(data, cluster_labels))
            calinski_scores.append(calinski_harabasz_score(data, cluster_labels))
        
        # Find elbow point (simplified)
        # Calculate rate of change in inertia
        inertia_diff = np.diff(inertias)
        inertia_diff2 = np.diff(inertia_diff)
        elbow_k = k_range[np.argmax(inertia_diff2) + 2]  # +2 due to double diff
        
        # Best silhouette score
        best_silhouette_k = k_range[np.argmax(silhouette_scores)]
        
        results = {
            'k_range': list(k_range),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'calinski_scores': calinski_scores,
            'elbow_k': elbow_k,
            'best_silhouette_k': best_silhouette_k,
            'recommended_k': best_silhouette_k  # Use silhouette as primary metric
        }
        
        print(f"Optimal K analysis complete:")
        print(f"  Elbow method suggests: {elbow_k}")
        print(f"  Best silhouette score: {best_silhouette_k}")
        print(f"  Recommended K: {results['recommended_k']}")
        
        return results
    
    def fit_kmeans_clustering(self, data: np.ndarray, n_clusters: int = None) -> Dict:
        """Fit K-means clustering model"""
        
        if n_clusters is None:
            # Find optimal clusters first
            optimal_k = self.find_optimal_kmeans_clusters(data)
            n_clusters = optimal_k['recommended_k']
        
        print(f"Fitting K-means with {n_clusters} clusters...")
        
        # Fit K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(data)
        
        # Calculate metrics
        silhouette_avg = silhouette_score(data, cluster_labels)
        calinski_score = calinski_harabasz_score(data, cluster_labels)
        
        results = {
            'model': kmeans,
            'labels': cluster_labels,
            'n_clusters': n_clusters,
            'cluster_centers': kmeans.cluster_centers_,
            'metrics': {
                'silhouette_score': silhouette_avg,
                'calinski_harabasz_score': calinski_score,
                'inertia': kmeans.inertia_
            }
        }
        
        print(f"K-means clustering complete:")
        print(f"  Silhouette Score: {silhouette_avg:.3f}")
        print(f"  Calinski-Harabasz Score: {calinski_score:.1f}")
        
        return results
    
    def fit_dbscan_clustering(self, data: np.ndarray, eps: float = None, 
                             min_samples: int = None) -> Dict:
        """Fit DBSCAN clustering model"""
        
        # Auto-tune parameters if not provided
        if eps is None or min_samples is None:
            print("Auto-tuning DBSCAN parameters...")
            eps, min_samples = self._tune_dbscan_parameters(data)
        
        print(f"Fitting DBSCAN with eps={eps:.3f}, min_samples={min_samples}...")
        
        # Fit DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(data)
        
        # Calculate metrics
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        metrics = {
            'n_clusters': n_clusters,
            'n_noise_points': n_noise,
            'noise_ratio': n_noise / len(cluster_labels)
        }
        
        if n_clusters > 1:
            # Only calculate if we have valid clusters
            non_noise_mask = cluster_labels != -1
            if np.sum(non_noise_mask) > 1:
                metrics['silhouette_score'] = silhouette_score(
                    data[non_noise_mask], cluster_labels[non_noise_mask]
                )
        
        results = {
            'model': dbscan,
            'labels': cluster_labels,
            'eps': eps,
            'min_samples': min_samples,
            'metrics': metrics
        }
        
        print(f"DBSCAN clustering complete:")
        print(f"  Number of clusters: {n_clusters}")
        print(f"  Noise points: {n_noise} ({metrics['noise_ratio']:.1%})")
        
        return results
    
    def _tune_dbscan_parameters(self, data: np.ndarray) -> Tuple[float, int]:
        """Auto-tune DBSCAN parameters using k-distance graph"""
        
        # Calculate k-distances (k = 4 is common heuristic)
        k = 4
        distances = []
        
        for point in data:
            # Calculate distances to all other points
            point_distances = np.sqrt(np.sum((data - point)**2, axis=1))
            # Get k-th nearest distance
            kth_distance = np.sort(point_distances)[k]
            distances.append(kth_distance)
        
        distances = np.sort(distances)
        
        # Find elbow in k-distance graph
        # Use simple gradient approach
        gradients = np.gradient(distances)
        elbow_idx = np.argmax(gradients)
        eps = distances[elbow_idx]
        
        # Set min_samples based on dimensionality (common heuristic)
        min_samples = max(4, data.shape[1])
        
        return eps, min_samples
    
    def analyze_clusters(self, df: pd.DataFrame, cluster_labels: np.ndarray, 
                        model_type: str) -> Dict:
        """Analyze cluster characteristics and create hotspot profiles"""
        
        print(f"Analyzing {model_type} clusters...")
        
        # Add cluster labels to dataframe
        df_clustered = df.copy()
        df_clustered['cluster'] = cluster_labels
        
        # Cluster statistics
        cluster_stats = []
        
        unique_clusters = np.unique(cluster_labels)
        if -1 in unique_clusters:  # DBSCAN noise points
            unique_clusters = unique_clusters[unique_clusters != -1]
        
        for cluster_id in unique_clusters:
            cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
            
            stats = {
                'cluster_id': int(cluster_id),
                'size': len(cluster_data),
                'center_lat': cluster_data['latitude'].mean(),
                'center_lon': cluster_data['longitude'].mean(),
                'lat_std': cluster_data['latitude'].std(),
                'lon_std': cluster_data['longitude'].std(),
                'crime_types': cluster_data['primary_type'].value_counts().to_dict(),
                'top_crime_type': cluster_data['primary_type'].mode().iloc[0],
                'violent_crime_ratio': cluster_data['is_violent_crime'].mean(),
                'property_crime_ratio': cluster_data['is_property_crime'].mean(),
                'avg_distance_from_center': cluster_data['distance_from_center'].mean(),
                'peak_hour': cluster_data['hour'].mode().iloc[0],
                'weekend_ratio': cluster_data['is_weekend'].mean(),
                'districts': cluster_data['district_num'].dropna().unique().tolist(),
                'risk_score': self._calculate_risk_score(cluster_data)
            }
            
            cluster_stats.append(stats)
        
        # Sort by risk score
        cluster_stats = sorted(cluster_stats, key=lambda x: x['risk_score'], reverse=True)
        
        # Overall analysis
        analysis = {
            'model_type': model_type,
            'total_clusters': len(cluster_stats),
            'cluster_stats': cluster_stats,
            'hotspot_summary': self._create_hotspot_summary(cluster_stats)
        }
        
        print(f"Cluster analysis complete: {len(cluster_stats)} clusters identified")
        
        return analysis
    
    def _calculate_risk_score(self, cluster_data: pd.DataFrame) -> float:
        """Calculate risk score for a cluster based on crime characteristics"""
        
        # Factors contributing to risk score
        size_factor = min(len(cluster_data) / 100, 1.0)  # Normalize by 100 crimes
        violent_factor = cluster_data['is_violent_crime'].mean() * 2  # Weight violent crimes more
        density_factor = cluster_data['grid_crime_count'].mean() / cluster_data['grid_crime_count'].max()
        
        # Time-based risk (night crimes are riskier)
        night_factor = (cluster_data['hour'].between(22, 5)).mean()
        
        # Combine factors
        risk_score = (size_factor * 0.3 + 
                     violent_factor * 0.4 + 
                     density_factor * 0.2 + 
                     night_factor * 0.1) * 100
        
        return round(risk_score, 2)
    
    def _create_hotspot_summary(self, cluster_stats: List[Dict]) -> Dict:
        """Create summary of top hotspots"""
        
        # Top 5 highest risk clusters
        top_hotspots = cluster_stats[:5]
        
        summary = {
            'total_hotspots': len(cluster_stats),
            'high_risk_hotspots': len([c for c in cluster_stats if c['risk_score'] > 70]),
            'medium_risk_hotspots': len([c for c in cluster_stats if 40 <= c['risk_score'] <= 70]),
            'low_risk_hotspots': len([c for c in cluster_stats if c['risk_score'] < 40]),
            'top_5_hotspots': [
                {
                    'cluster_id': h['cluster_id'],
                    'location': f"({h['center_lat']:.4f}, {h['center_lon']:.4f})",
                    'size': h['size'],
                    'risk_score': h['risk_score'],
                    'top_crime': h['top_crime_type'],
                    'violent_ratio': f"{h['violent_crime_ratio']:.1%}"
                }
                for h in top_hotspots
            ]
        }
        
        return summary
    
    def save_clusters_to_db(self, df: pd.DataFrame, cluster_labels: np.ndarray, 
                           model_type: str, crime_type: str = 'ALL') -> str:
        """Save cluster results to database"""
        
        print(f"Saving {model_type} clusters to database...")
        
        # Prepare cluster data
        cluster_df = df.copy()
        cluster_df['cluster_id'] = cluster_labels
        cluster_df['model_type'] = model_type
        cluster_df['crime_type_filter'] = crime_type
        cluster_df['created_at'] = datetime.now()
        
        # Select relevant columns
        columns_to_save = [
            'id', 'date', 'primary_type', 'latitude', 'longitude',
            'cluster_id', 'model_type', 'crime_type_filter', 'created_at'
        ]
        
        cluster_df = cluster_df[columns_to_save]
        
        # Create clusters table
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS ml_features.crime_clusters (
            id TEXT,
            date TIMESTAMP,
            primary_type TEXT,
            latitude DECIMAL(10,8),
            longitude DECIMAL(11,8),
            cluster_id INTEGER,
            model_type TEXT,
            crime_type_filter TEXT,
            created_at TIMESTAMP,
            PRIMARY KEY (id, model_type, crime_type_filter)
        );
        """
        
        with self.engine.begin() as conn:
            conn.execute(text("CREATE SCHEMA IF NOT EXISTS ml_features"))
            conn.execute(text(create_table_sql))
        
        # Save clusters
        cluster_df.to_sql(
            'crime_clusters',
            self.engine,
            schema='ml_features',
            if_exists='append',
            index=False,
            method='multi'
        )
        
        result = f"Saved {len(cluster_df)} cluster records to ml_features.crime_clusters"
        print(result)
        return result
    
    def save_hotspot_analysis_to_db(self, analysis: Dict, crime_type: str = 'ALL') -> str:
        """Save hotspot analysis results to database"""
        
        print("Saving hotspot analysis to database...")
        
        # Prepare hotspot data
        hotspot_records = []
        
        for cluster_stat in analysis['cluster_stats']:
            record = {
                'cluster_id': cluster_stat['cluster_id'],
                'model_type': analysis['model_type'],
                'crime_type_filter': crime_type,
                'center_latitude': cluster_stat['center_lat'],
                'center_longitude': cluster_stat['center_lon'],
                'cluster_size': cluster_stat['size'],
                'risk_score': cluster_stat['risk_score'],
                'top_crime_type': cluster_stat['top_crime_type'],
                'violent_crime_ratio': cluster_stat['violent_crime_ratio'],
                'property_crime_ratio': cluster_stat['property_crime_ratio'],
                'avg_distance_from_center': cluster_stat['avg_distance_from_center'],
                'peak_hour': cluster_stat['peak_hour'],
                'weekend_ratio': cluster_stat['weekend_ratio'],
                'districts': ','.join(map(str, cluster_stat['districts'])),
                'created_at': datetime.now()
            }
            hotspot_records.append(record)
        
        hotspot_df = pd.DataFrame(hotspot_records)
        
        # Create hotspots table
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS ml_features.crime_hotspots (
            cluster_id INTEGER,
            model_type TEXT,
            crime_type_filter TEXT,
            center_latitude DECIMAL(10,8),
            center_longitude DECIMAL(11,8),
            cluster_size INTEGER,
            risk_score DECIMAL(5,2),
            top_crime_type TEXT,
            violent_crime_ratio DECIMAL(5,4),
            property_crime_ratio DECIMAL(5,4),
            avg_distance_from_center DECIMAL(8,2),
            peak_hour INTEGER,
            weekend_ratio DECIMAL(5,4),
            districts TEXT,
            created_at TIMESTAMP,
            PRIMARY KEY (cluster_id, model_type, crime_type_filter)
        );
        """
        
        with self.engine.begin() as conn:
            conn.execute(text("CREATE SCHEMA IF NOT EXISTS ml_features"))
            conn.execute(text(create_table_sql))
        
        # Save hotspots
        hotspot_df.to_sql(
            'crime_hotspots',
            self.engine,
            schema='ml_features',
            if_exists='append',
            index=False,
            method='multi'
        )
        
        result = f"Saved {len(hotspot_df)} hotspot records to ml_features.crime_hotspots"
        print(result)
        return result
    
    def run_full_spatial_analysis(self, crime_type: Optional[str] = None, 
                                 time_period: Optional[str] = None) -> Dict:
        """Run complete spatial clustering and hotspot analysis pipeline"""
        
        print(f"=== Starting Spatial Clustering Pipeline ===")
        print(f"Crime Type: {crime_type or 'ALL'}")
        print(f"Time Period: {time_period or 'ALL'}")
        
        # Load spatial data
        df = self.load_spatial_data(crime_type, time_period)
        
        if len(df) < 100:
            print("Insufficient data for clustering analysis")
            return {'error': 'Insufficient data'}
        
        # Create spatial features
        df = self.create_spatial_features(df)
        
        # Prepare clustering data
        clustering_data, features, scaler = self.prepare_clustering_data(df)
        
        # Fit K-means clustering
        kmeans_results = self.fit_kmeans_clustering(clustering_data)
        
        # Fit DBSCAN clustering
        dbscan_results = self.fit_dbscan_clustering(clustering_data)
        
        # Analyze clusters
        kmeans_analysis = self.analyze_clusters(df, kmeans_results['labels'], 'K-means')
        dbscan_analysis = self.analyze_clusters(df, dbscan_results['labels'], 'DBSCAN')
        
        # Save results to database
        kmeans_save_result = self.save_clusters_to_db(
            df, kmeans_results['labels'], 'K-means', crime_type or 'ALL'
        )
        dbscan_save_result = self.save_clusters_to_db(
            df, dbscan_results['labels'], 'DBSCAN', crime_type or 'ALL'
        )
        
        # Save hotspot analysis
        kmeans_hotspot_save = self.save_hotspot_analysis_to_db(
            kmeans_analysis, crime_type or 'ALL'
        )
        dbscan_hotspot_save = self.save_hotspot_analysis_to_db(
            dbscan_analysis, crime_type or 'ALL'
        )
        
        results = {
            'data_summary': {
                'total_records': len(df),
                'date_range': (df['date'].min(), df['date'].max()),
                'spatial_bounds': {
                    'lat_range': (df['latitude'].min(), df['latitude'].max()),
                    'lon_range': (df['longitude'].min(), df['longitude'].max())
                }
            },
            'kmeans': {
                'model_results': kmeans_results,
                'analysis': kmeans_analysis,
                'save_result': kmeans_save_result,
                'hotspot_save_result': kmeans_hotspot_save
            },
            'dbscan': {
                'model_results': dbscan_results,
                'analysis': dbscan_analysis,
                'save_result': dbscan_save_result,
                'hotspot_save_result': dbscan_hotspot_save
            },
            'features_used': features
        }
        
        print(f"=== Spatial Clustering Pipeline Complete ===")
        
        return results