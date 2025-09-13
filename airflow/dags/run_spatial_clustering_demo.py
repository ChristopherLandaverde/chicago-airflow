#!/usr/bin/env python3
"""
Demo of spatial clustering pipeline with actual ML libraries
"""

import sys
sys.path.append('/opt/airflow/dags/src')

def run_spatial_clustering_demo():
    """Run spatial clustering on a subset of data for demo"""
    print("=== Spatial Clustering Pipeline Demo ===")
    
    try:
        from ml.spatial_clustering import CrimeSpatialAnalyzer
        
        # Initialize analyzer
        analyzer = CrimeSpatialAnalyzer()
        print("✓ CrimeSpatialAnalyzer initialized with ML libraries")
        
        # Check data availability
        with analyzer.engine.connect() as conn:
            result = conn.execute("SELECT COUNT(*) FROM ml_features.crime_features WHERE latitude IS NOT NULL AND longitude IS NOT NULL")
            count = result.fetchone()[0]
            print(f"✓ Found {count} records with coordinates for clustering")
        
        # Load spatial data (limit to recent data for demo)
        print("\nLoading spatial data for clustering...")
        spatial_data = analyzer.load_spatial_data(time_period='recent_month')
        print(f"✓ Loaded {len(spatial_data)} records for clustering")
        
        if len(spatial_data) < 100:
            print("⚠️  Using all available data (not enough recent data)")
            spatial_data = analyzer.load_spatial_data()
            print(f"✓ Loaded {len(spatial_data)} total records")
        
        # Create spatial features
        print("\nCreating spatial features...")
        spatial_data = analyzer.create_spatial_features(spatial_data)
        print(f"✓ Enhanced data with spatial features")
        
        # Prepare clustering data
        print("\nPreparing data for clustering algorithms...")
        clustering_data, features, scaler = analyzer.prepare_clustering_data(spatial_data)
        print(f"✓ Prepared {clustering_data.shape[0]} samples with {clustering_data.shape[1]} features")
        print(f"Features used: {features}")
        
        # Run K-means clustering
        print("\n=== K-means Clustering ===")
        kmeans_results = analyzer.fit_kmeans_clustering(clustering_data)
        print(f"✓ K-means completed with {kmeans_results['n_clusters']} clusters")
        print(f"  Silhouette Score: {kmeans_results['metrics']['silhouette_score']:.3f}")
        
        # Run DBSCAN clustering  
        print("\n=== DBSCAN Clustering ===")
        dbscan_results = analyzer.fit_dbscan_clustering(clustering_data)
        print(f"✓ DBSCAN completed")
        print(f"  Clusters found: {dbscan_results['metrics']['n_clusters']}")
        print(f"  Noise points: {dbscan_results['metrics']['n_noise_points']}")
        print(f"  Noise ratio: {dbscan_results['metrics']['noise_ratio']:.1%}")
        
        # Analyze clusters
        print("\n=== Cluster Analysis ===")
        kmeans_analysis = analyzer.analyze_clusters(spatial_data, kmeans_results['labels'], 'K-means')
        dbscan_analysis = analyzer.analyze_clusters(spatial_data, dbscan_results['labels'], 'DBSCAN')
        
        # Show K-means hotspots
        print(f"\n--- K-means Hotspots ---")
        for i, hotspot in enumerate(kmeans_analysis['hotspot_summary']['top_5_hotspots'][:3]):
            print(f"Hotspot {i+1}:")
            print(f"  Location: {hotspot['location']}")
            print(f"  Risk Score: {hotspot['risk_score']}")
            print(f"  Size: {hotspot['size']} crimes")
            print(f"  Top Crime: {hotspot['top_crime']}")
            print(f"  Violent Rate: {hotspot['violent_ratio']}")
        
        # Show DBSCAN hotspots
        print(f"\n--- DBSCAN Hotspots ---")
        for i, hotspot in enumerate(dbscan_analysis['hotspot_summary']['top_5_hotspots'][:3]):
            print(f"Hotspot {i+1}:")
            print(f"  Location: {hotspot['location']}")
            print(f"  Risk Score: {hotspot['risk_score']}")
            print(f"  Size: {hotspot['size']} crimes")
            print(f"  Top Crime: {hotspot['top_crime']}")
            print(f"  Violent Rate: {hotspot['violent_ratio']}")
        
        # Save results to database
        print(f"\n=== Saving Results ===")
        kmeans_save = analyzer.save_clusters_to_db(spatial_data, kmeans_results['labels'], 'K-means', 'DEMO')
        dbscan_save = analyzer.save_clusters_to_db(spatial_data, dbscan_results['labels'], 'DBSCAN', 'DEMO')
        
        kmeans_hotspot_save = analyzer.save_hotspot_analysis_to_db(kmeans_analysis, 'DEMO')
        dbscan_hotspot_save = analyzer.save_hotspot_analysis_to_db(dbscan_analysis, 'DEMO')
        
        print(f"✓ {kmeans_save}")
        print(f"✓ {dbscan_save}")
        print(f"✓ {kmeans_hotspot_save}")
        print(f"✓ {dbscan_hotspot_save}")
        
        return True
        
    except Exception as e:
        print(f"✗ Spatial clustering demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_spatial_clustering_demo()
    if success:
        print("\n🎉 Spatial Clustering Pipeline Demo completed successfully!")
        print("\n📊 Results saved to database:")
        print("  - ml_features.crime_clusters (cluster assignments)")
        print("  - ml_features.crime_hotspots (hotspot analysis)")
        print("\n🗺️  Ready for dashboard visualization!")
    else:
        print("\n❌ Spatial Clustering Demo failed")
    exit(0 if success else 1)