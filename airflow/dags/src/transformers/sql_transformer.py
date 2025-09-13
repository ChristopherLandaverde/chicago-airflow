from sqlalchemy import create_engine, text
import os
from datetime import datetime

class SQLTransformer:
    """SQL-based data transformer for analytics pipeline"""
    
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
    
    def create_staging_tables(self):
        """Create staging tables with cleaned data"""
        
        staging_sql = """
        -- Create or replace staging view
        CREATE OR REPLACE VIEW analytics.stg_crime_data AS
        SELECT
            id,
            date::date as incident_date,
            record_data::json->>'case_number' as case_number,
            record_data::json->>'primary_type' as primary_type,
            record_data::json->>'description' as description,
            record_data::json->>'location_description' as location_description,
            record_data::json->>'arrest' as arrest,
            record_data::json->>'domestic' as domestic,
            record_data::json->>'beat' as beat,
            record_data::json->>'district' as district,
            record_data::json->>'ward' as ward,
            record_data::json->>'community_area' as community_area,
            record_data::json->>'year' as year,
            record_data::json->>'latitude' as latitude,
            record_data::json->>'longitude' as longitude,
            loaded_at,
            source
        FROM raw_crime_data
        WHERE id IS NOT NULL 
          AND date IS NOT NULL
          AND record_data IS NOT NULL;
        """
        
        try:
            with self.engine.begin() as conn:
                conn.execute(text(staging_sql))
            
            # Get count for verification
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM analytics.stg_crime_data"))
                count = result.scalar()
            
            print(f"Staging table created with {count} records")
            return f"Success: {count} records in staging"
            
        except Exception as e:
            print(f"Error creating staging tables: {e}")
            return f"Error: {str(e)}"
    
    def create_analytics_tables(self):
        """Create analytics tables for insights"""
        
        analytics_sql = """
        -- Crime type analysis
        CREATE TABLE IF NOT EXISTS analytics.crime_type_analysis AS
        SELECT
            primary_type,
            COUNT(*) as total_incidents,
            COUNT(CASE WHEN arrest = 'true' THEN 1 END) as arrests_made,
            COUNT(CASE WHEN domestic = 'true' THEN 1 END) as domestic_incidents,
            ROUND(
                (COUNT(CASE WHEN arrest = 'true' THEN 1 END)::numeric / COUNT(*)) * 100, 2
            ) as arrest_rate_percent,
            ROUND(
                (COUNT(CASE WHEN domestic = 'true' THEN 1 END)::numeric / COUNT(*)) * 100, 2
            ) as domestic_rate_percent,
            MIN(incident_date) as earliest_incident,
            MAX(incident_date) as latest_incident
        FROM analytics.stg_crime_data
        WHERE primary_type IS NOT NULL
        GROUP BY primary_type;
        
        -- Daily crime patterns
        CREATE TABLE IF NOT EXISTS analytics.daily_crime_patterns AS
        SELECT
            incident_date,
            COUNT(*) as total_incidents,
            COUNT(DISTINCT primary_type) as unique_crime_types,
            COUNT(CASE WHEN arrest = 'true' THEN 1 END) as arrests_made,
            COUNT(CASE WHEN domestic = 'true' THEN 1 END) as domestic_incidents,
            -- Most common crime type per day
            MODE() WITHIN GROUP (ORDER BY primary_type) as most_common_crime,
            -- Geographic spread
            COUNT(DISTINCT district) as districts_affected,
            COUNT(DISTINCT ward) as wards_affected
        FROM analytics.stg_crime_data
        GROUP BY incident_date
        ORDER BY incident_date DESC;
        
        -- Location analysis
        CREATE TABLE IF NOT EXISTS analytics.location_analysis AS
        SELECT
            location_description,
            district,
            ward,
            COUNT(*) as total_incidents,
            COUNT(CASE WHEN arrest = 'true' THEN 1 END) as arrests_made,
            ROUND(
                (COUNT(CASE WHEN arrest = 'true' THEN 1 END)::numeric / COUNT(*)) * 100, 2
            ) as arrest_rate_percent,
            -- Top crime types at this location
            MODE() WITHIN GROUP (ORDER BY primary_type) as most_common_crime
        FROM analytics.stg_crime_data
        WHERE location_description IS NOT NULL
        GROUP BY location_description, district, ward
        HAVING COUNT(*) >= 2  -- Filter locations with multiple incidents
        ORDER BY total_incidents DESC;
        
        -- Monthly trends with YoY comparison
        CREATE TABLE IF NOT EXISTS analytics.monthly_trends AS
        SELECT
            DATE_TRUNC('month', incident_date) as month,
            EXTRACT(year FROM incident_date) as year,
            EXTRACT(month FROM incident_date) as month_num,
            COUNT(*) as total_incidents,
            COUNT(CASE WHEN arrest = 'true' THEN 1 END) as arrests_made,
            COUNT(DISTINCT primary_type) as unique_crime_types,
            ROUND(
                (COUNT(CASE WHEN arrest = 'true' THEN 1 END)::numeric / COUNT(*)) * 100, 2
            ) as arrest_rate_percent,
            -- Moving averages
            AVG(COUNT(*)) OVER (
                ORDER BY DATE_TRUNC('month', incident_date) 
                ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
            ) as three_month_avg,
            -- YoY comparison (lag by 12 months)
            LAG(COUNT(*), 12) OVER (
                ORDER BY DATE_TRUNC('month', incident_date)
            ) as same_month_prev_year,
            -- YoY percentage change
            CASE 
                WHEN LAG(COUNT(*), 12) OVER (ORDER BY DATE_TRUNC('month', incident_date)) > 0 THEN
                    ROUND(
                        ((COUNT(*) - LAG(COUNT(*), 12) OVER (ORDER BY DATE_TRUNC('month', incident_date)))::numeric / 
                         LAG(COUNT(*), 12) OVER (ORDER BY DATE_TRUNC('month', incident_date))) * 100, 2
                    )
                ELSE NULL
            END as yoy_change_percent
        FROM analytics.stg_crime_data
        GROUP BY DATE_TRUNC('month', incident_date), EXTRACT(year FROM incident_date), EXTRACT(month FROM incident_date)
        ORDER BY month DESC;
        
        -- Year-over-Year Crime Type Analysis
        CREATE TABLE IF NOT EXISTS analytics.yoy_crime_type_analysis AS
        WITH yearly_crime_stats AS (
            SELECT
                EXTRACT(year FROM incident_date) as year,
                primary_type,
                COUNT(*) as incidents,
                COUNT(CASE WHEN arrest = 'true' THEN 1 END) as arrests,
                ROUND((COUNT(CASE WHEN arrest = 'true' THEN 1 END)::numeric / COUNT(*)) * 100, 2) as arrest_rate
            FROM analytics.stg_crime_data
            WHERE primary_type IS NOT NULL
            GROUP BY EXTRACT(year FROM incident_date), primary_type
        ),
        yoy_comparison AS (
            SELECT
                year,
                primary_type,
                incidents,
                arrests,
                arrest_rate,
                LAG(incidents) OVER (PARTITION BY primary_type ORDER BY year) as prev_year_incidents,
                LAG(arrests) OVER (PARTITION BY primary_type ORDER BY year) as prev_year_arrests,
                LAG(arrest_rate) OVER (PARTITION BY primary_type ORDER BY year) as prev_year_arrest_rate
            FROM yearly_crime_stats
        )
        SELECT
            year,
            primary_type,
            incidents,
            arrests,
            arrest_rate,
            prev_year_incidents,
            prev_year_arrests,
            prev_year_arrest_rate,
            -- YoY change calculations
            CASE 
                WHEN prev_year_incidents > 0 THEN
                    ROUND(((incidents - prev_year_incidents)::numeric / prev_year_incidents) * 100, 2)
                ELSE NULL
            END as yoy_incident_change_percent,
            CASE 
                WHEN prev_year_arrests > 0 THEN
                    ROUND(((arrests - prev_year_arrests)::numeric / prev_year_arrests) * 100, 2)
                ELSE NULL
            END as yoy_arrest_change_percent,
            -- Absolute changes
            incidents - prev_year_incidents as incident_change_absolute,
            arrests - prev_year_arrests as arrest_change_absolute,
            arrest_rate - prev_year_arrest_rate as arrest_rate_change_points
        FROM yoy_comparison
        WHERE prev_year_incidents IS NOT NULL  -- Only show years with comparison data
        ORDER BY year DESC, incidents DESC;
        
        -- Seasonal YoY Analysis
        CREATE TABLE IF NOT EXISTS analytics.seasonal_yoy_analysis AS
        WITH seasonal_stats AS (
            SELECT
                EXTRACT(year FROM incident_date) as year,
                CASE 
                    WHEN EXTRACT(month FROM incident_date) IN (12, 1, 2) THEN 'Winter'
                    WHEN EXTRACT(month FROM incident_date) IN (3, 4, 5) THEN 'Spring'
                    WHEN EXTRACT(month FROM incident_date) IN (6, 7, 8) THEN 'Summer'
                    WHEN EXTRACT(month FROM incident_date) IN (9, 10, 11) THEN 'Fall'
                END as season,
                COUNT(*) as total_incidents,
                COUNT(CASE WHEN arrest = 'true' THEN 1 END) as arrests_made,
                COUNT(DISTINCT primary_type) as unique_crime_types,
                -- Top crime type per season
                MODE() WITHIN GROUP (ORDER BY primary_type) as most_common_crime
            FROM analytics.stg_crime_data
            GROUP BY EXTRACT(year FROM incident_date), 
                     CASE 
                         WHEN EXTRACT(month FROM incident_date) IN (12, 1, 2) THEN 'Winter'
                         WHEN EXTRACT(month FROM incident_date) IN (3, 4, 5) THEN 'Spring'
                         WHEN EXTRACT(month FROM incident_date) IN (6, 7, 8) THEN 'Summer'
                         WHEN EXTRACT(month FROM incident_date) IN (9, 10, 11) THEN 'Fall'
                     END
        )
        SELECT
            year,
            season,
            total_incidents,
            arrests_made,
            unique_crime_types,
            most_common_crime,
            ROUND((arrests_made::numeric / total_incidents) * 100, 2) as arrest_rate_percent,
            -- YoY seasonal comparison
            LAG(total_incidents) OVER (PARTITION BY season ORDER BY year) as prev_year_incidents,
            CASE 
                WHEN LAG(total_incidents) OVER (PARTITION BY season ORDER BY year) > 0 THEN
                    ROUND(((total_incidents - LAG(total_incidents) OVER (PARTITION BY season ORDER BY year))::numeric / 
                           LAG(total_incidents) OVER (PARTITION BY season ORDER BY year)) * 100, 2)
                ELSE NULL
            END as yoy_seasonal_change_percent
        FROM seasonal_stats
        ORDER BY year DESC, 
                 CASE season 
                     WHEN 'Winter' THEN 1 
                     WHEN 'Spring' THEN 2 
                     WHEN 'Summer' THEN 3 
                     WHEN 'Fall' THEN 4 
                 END;
        
        -- Top Growing/Declining Crime Types YoY
        CREATE TABLE IF NOT EXISTS analytics.crime_trend_analysis AS
        WITH recent_years AS (
            SELECT DISTINCT EXTRACT(year FROM incident_date) as year
            FROM analytics.stg_crime_data
            ORDER BY year DESC
            LIMIT 3  -- Last 3 years
        ),
        crime_growth AS (
            SELECT
                primary_type,
                COUNT(CASE WHEN EXTRACT(year FROM incident_date) = (SELECT MAX(year) FROM recent_years) THEN 1 END) as current_year_incidents,
                COUNT(CASE WHEN EXTRACT(year FROM incident_date) = (SELECT MAX(year) - 1 FROM recent_years) THEN 1 END) as prev_year_incidents,
                COUNT(CASE WHEN EXTRACT(year FROM incident_date) = (SELECT MAX(year) - 2 FROM recent_years) THEN 1 END) as two_years_ago_incidents
            FROM analytics.stg_crime_data
            WHERE primary_type IS NOT NULL
              AND EXTRACT(year FROM incident_date) IN (SELECT year FROM recent_years)
            GROUP BY primary_type
        )
        SELECT
            primary_type,
            current_year_incidents,
            prev_year_incidents,
            two_years_ago_incidents,
            -- YoY change
            CASE 
                WHEN prev_year_incidents > 0 THEN
                    ROUND(((current_year_incidents - prev_year_incidents)::numeric / prev_year_incidents) * 100, 2)
                ELSE NULL
            END as yoy_change_percent,
            -- 2-year trend
            CASE 
                WHEN two_years_ago_incidents > 0 THEN
                    ROUND(((current_year_incidents - two_years_ago_incidents)::numeric / two_years_ago_incidents) * 100, 2)
                ELSE NULL
            END as two_year_change_percent,
            -- Trend classification
            CASE 
                WHEN prev_year_incidents > 0 AND ((current_year_incidents - prev_year_incidents)::numeric / prev_year_incidents) > 0.1 THEN 'Growing'
                WHEN prev_year_incidents > 0 AND ((current_year_incidents - prev_year_incidents)::numeric / prev_year_incidents) < -0.1 THEN 'Declining'
                ELSE 'Stable'
            END as trend_category
        FROM crime_growth
        WHERE prev_year_incidents > 0  -- Only crimes with previous year data
        ORDER BY yoy_change_percent DESC NULLS LAST;
        """
        
        try:
            with self.engine.begin() as conn:
                # Drop existing tables to refresh data
                conn.execute(text("DROP TABLE IF EXISTS analytics.crime_type_analysis CASCADE"))
                conn.execute(text("DROP TABLE IF EXISTS analytics.daily_crime_patterns CASCADE"))
                conn.execute(text("DROP TABLE IF EXISTS analytics.location_analysis CASCADE"))
                conn.execute(text("DROP TABLE IF EXISTS analytics.monthly_trends CASCADE"))
                conn.execute(text("DROP TABLE IF EXISTS analytics.yoy_crime_type_analysis CASCADE"))
                conn.execute(text("DROP TABLE IF EXISTS analytics.seasonal_yoy_analysis CASCADE"))
                conn.execute(text("DROP TABLE IF EXISTS analytics.crime_trend_analysis CASCADE"))
                
                # Create new tables
                conn.execute(text(analytics_sql))
            
            # Get counts for verification
            tables_created = []
            table_names = [
                'crime_type_analysis', 'daily_crime_patterns', 'location_analysis', 
                'monthly_trends', 'yoy_crime_type_analysis', 'seasonal_yoy_analysis', 
                'crime_trend_analysis'
            ]
            
            with self.engine.connect() as conn:
                for table in table_names:
                    result = conn.execute(text(f"SELECT COUNT(*) FROM analytics.{table}"))
                    count = result.scalar()
                    tables_created.append(f"{table}: {count} records")
            
            result_msg = "Analytics tables created: " + ", ".join(tables_created)
            print(result_msg)
            return result_msg
            
        except Exception as e:
            print(f"Error creating analytics tables: {e}")
            return f"Error: {str(e)}"
    
    def get_analytics_summary(self):
        """Get summary of analytics data"""
        
        summary_sql = """
        SELECT 
            'Raw Data' as table_name,
            COUNT(*) as record_count
        FROM raw_crime_data
        
        UNION ALL
        
        SELECT 
            'Staging Data' as table_name,
            COUNT(*) as record_count
        FROM analytics.stg_crime_data
        
        UNION ALL
        
        SELECT 
            'Crime Types' as table_name,
            COUNT(*) as record_count
        FROM analytics.crime_type_analysis
        
        UNION ALL
        
        SELECT 
            'Daily Patterns' as table_name,
            COUNT(*) as record_count
        FROM analytics.daily_crime_patterns
        
        UNION ALL
        
        SELECT 
            'Location Analysis' as table_name,
            COUNT(*) as record_count
        FROM analytics.location_analysis;
        """
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(summary_sql))
                summary = result.fetchall()
            
            return [dict(row._mapping) for row in summary]
            
        except Exception as e:
            print(f"Error getting summary: {e}")
            return []