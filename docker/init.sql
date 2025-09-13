-- Create airflow database for Airflow metadata
CREATE DATABASE airflow;

-- Create crime_analytics database for our data
CREATE DATABASE crime_analytics;

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE airflow TO airflow;
GRANT ALL PRIVILEGES ON DATABASE crime_analytics TO airflow;