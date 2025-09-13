FROM apache/airflow:2.7.3-python3.11

USER root
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Install additional Python packages
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Additional packages for crime analytics
RUN pip install scikit-learn==1.3.2 plotly==5.17.0