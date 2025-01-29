from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from src.data.data_ingestion import download_2023_data, download_2024_data
