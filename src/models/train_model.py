import os
import pandas as pd
import mlflow
import mlflow.sklearn
import boto3
from botocore.exceptions import NoCredentialsError
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
RAW_DATA_DIR = "data/raw/"
PROCESSED_DATA_DIR = "data/processed/"

# MinIO configuration
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
MINIO_ACCESS_KEY = os.getenv("MINIO_ROOT_USER")
MINIO_SECRET_KEY = os.getenv("MINIO_ROOT_PASSWORD")
BUCKET_NAME = os.getenv("MINIO_BUCKET")

# Initialize MinIO client
s3_client = boto3.client(
    "s3",
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY,
)

def download_from_minio(bucket, object_name, local_path):
    """Download a file from MinIO."""
    try:
        s3_client.download_file(bucket, object_name, local_path)
        print(f"Downloaded {object_name} from MinIO to {local_path}")
        return True
    except Exception as e:
        print(f"Error downloading {object_name}: {e}")
        return False

def load_data():
    """Load the train and test datasets from MinIO"""
    # Ensure the directory exists before downloading
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    X_train_path = os.path.join(PROCESSED_DATA_DIR, "X_train.csv")
    X_test_path = os.path.join(PROCESSED_DATA_DIR, "X_test.csv")
    y_train_path = os.path.join(PROCESSED_DATA_DIR, "y_train.csv")
    y_test_path = os.path.join(PROCESSED_DATA_DIR, "y_test.csv")
    
    # Download datasets from MinIO
    if not download_from_minio(BUCKET_NAME, "X_train.csv", X_train_path):
        return None, None, None, None  # Return None if download fails
    if not download_from_minio(BUCKET_NAME, "X_test.csv", X_test_path):
        return None, None, None, None  # Return None if download fails
    if not download_from_minio(BUCKET_NAME, "y_train.csv", y_train_path):
        return None, None, None, None  # Return None if download fails
    if not download_from_minio(BUCKET_NAME, "y_test.csv", y_test_path):
        return None, None, None, None  # Return None if download fails

    # Load the datasets into DataFrames
    try:
        X_train = pd.read_csv(X_train_path)
        X_test = pd.read_csv(X_test_path)
        y_train = pd.read_csv(y_train_path).values.ravel()  # Flatten y_train
        y_test = pd.read_csv(y_test_path).values.ravel()    # Flatten y_test
        print("Data loaded successfully.")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None

def train_model(X_train, X_test, y_train, y_test):
    """Train a RandomForest model"""
    # Train the RandomForest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error: {mae}")

    return model, mae

def log_model(model, mae):
    """Log the model and metrics into MLflow"""
    with mlflow.start_run():
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_metric("mae", mae)
        mlflow.sklearn.log_model(model, "model")
        print("Logged model and metrics to MLflow")

def main():
    # Load the data
    X_train, X_test, y_train, y_test = load_data()

    # Check if the data was loaded successfully
    if X_train is None:
        print("Error: Failed to download or load data.")
        return

    # Train the model and get evaluation metrics
    model, mae = train_model(X_train, X_test, y_train, y_test)

    # Log the model and metrics to MLflow
    log_model(model, mae)

if __name__ == "__main__":
    mlflow.set_experiment("nyc_taxi_fare_prediction")  # Set the experiment name
    main()
