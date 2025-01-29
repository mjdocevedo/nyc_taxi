import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

FEATURES_FILE_PATH = "data/features/nyc_taxi_2024_features.csv"  # Path to the features file

def load_data():
    """Load the features dataset"""
    df = pd.read_csv(FEATURES_FILE_PATH)
    print(f"Loaded data from {FEATURES_FILE_PATH}")
    return df

def train_model(df):
    """Train a RandomForest model"""
    # Define features and target
    X = df.drop(["total_fare", "fare_amount"], axis = 1)
    y = df['total_fare']

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
    df = load_data()

    # Train the model and get evaluation metrics
    model, mae = train_model(df)

    # Log the model and metrics to MLflow
    log_model(model, mae)

if __name__ == "__main__":
    mlflow.set_experiment("nyc_taxi_fare_prediction")  # Set the experiment name
    main()
