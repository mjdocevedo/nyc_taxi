import os
import pandas as pd
from datetime import datetime

PROCESSED_DATA_DIR = "data/processed/"
FEATURES_DATA_DIR = "data/features/"
CONSOLIDATED_FILE_PATH = "data/processed/nyc_taxi_2024.csv"  # The consolidated CSV file for the year
FEATURES_FILE_PATH = "data/features/nyc_taxi_2024_features.csv"  # Output file with features

def create_features(df):
    # Time-based features
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    df["pickup_dayofweek"] = df["tpep_pickup_datetime"].dt.dayofweek
    df["pickup_month"] = df["tpep_pickup_datetime"].dt.month
    df["pickup_weekday"] = df["pickup_dayofweek"].apply(lambda x: 1 if x < 5 else 0)

    # Calculate trip duration in minutes
    df['trip_duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
    
    # Aggregated fares (optional, based on the rate code and payment type)
    df['total_fare'] = df['fare_amount'] + df['extra'] + df['mta_tax'] + df['improvement_surcharge'] + df['tolls_amount'] + df['congestion_surcharge'] + df['Airport_fee']
    
    # Drop columns that are unnecessary
    df = df.drop(columns=['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'store_and_fwd_flag'])
    
    return df

def build_features():
    # Load the consolidated dataset (this will contain all the data)
    if os.path.exists(CONSOLIDATED_FILE_PATH):
        print(f"Loading consolidated data from {CONSOLIDATED_FILE_PATH}")
        df = pd.read_csv(CONSOLIDATED_FILE_PATH)
        
        # Ensure that the datetime columns are correctly parsed
        df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
        df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
        
        # Create features
        df = create_features(df)
        
        # Save the features data to a new directory
        if not os.path.exists(FEATURES_DATA_DIR):
            os.makedirs(FEATURES_DATA_DIR)
        
        df.to_csv(FEATURES_FILE_PATH, index=False)
        print(f"Saved features data to {FEATURES_FILE_PATH}")
    else:
        print(f"Error: The consolidated file {CONSOLIDATED_FILE_PATH} does not exist.")

if __name__ == "__main__":
    build_features()
