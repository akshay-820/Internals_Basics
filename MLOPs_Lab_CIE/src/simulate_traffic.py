import requests
import pandas as pd
import time

API_URL = "http://localhost:9000/score"

def run_simulation():
    # Load data
    train_df = pd.read_csv("data/training_data.csv").drop(columns=['flight_time_min'])
    new_df = pd.read_csv("data/new_data.csv").drop(columns=['flight_time_min'])
    
    print("Sending 30 normal requests...")
    # Send 30 requests from training distribution
    for i in range(30):
        row = train_df.iloc[i % len(train_df)].to_dict()
        requests.post(API_URL, json=row)
    
    print("Sending 20 drifted requests...")
    # Send 20 requests from shifted distribution
    for i in range(20):
        row = new_df.iloc[i % len(new_df)].to_dict()
        requests.post(API_URL, json=row)
        
    print("Traffic simulation complete.")

if __name__ == "__main__":
    run_simulation()