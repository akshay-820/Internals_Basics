import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

# Configuration
mlflow.set_experiment("skydrop-flight-time-min")
DATA_PATH = "data/training_data.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2

def train():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=['flight_time_min'])
    y = df['flight_time_min']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    models = {
        "RandomForest": RandomForestRegressor(random_state=RANDOM_STATE),
        "GradientBoosting": GradientBoostingRegressor(random_state=RANDOM_STATE)
    }
    
    results = {"experiment_name": "skydrop-flight-time-min", "models": []}
    
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            mlflow.set_tag("project_phase", "model_selection")
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            metrics = {
                "mae": mean_absolute_error(y_test, preds),
                "rmse": np.sqrt(mean_squared_error(y_test, preds)),
                "r2": r2_score(y_test, preds),
                "mape": mean_absolute_percentage_error(y_test, preds)
            }
            
            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, name)
            
            res_entry = {"name": name}
            res_entry.update(metrics)
            results["models"].append(res_entry)

    best_model_info = min(results["models"], key=lambda x: x['mae'])
    results["best_model"] = best_model_info["name"]
    results["best_metric_name"] = "mae"
    results["best_metric_value"] = best_model_info["mae"]

    os.makedirs("results", exist_ok=True)
    with open("results/step1_s1.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    train()