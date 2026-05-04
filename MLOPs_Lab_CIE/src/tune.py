import pandas as pd
import json
import mlflow
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor # Assuming GB was best

def tune():
    df = pd.read_csv("data/training_data.csv")
    X, y = df.drop(columns=['flight_time_min']), df['flight_time_min']
    
    param_dist = {
        'n_estimators': [50, 150],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [3, 5, 10]
    }
    
    with mlflow.start_run(run_name="tuning-skydrop") as parent_run:
        model = GradientBoostingRegressor(random_state=42)
        search = RandomizedSearchCV(
            model, param_distributions=param_dist, 
            n_iter=10, cv=5, scoring='neg_mean_absolute_error', random_state=42
        )
        search.fit(X, y)
        
        # Log best model
        mlflow.sklearn.log_model(search.best_estimator_, "best_tuned_model")
        import joblib
        joblib.dump(search.best_estimator_, "models/best_model.pkl")

        output = {
            "search_type": "random",
            "n_folds": 5,
            "total_trials": 10,
            "best_params": search.best_params_,
            "best_mae": abs(search.best_score_), # Approximate
            "best_cv_mae": abs(search.best_score_),
            "parent_run_name": "tuning-skydrop"
        }
        
        with open("results/step2_s2.json", "w") as f:
            json.dump(output, f, indent=4)

if __name__ == "__main__":
    tune()