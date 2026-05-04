import pandas as pd
import json

def check_drift():
    train_df = pd.read_csv("data/training_data.csv")
    
    live_data = []
    with open("logs/predictions.jsonl", "r") as f:
        for line in f:
            live_data.append(json.loads(line)["input"])
    
    live_df = pd.DataFrame(live_data)
    
    metrics = []
    drift_detected = False
    
    checks = [
        ("distance_km", 2.85),
        ("wind_speed_kmph", 7.88)
    ]
    
    for feat, thresh in checks:
        t_mean = train_df[feat].mean()
        l_mean = live_df[feat].mean()
        shift = abs(l_mean - t_mean)
        status = "ALERT" if shift > thresh else "OK"
        if status == "ALERT": drift_detected = True
        
        metrics.append({
            "feature": feat,
            "train_mean": round(t_mean, 2),
            "live_mean": round(l_mean, 2),
            "shift": round(shift, 2),
            "threshold": thresh,
            "status": status
        })

    results = {
        "total_predictions": len(live_df),
        "mean_prediction": 0.0, # Calculate from log file
        "drift_detected": drift_detected,
        "alerts": metrics
    }
    
    with open("results/step4_s5.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    check_drift()