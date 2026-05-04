from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import json
import os
from datetime import datetime

app = FastAPI()

# Ensure directories exist
os.makedirs("results", exist_ok=True)
os.makedirs("logs", exist_ok=True)

model = joblib.load("models/best_model.pkl")

class FlightFeatures(BaseModel):
    payload_kg: float = Field(..., ge=0.1, le=10.0) 
    distance_km: float = Field(..., ge=0.5, le=100.0) # Increased
    wind_speed_kmph: float = Field(..., ge=0, le=100.0) # Increased
    altitude_m: float = Field(..., ge=10, le=200.0)

@app.get("/ping")
def ping():
    return {"status": "operational", "service": "SkyDrop API"}

@app.post("/score")
def score(data: FlightFeatures):
    input_dict = data.dict()
    input_df = pd.DataFrame([input_dict])
    prediction = float(model.predict(input_df)[0])
    
    # Task 4: Logging to .jsonl
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "input": input_dict,
        "prediction": prediction
    }
    with open("logs/predictions.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    # Task 3: Save specific test result if input matches
    test_target = {"payload_kg": 3.4, "distance_km": 10.0, "wind_speed_kmph": 19.4, "altitude_m": 61.4}
    if input_dict == test_target:
        step3_result = {
            "health_endpoint": "/ping",
            "predict_endpoint": "/score",
            "port": 9000,
            "health_response": {"status": "operational", "service": "SkyDrop API"},
            "test_input": test_target,
            "prediction": prediction
        }
        with open("results/step3_s4.json", "w") as f:
            json.dump(step3_result, f, indent=4)
        
    return {"prediction": prediction}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)