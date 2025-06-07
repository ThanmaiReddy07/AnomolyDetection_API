from fastapi import FastAPI
import joblib
import os
import numpy as np
from pydantic import BaseModel

app = FastAPI()

# Load trained model and scaler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "isolation_forest.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

# Define the expected JSON input structure
class DataInput(BaseModel):
    data: list  # This ensures FastAPI expects a JSON list

@app.post("/predict")
async def predict_anomaly(input_data: DataInput):
    scaled_data = scaler.transform([input_data.data])  # Apply scaling
    prediction = model.predict(scaled_data)  # Make prediction
    return {"anomaly_detected": int(prediction[0])}  # Return result
@app.get("/")
def home():
    return {"message": "Welcome to Thanmai's Anomaly Detection API!"}
import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)