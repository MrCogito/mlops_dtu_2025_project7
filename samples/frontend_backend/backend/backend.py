import io
import json
import os
import sys
from contextlib import asynccontextmanager
from typing import List

import pandas as pd
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from sklearn.preprocessing import MinMaxScaler

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.mlops_2025_floods_prediction.model import LSTMModel

# Global variables for the model and scaler
model = None
scaler = MinMaxScaler()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan function to manage application startup and shutdown.
    """
    global model

    print("Application starting...")
    # Load your trained LSTMModel
    model = LSTMModel()  # Initialize the model architecture
    model_path = os.path.join(os.path.dirname(__file__), "../../../models/lstm_model.pth")  # Update this path if necessary
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found. Train the model first.")
    
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()  # Set the model to evaluation mode
    print("Model loaded successfully.")

    yield  # Application is running

    print("Application shutting down...")

# Initialize FastAPI application
app = FastAPI(lifespan=lifespan)


# Pydantic models for request and response validation
class PredictionRequest(BaseModel):
    precipitation_sequence: List[float]


class PredictionResponse(BaseModel):
    predictions: List[float]


@app.get("/")
def root():
    """
    Root endpoint for the API.
    """
    return {"message": "Welcome to the Flood Prediction API!"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Predict endpoint for single-sequence predictions.
    """
    global model, scaler

    sequence = request.precipitation_sequence
    if len(sequence) != 50:  # Assuming sequence length of 50; update as necessary
        raise HTTPException(status_code=400, detail="The precipitation sequence must contain exactly 50 values.")

    try:
        # Scale the input sequence
        sequence_array = pd.DataFrame(sequence).values.reshape(-1, 1)
        scaled_sequence = scaler.fit_transform(sequence_array).reshape(1, -1, 1)

        # Convert to PyTorch tensor
        input_tensor = torch.tensor(scaled_sequence, dtype=torch.float32)

        # Run the model for inference
        with torch.no_grad():
            prediction = model(input_tensor).item()

        return {"predictions": [prediction]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")


@app.post("/predict_csv", response_model=PredictionResponse)
async def predict_csv(file: UploadFile = File(...)):
    """
    Endpoint for batch predictions from a CSV file.
    """
    global model

    contents = await file.read()
    try:
        # Parse CSV with pandas
        df = pd.read_csv(io.BytesIO(contents))

        # Validate required columns
        required_cols = ["event_id", "precipitation"]
        if not all(col in df.columns for col in required_cols):
            raise HTTPException(
                status_code=400,
                detail=f"CSV must include columns: {required_cols}",
            )

        # Ensure the 'precipitation' column contains numeric data
        df["precipitation"] = pd.to_numeric(df["precipitation"], errors="coerce")
        if df["precipitation"].isnull().any():
            raise HTTPException(
                status_code=400,
                detail="Invalid values in 'precipitation' column. All values must be numeric.",
            )

        # Convert to PyTorch tensor
        X = torch.tensor(df[["precipitation"]].values, dtype=torch.float).unsqueeze(-1)

        # Run the model for batch predictions
        with torch.no_grad():
            outputs = model(X)
            preds = outputs.squeeze(-1).tolist()  # Convert predictions to a list of floats

        return {"predictions": preds}

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid CSV file or inference error: {e}"
        )