import io
import json
from contextlib import asynccontextmanager
from typing import List

import anyio
import pandas as pd
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.mlops_2025_floods_prediction.model import LSTMModel

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    # 1) Load your trained LSTMModel
    model = LSTMModel()  # Initialize the model architecture
    model_path = os.path.join(os.path.dirname(__file__), "../../../models/lstm_model.pth")  # Replace with the actual path
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode

    # 2) Possibly load other resources (label maps, etc.)
    # e.g. with open("my_label_mapping.json") as f: label_map = json.load(f)

    yield
    # Cleanup at shutdown if needed

app = FastAPI(lifespan=lifespan)


class PredictionResponse(BaseModel):
    predictions: List[float]


@app.post("/predict_csv", response_model=PredictionResponse)
async def predict_csv(file: UploadFile = File(...)):
    """
    This endpoint accepts a CSV file, parses it into a DataFrame,
    runs inference on each row, and returns predictions as JSON.
    """
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
        X = torch.tensor(df[["precipitation"]].values, dtype=torch.float).unsqueeze(-1)  # Ensure proper shape

        # Run the model
        with torch.no_grad():
            outputs = model(X)  # shape [num_rows, 1]
            preds = outputs.squeeze(-1).tolist()  # Convert to a list of floats

        return {"predictions": preds}

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid CSV file or inference error: {e}"
        )