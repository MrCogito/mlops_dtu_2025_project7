import os
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.preprocessing import MinMaxScaler
from tsai.models.RNN import LSTM as tsai_LSTM
from torch import nn

# FastAPI app initialization
app = FastAPI()

# Define the request schema
class PredictionRequest(BaseModel):
    precipitation_sequence: list[float]

# LSTM model definition (should match the one in train.py)
class LSTMModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Get output from the last timestep
        x = self.fc(x)
        return self.sigmoid(x)

# Globals for model and scaler
model = None
scaler = MinMaxScaler()
model_path = "lstm_model.pth"  # Update this if the model is saved elsewhere

# Load the model on startup
@app.on_event("startup")
def load_model(model_path: str) -> nn.Module:
    model = nn.Sequential(
        tsai_LSTM(c_in=1, c_out=1, hidden_size=64, n_layers=2),
        nn.Sigmoid()
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Root endpoint
@app.get("/")
def root():
    return {"message": "Welcome to the Flood Probability Prediction API!"}

# Prediction endpoint
@app.post("/predict")
def predict(request: PredictionRequest):
    global model, scaler

    sequence = request.precipitation_sequence
    if len(sequence) != 50:  # Assuming sequence length of 50; update if needed
        raise HTTPException(status_code=400, detail="The precipitation sequence must contain exactly 50 values.")

    try:
        # Scale input
        sequence_array = np.array(sequence).reshape(-1, 1)
        scaled_sequence = scaler.fit_transform(sequence_array).reshape(1, -1, 1)
        
        # Convert to tensor
        input_tensor = torch.tensor(scaled_sequence, dtype=torch.float32)
        
        # Perform inference
        with torch.no_grad():
            prediction = model(input_tensor).item()
        
        return {"flood_probability": prediction}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
