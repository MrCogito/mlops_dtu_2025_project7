import os
import torch
import pytest
from fastapi.testclient import TestClient
from src.mlops_2025_floods_prediction.main import app, load_model, model, LSTMModel


# Import paths from __init__.py
from tests import _PATH_MODEL

# Initialize FastAPI test client
client = TestClient(app)

def test_model_architecture():
    """Test model architecture without loading weights"""
    # Create model with expected parameters
    model = LSTMModel(input_size=1, hidden_size=64, num_layers=2)
    
    # Test input/output dimensions
    dummy_input = torch.randn(1, 50, 1)  # (batch, seq, features)
    output = model(dummy_input)
    
    assert output.shape == (1, 1), "Output should be (batch, 1)"
    assert 0 <= output.item() <= 1, "Output should be probability"

