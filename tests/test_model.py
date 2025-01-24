import os
import torch
import pytest
from torch import nn
from tsai.models.RNN import LSTM as tsai_LSTM

def test_model_architecture():
    """Test model architecture without loading weights"""
    # Create model with same architecture as training code
    model = nn.Sequential(
        tsai_LSTM(c_in=1, 
                 c_out=1, 
                 hidden_size=64,
                 n_layers=2),
        nn.Sigmoid()
    )
    
    # Test input/output dimensions - note tsai expects [batch_size, n_vars, seq_len]
    dummy_input = torch.randn(16, 1, 50)  # (batch, features, seq)
    output = model(dummy_input)
    
    assert output.shape == (16, 1), "Output should be (batch, 1)"
    assert torch.all(output >= 0) and torch.all(output <= 1), "Outputs should be probabilities"