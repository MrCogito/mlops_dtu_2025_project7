# test_model.py
import sys
import os
import pytest
import torch
from omegaconf import OmegaConf


# 1) Ensure we can find the project root, which contains "src/"
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
)

from src.mlops_2025_floods_prediction.model import LSTMModel

# 2) Build a config that matches the LSTMModel structure
cfg = OmegaConf.create({
    "wandb": {
        "project": "floods-prediction",
    },
    "model": {
        "input_size": 1,
        "hidden_size": 64,
        "num_layers": 2,
    },
    "training": {
        "batch_size": 32,
        "epochs": 1,
        "learning_rate": 0.001,
        "sequence_length": 5,
    },
    "paths": {
        "raw_folder": "data/raw",
        "train_file": "train.csv",
        "test_file": "test.csv",
        "predictions_output": "predictions.csv",
        "model_output": "lstm_model.pt",
    },
})

@pytest.mark.parametrize("num_batches", [2])
def test_training_loop(num_batches):
    """Checks that we can instantiate and train the model for a few mini-batches."""
    # Create an instance of the LSTM model
    model = LSTMModel(
        input_size=cfg.model.input_size,
        hidden_size=cfg.model.hidden_size,
        num_layers=cfg.model.num_layers,
    )
    model.train()

    dummy_x = torch.randn(num_batches, cfg.training.sequence_length, cfg.model.input_size)
    dummy_y = torch.ones(num_batches)  # pretend they're all positive

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    criterion = torch.nn.BCELoss()

    for _ in range(2):  # train for a few steps
        optimizer.zero_grad()
        out = model(dummy_x)
        loss = criterion(out.squeeze(), dummy_y)
        loss.backward()
        optimizer.step()

    assert loss.item() > 0, "Loss should be > 0 after a short training step"