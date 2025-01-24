import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import wandb
import hydra
from omegaconf import DictConfig
import sys

# Get the absolute path to the project root (mlops_dtu_2025_project7)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.insert(0, project_root)

# Now import your modules
from src.mlops_2025_floods_prediction.logging_util import setup_logging
import logging

# tsai imports
from tsai.models.RNN import LSTM as tsai_LSTM

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Hydra config decorator
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Initialize W&B using Hydra config
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config={
            "architecture": cfg.model.architecture,
            "batch_size": cfg.training.batch_size,
            "epochs": cfg.training.epochs,
            "learning_rate": cfg.training.learning_rate,
            "sequence_length": cfg.training.sequence_length,
            "hidden_size": cfg.model.hidden_size,
        },
    )

    # Path configurations
    raw_folder = os.path.join(project_root, cfg.paths.raw_folder)
    train_file_path = os.path.join(raw_folder, cfg.paths.train_file)
    test_file_path = os.path.join(raw_folder, cfg.paths.test_file)

    # Load data
    test_data = pd.read_csv(test_file_path)
    train_data = pd.read_csv(train_file_path)

    # Process event IDs and timestamps
    for df in [train_data, test_data]:
        df["event_id"] = df["event_id"].apply(lambda x: "_".join(x.split("_")[:2]))
        df["event_idx"] = df.groupby("event_id", sort=False).ngroup()
        df["event_t"] = df.groupby("event_id").cumcount()

    # Sort data by event and time
    train_df = train_data.sort_values(by=["event_id", "event_t"])
    test_df = test_data.sort_values(by=["event_id", "event_t"])

    # Create sequences for training
    sequence_length = cfg.training.sequence_length
    X_train, y_train = [], []

    for event_id, group in train_df.groupby("event_id"):
        precip_values = group["precipitation"].values
        labels = group["label"].values

        for i in range(len(precip_values) - sequence_length):
            X_train.append(precip_values[i : i + sequence_length])
            y_train.append(labels[i + sequence_length - 1])

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Normalize precipitation values
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    # Create DataLoader
    batch_size = cfg.training.batch_size
    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=batch_size,
        shuffle=True,
    )

    # Initialize tsai LSTM model with sigmoid activation
    model = nn.Sequential(
        tsai_LSTM(c_in=1, 
                 c_out=1, 
                 hidden_size=cfg.model.hidden_size,
                 n_layers=cfg.model.num_layers),
        nn.Sigmoid()
    )

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)

    # Training Loop
    epochs = cfg.training.epochs
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            # Reshape input for tsai LSTM: [batch_size, 1, sequence_length]
            outputs = model(X_batch.unsqueeze(1))
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        wandb.log({"epoch": epoch + 1, "loss": avg_loss})
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    # Prepare test sequences
    X_test = []
    for event_id, group in test_df.groupby("event_id"):
        precip_values = group["precipitation"].values
        for i in range(len(precip_values) - sequence_length):
            X_test.append(precip_values[i : i + sequence_length])

    X_test = np.array(X_test)
    X_test = scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)

    # Predict on test data
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test_tensor.unsqueeze(1)).squeeze().numpy()

    # Save predictions
    test_df = test_df.iloc[: len(test_predictions)]
    test_df["flood_probability"] = test_predictions
    test_df.to_csv(cfg.paths.predictions_output, index=False)

    # Log artifacts and finish
    wandb.save(cfg.paths.predictions_output)
    torch.save(model.state_dict(), cfg.paths.model_output)
    wandb.save(cfg.paths.model_output)
    wandb.finish()

if __name__ == "__main__":
    main()