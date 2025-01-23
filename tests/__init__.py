import os
# tests/test_model.py
import os
import torch
import pytest
from src.mlops_2025_floods_prediction.main import app, load_model, model
from src.mlops_2025_floods_prediction.train import LSTMModel  # Correct import path

# Define root paths
_TEST_ROOT = os.path.dirname(__file__)  # Root of the test folder
_PROJECT_ROOT = os.path.dirname(_TEST_ROOT)  # Root of the project
_PATH_DATA = os.path.join(_PROJECT_ROOT, "data")  # Path to the data folder
_PATH_RAW_DATA = os.path.join(_PATH_DATA, "raw")  # Path to the raw data folder
_PATH_PROCESSED_DATA = os.path.join(_PATH_DATA, "processed")  # Path to the processed data folder
_PATH_MODEL = os.path.join(_PROJECT_ROOT, "src", "mlops_2025_floods_prediction")  # Path to the model folder

# Export paths for use in other test files
__all__ = ["_TEST_ROOT", "_PROJECT_ROOT", "_PATH_DATA", "_PATH_RAW_DATA", "_PATH_PROCESSED_DATA", "_PATH_MODEL"]