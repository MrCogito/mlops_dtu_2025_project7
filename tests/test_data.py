import os
from pathlib import Path
import pytest
from src.mlops_2025_floods_prediction.data import MyDataset

# Import paths from __init__.py
from tests import _PATH_RAW_DATA, _PATH_PROCESSED_DATA

def test_dataset_initialization():
    """Test if the dataset initializes correctly with the raw data path."""
    raw_data_path = Path(_PATH_RAW_DATA) / "Train.csv"
    dataset = MyDataset(raw_data_path)
    
    # Check if the dataset's data_path is correctly set
    assert dataset.data_path == raw_data_path, "Dataset raw data path is incorrect."

def test_preprocess_output_folder():
    """Test if the preprocess method creates the output folder."""
    raw_data_path = Path(_PATH_RAW_DATA) / "Train.csv"
    output_folder = Path(_PATH_PROCESSED_DATA)
    
    dataset = MyDataset(raw_data_path)
    dataset.preprocess(output_folder)
    
    # Check if the output folder exists
    assert output_folder.exists(), "Processed data folder was not created."

def test_preprocess_output_files():
    """Test if the preprocess method generates the expected output files."""
    raw_data_path = Path(_PATH_RAW_DATA) / "Train.csv"
    output_folder = Path(_PATH_PROCESSED_DATA)
    
    dataset = MyDataset(raw_data_path)
    dataset.preprocess(output_folder)
    
    # Check if the processed files exist
    processed_files = list(output_folder.glob("*.csv"))
    assert len(processed_files) > 0, "No processed files were found in the output folder."