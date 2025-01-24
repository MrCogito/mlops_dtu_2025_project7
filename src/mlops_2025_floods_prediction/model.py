import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super(LSTMModel, self).__init__()
        # Define LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        # Define a fully connected layer
        self.fc = nn.Linear(hidden_size, 1)
        # Define activation function
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Pass through LSTM
        x, _ = self.lstm(x)
        # Take only the last timestep
        x = x[:, -1, :]  # Shape: (batch_size, hidden_size)
        # Pass through fully connected layer
        x = self.fc(x)  # Shape: (batch_size, 1)
        return self.sigmoid(x)