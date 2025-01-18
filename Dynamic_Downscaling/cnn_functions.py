import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

def process_data(window_size,lead_time,n,seed = 1234):
    # Define the Nino 3.4 region for label calculation
    torch.manual_seed(seed)
    nino34_lat_range = slice(-5, 5)
    nino34_lon_range = slice(190, 240)
    sst_nino34 = sst_broad.sel(lat=nino34_lat_range, lon=nino34_lon_range)

    # Downsample spatially for the CNN input
    sst_downsampled = sst_broad.isel(lat=slice(0, None, n), lon=slice(0, None, n))

    # Resample to monthly means if not already monthly
    sst_downsampled = sst_downsampled.resample(time='M').mean()
    sst_nino34 = sst_nino34.resample(time='M').mean()

    # Calculate anomalies: deviations from monthly climatology
    sst_anomalies = sst_downsampled.groupby("time.month") - sst_downsampled.groupby("time.month").mean(dim="time")
    sst_nino34_anomalies = sst_nino34.groupby("time.month") - sst_nino34.groupby("time.month").mean(dim="time")

    # Normalize the anomalies (standardization)
    sst_normalized = (sst_anomalies - sst_anomalies.mean(dim="time")) / sst_anomalies.std(dim="time")

    data_windows = []
    labels = []

    for i in range(len(sst_normalized['time']) - window_size - lead_time):
        # Input: SST anomalies for the current window (months i to i + window_size - 1)
        window_data = sst_normalized.isel(time=slice(i, i + window_size)).values
        data_windows.append(window_data)

        # Output (Label): Anomaly of the Nino 3.4 region at i + window_size + lead_time - 1
        future_anomaly = sst_nino34_anomalies.isel(
            time=slice(i + window_size, i + window_size + lead_time)
        ).mean(dim=["time", "lat", "lon"]).values
        label = 1 if future_anomaly > 0.5 else 0
        labels.append(label)

    # Convert lists to numpy arrays
    data_windows = np.array(data_windows)
    labels = np.array(labels)
    return data_windows,labels


class SimpleCNN(nn.Module):
    def __init__(self, input_channels=1):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=4, padding=1)  # First convolutional layer
        self.conv2 = nn.Conv2d(16,16, kernel_size=4, padding=1)               # Second convolutional layer
        self.conv3 = nn.Conv2d(16,16, kernel_size=4, padding=1)              # New third convolutional layer
        self.pool = nn.MaxPool2d(kernel_size=4, stride=2)                    # Max pooling layer
        self.dropout = nn.Dropout(p=0.1)

        # Dynamically calculate the flattened size
        with torch.no_grad():
            sample_input = torch.zeros(1, input_channels, 30, 90)  # Adjust channels
            sample_output = self._forward_features(sample_input)
            flattened_size = sample_output.view(-1).size(0)
        print(f"Flattened size for fc1: {flattened_size}")

        self.fc1 = nn.Linear(flattened_size, 64)
        self.fc2 = nn.Linear(64, 1)

    def _forward_features(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # First conv-pool
        x = self.pool(torch.relu(self.conv2(x)))  # Second conv-pool
        x = self.pool(torch.relu(self.conv3(x)))  # New third conv-pool
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)  # Flatten dynamically
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)  # Logits for BCEWithLogitsLoss
        return x


# Define the custom dataset
class SSTDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class SimpleCNN3D(nn.Module):
    def __init__(self, input_channels=1, input_depth=1, input_height=30, input_width=90):
        super(SimpleCNN3D, self).__init__()
        self.conv1 = nn.Conv3d(
            in_channels=input_channels,
            out_channels=16,
            kernel_size=(3, 3, 3),
            padding=1
        )
        self.conv2 = nn.Conv3d(
            in_channels=16,
            out_channels=32,
            kernel_size=(3, 3, 3),
            padding=1
        )
        # Adjusted pooling layer to preserve depth dimension
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dropout = nn.Dropout(p=0.5)

        # Dynamically calculate the flattened size
        with torch.no_grad():
            sample_input = torch.zeros(1, input_channels, input_depth, input_height, input_width)
            sample_output = self._forward_features(sample_input)
            flattened_size = sample_output.view(-1).size(0)
        print(f"Flattened size for fc1: {flattened_size}")

        self.fc1 = nn.Linear(flattened_size, 64)
        self.fc2 = nn.Linear(64, 1)

    def _forward_features(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # Pooling only over spatial dimensions
        x = self.pool(torch.relu(self.conv2(x)))  # Pooling only over spatial dimensions
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)  # Flatten dynamically
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
