# Step 1: Download the data file

import xarray as xr

# Load the dataset
file_path = "sst_data.nc"  # Path to the uploaded file
sst_broad = xr.open_dataset(file_path)['sst']

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os

from sklearn.model_selection import KFold
import csv
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def process_data(window_size, lead_time, n, seed=1975):
    # Set seed for reproducibility
    torch.manual_seed(seed)
    
    # Define the Nino 3.4 region for label calculation
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
    
    T = len(sst_normalized['time'])
    
    # Compute labels for all time steps
    labels_full = []
    for t in range(T - lead_time):
        future_anomaly = sst_nino34_anomalies.isel(time=t + lead_time).mean(dim=["lat", "lon"]).values
        label = 1 if future_anomaly > 0.5 else 0
        labels_full.append(label)
    
    # Compute time since last El Niño event
    time_since_last_el_nino = []
    last_el_nino_time = -1
    for t in range(len(labels_full)):
        if labels_full[t] == 1:
            last_el_nino_time = t
            time_since_last_el_nino.append(0)
        else:
            if last_el_nino_time == -1:
                time_since_last_el_nino.append(-1)  # Indicates no prior El Niño event
            else:
                time_since_last_el_nino.append(t - last_el_nino_time)
    
    # Extract month for each time step
    months = [sst_normalized['time'][t].dt.month.values for t in range(T - lead_time)]
    
    data_windows = []
    labels = []
    additional_features = []
    
    for i in range(T - window_size - lead_time):
        # Input window: SST anomalies for current time and previous points
        window_data = sst_anomalies.isel(time=slice(i, i + window_size)).values
        data_windows.append(window_data)
        
        # Label: El Niño event at t + lead_time
        label = labels_full[i + window_size - 1]
        labels.append(label)
        
         # Additional features
        tsle = time_since_last_el_nino[i + window_size - 1]
        month = months[i + window_size - 1]
        additional_features.append([tsle, month])

    # Convert lists to numpy arrays
    data_windows = np.array(data_windows)
    labels = np.array(labels)
    additional_features = np.array(additional_features)
    
    return data_windows, labels, additional_features


# Define the CNN model
class SimpleCNN(nn.Module):
    def __init__(self, input_channels=1, num_additional_features=2):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=4, padding=1)  # First convolutional layer
        self.conv2 = nn.Conv2d(16, 16, kernel_size=4, padding=1)              # Second convolutional layer
        self.conv3 = nn.Conv2d(16, 16, kernel_size=4, padding=1)              # Third convolutional layer
        self.pool = nn.MaxPool2d(kernel_size=4, stride=2)                     # Max pooling layer
        self.dropout = nn.Dropout(p=0.1)

        # Dynamically calculate the flattened size
        with torch.no_grad():
            sample_input = torch.zeros(1, input_channels, 30, 90)  # Adjust input dimensions as needed
            sample_output = self._forward_features(sample_input)
            flattened_size = sample_output.view(-1).size(0)

        # Adjust flattened size to include additional features
        adjusted_size = flattened_size + num_additional_features
        print(f"Adjusted flattened size for fc1: {adjusted_size}")

        self.fc1 = nn.Linear(adjusted_size, 64)
        self.fc2 = nn.Linear(64, 1)

    def _forward_features(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # First conv-pool
        x = self.pool(torch.relu(self.conv2(x)))  # Second conv-pool
        x = self.pool(torch.relu(self.conv3(x)))  # Third conv-pool
        return x

    def forward(self, x, additional_features):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)  # Flatten dynamically
        # Concatenate additional features to the flattened CNN features
        x = torch.cat((x, additional_features), dim=1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)  # Logits for BCEWithLogitsLoss
        return x


class SSTDataset(Dataset):
    def __init__(self, data, labels, additional_features):
        self.data = data
        self.labels = labels
        self.additional_features = additional_features

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Convert additional features to float32
        additional_feats = torch.tensor(self.additional_features[idx], dtype=torch.float32)
        return self.data[idx], self.labels[idx], additional_feats



class SimpleCNN3D(nn.Module):
    def __init__(self, input_channels=1, input_depth=5, input_height=30, input_width=90, num_additional_features=2):
        super(SimpleCNN3D, self).__init__()
        # Separate convolution for the first lag
        self.first_lag_conv = nn.Conv2d(
            in_channels=input_channels,
            out_channels=16,
            kernel_size=3,
            padding=1
        )
        self.first_lag_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Standard 3D convolutions for other lags
        self.conv1 = nn.Conv3d(
            in_channels=input_channels,
            out_channels=8,
            kernel_size=(2, 3, 3),
            padding=1
        )
        self.conv2 = nn.Conv3d(
            in_channels=8,
            out_channels=16,
            kernel_size=(2, 3, 3),
            padding=1
        )
        self.pool = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.dropout = nn.Dropout(p=0.1)

        # Dynamically calculate the flattened size
        with torch.no_grad():
            # First lag feature size
            sample_first_lag = torch.zeros(1, input_channels, input_height, input_width)
            first_lag_output = self.first_lag_pool(self.first_lag_conv(sample_first_lag))
            flattened_first_lag_size = first_lag_output.view(-1).size(0)

            # 3D convolution feature size
            sample_input = torch.zeros(1, input_channels, input_depth, input_height, input_width)
            sample_output = self._forward_features(sample_input)
            flattened_3d_size = sample_output.view(-1).size(0)

        # Total flattened size
        adjusted_size = flattened_first_lag_size + flattened_3d_size + num_additional_features
        print(f"Adjusted flattened size for fc1: {adjusted_size}")

        self.fc1 = nn.Linear(adjusted_size, 64)
        self.fc2 = nn.Linear(64, 1)

    def _forward_features(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # Pooling only over spatial dimensions
        x = self.pool(torch.relu(self.conv2(x)))  # Pooling only over spatial dimensions
        return x

    def forward(self, x, additional_features):
        # Separate the first lag and process it independently
        first_lag = x[:, :, 0, :, :]  # Shape: (batch, channels, height, width)
        other_lags = x[:, :, 1:, :, :]  # Shape: (batch, channels, depth-1, height, width)

        # Process first lag
        first_lag_features = self.first_lag_pool(torch.relu(self.first_lag_conv(first_lag)))
        first_lag_features = first_lag_features.view(first_lag_features.size(0), -1)  # Flatten

        # Process other lags through 3D CNN
        other_lag_features = self._forward_features(other_lags)
        other_lag_features = other_lag_features.view(other_lag_features.size(0), -1)  # Flatten

        # Concatenate features
        combined_features = torch.cat((first_lag_features, other_lag_features, additional_features), dim=1)
        combined_features = self.dropout(torch.relu(self.fc1(combined_features)))
        output = self.fc2(combined_features)
        return output




# Path for the results file
results_file = "results.csv"

# Ensure the file exists, and write the header if it doesn't
if not os.path.exists(results_file):
    with open(results_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "window_size", "lead_time", "accuracy", "precision", 
            "recall", "f1_score", "roc_auc"
        ])


def run_experiment_with_cv(window_size, lead_time, epochs=200, n=4, device='cpu', k_folds=5):
    # Prepare data
    data_windows, labels, additional_features = process_data(window_size, lead_time, n)
    X = torch.tensor(data_windows, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    additional_features = torch.tensor(additional_features, dtype=torch.float32)
    
    if window_size == 1:
        X = X.squeeze(1).unsqueeze(1)
        model_class = SimpleCNN
        model_args = {"input_channels": 1}
    else:
        X = X.unsqueeze(1)
        model_class = SimpleCNN3D
        model_args = {
            "input_channels": 1,
            "input_depth": window_size,
            "input_height": X.shape[3],
            "input_width": X.shape[4],
            "num_additional_features": additional_features.shape[1]
        }
    
    dataset = SSTDataset(X, y, additional_features)
    
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"Fold {fold + 1}/{k_folds}")
        model = model_class(**model_args).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss()
        
        train_data = torch.utils.data.Subset(dataset, train_idx)
        val_data = torch.utils.data.Subset(dataset, val_idx)
        
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
        
        # Track the best validation performance
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(epochs):
            # Training loop
            model.train()
            for inputs, labels, add_feats in train_loader:
                inputs, labels, add_feats = inputs.to(device), labels.to(device), add_feats.to(device)
                labels = labels.view(-1, 1)
                optimizer.zero_grad()
                outputs = model(inputs, add_feats)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            # Validation loop
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, labels, add_feats in val_loader:
                    inputs, labels, add_feats = inputs.to(device), labels.to(device), add_feats.to(device)
                    outputs = model(inputs, add_feats)
                    val_loss += criterion(outputs, labels).item()
            val_loss /= len(val_loader)
            
            # Save the best model state
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
        
        # Load the best model state after training
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        # Evaluate the best model on the validation set
        all_outputs = []
        all_labels = []
        model.eval()
        with torch.no_grad():
            for inputs, labels, add_feats in val_loader:
                inputs, labels, add_feats = inputs.to(device), labels.to(device), add_feats.to(device)
                outputs = torch.sigmoid(model(inputs, add_feats))
                all_outputs.extend(outputs.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
        
        all_outputs = np.array(all_outputs)
        all_labels = np.array(all_labels)
        predictions = (all_outputs > 0.5).astype(int)
        
        metrics = {
            "accuracy": accuracy_score(all_labels, predictions),
            "precision": precision_score(all_labels, predictions, zero_division=0),
            "recall": recall_score(all_labels, predictions, zero_division=0),
            "f1_score": f1_score(all_labels, predictions, zero_division=0),
            "roc_auc": roc_auc_score(all_labels, all_outputs),
        }
        fold_metrics.append(metrics)
        print(f"Metrics for fold {fold + 1}: {metrics}")
    
    # Aggregate results across folds
    mean_metrics = {key: np.mean([fold[key] for fold in fold_metrics]) for key in fold_metrics[0]}
    mean_metrics.update({"window_size": window_size, "lead_time": lead_time})
    
    return mean_metrics




import csv
import os

# Path for the results file
results_file = "results.csv"

# Ensure the file exists, and write the header if it doesn't
if not os.path.exists(results_file):
    with open(results_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "window_size", "lead_time", "accuracy", "precision", 
            "recall", "f1_score", "roc_auc"
        ])

# Run experiments
try:
    for window_size in range(1, 13):
        for lead_time in range(1, 13):
            print(f"Running model for window_size={window_size}, lead_time={lead_time}")
            result = run_experiment_with_cv(window_size, lead_time, epochs=100, device=device)
            
            # Append result to CSV
            with open(results_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    result["window_size"], result["lead_time"], 
                    result["accuracy"], result["precision"], 
                    result["recall"], result["f1_score"], 
                    result["roc_auc"]
                ])
            
            print(f"Result saved: {result}")
except KeyboardInterrupt:
    print("Experiment interrupted. Results saved so far.")

# Print completion message
print("Experiment complete. Results saved to", results_file)

