import xarray as xr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score
import os

# Load the dataset and subset to the specified region
file_path = "sst_data.nc"  # Path to the uploaded file
sst_broad = xr.open_dataset(file_path)['sst'].sel(
    lat=slice(-15, 15), lon=slice(170, 260)
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data processing function
def process_data_multi_res(lead_time, resolution=1, seed=1975):
    torch.manual_seed(seed)

    # Define the label calculation region (Nino 3.4 region)
    nino34_lat_range = slice(-5, 5)
    nino34_lon_range = slice(190, 240)
    sst_nino34 = sst_broad.sel(lat=nino34_lat_range, lon=nino34_lon_range)

    # Calculate anomalies
    sst_anomalies = sst_broad.groupby("time.month") - sst_broad.groupby("time.month").mean(dim="time")
    sst_nino34_anomalies = sst_nino34.groupby("time.month") - sst_nino34.groupby("time.month").mean(dim="time")

    # Compute labels for all time steps
    T = len(sst_nino34_anomalies['time'])
    labels = [
        1 if sst_nino34_anomalies.isel(time=t + lead_time).mean(dim=["lat", "lon"]).values > 0.5 else 0
        for t in range(T - lead_time)
    ]

    # Downsample data based on the resolution
    sst_downsampled = sst_anomalies.isel(lat=slice(0, None, resolution), lon=slice(0, None, resolution))
    data = np.array([
        sst_downsampled.isel(time=t).values for t in range(T - lead_time)
    ])

    # Add a channel dimension for CNN compatibility
    data = data[:, np.newaxis, :, :]

    return data, np.array(labels)

# CNN model definition
class SimpleCNN(nn.Module):
    def __init__(self, input_channels, input_height, input_width):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=4, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=4, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate the size of the flattened features dynamically
        with torch.no_grad():
            sample_input = torch.zeros(1, input_channels, input_height, input_width)
            sample_output = self.pool(torch.relu(self.conv2(self.pool(torch.relu(self.conv1(sample_input))))))
            self.flattened_size = sample_output.view(-1).size(0)

        self.fc1 = nn.Linear(self.flattened_size, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Dataset class
class SSTDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

# Training and evaluation function
def run_experiment(lead_time=12, resolution=1, epochs=50, k_folds=5):
    data, labels = process_data_multi_res(lead_time, resolution)
    dataset = SSTDataset(data, labels)

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_metrics = []
    best_model = None
    best_accuracy = 0.0
    best_model_state = None

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"Fold {fold + 1}/{k_folds}")
        train_data = torch.utils.data.Subset(dataset, train_idx)
        val_data = torch.utils.data.Subset(dataset, val_idx)

        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

        # Define model, loss, and optimizer
        input_height, input_width = data.shape[2], data.shape[3]  # Extract height and width from data
        model = SimpleCNN(input_channels=1, input_height=input_height, input_width=input_width).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        epoch_metrics = []  # Track out-of-sample metrics for each epoch

        for epoch in range(epochs):
            # Train the model
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device).view(-1, 1)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # Evaluate the model
            model.eval()
            val_preds, val_labels = [], []
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = torch.sigmoid(model(inputs))
                    val_preds.extend(outputs.cpu().numpy().flatten())
                    val_labels.extend(labels.cpu().numpy().flatten())

            val_preds_binary = (np.array(val_preds) > 0.5).astype(int)
            accuracy = accuracy_score(val_labels, val_preds_binary)
            precision = precision_score(val_labels, val_preds_binary, zero_division=1)
            recall = recall_score(val_labels, val_preds_binary, zero_division=1)
            epoch_metrics.append((epoch + 1, accuracy, precision, recall))

            # Check if this is the best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_state = model.state_dict()
                best_model = model

            print(f"Epoch {epoch + 1}: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")

        fold_metrics.append(epoch_metrics)

    # Save the best model
    if best_model_state is not None:
        torch.save(best_model_state, "best_model.pth")
        print(f"Best model saved with accuracy={best_accuracy:.4f}")

    return fold_metrics

# Run the experiment for a specific resolution and lead time
resolution = 2  # Modify as needed
lead_time = 12  # Modify as needed
run_experiment(lead_time=lead_time, resolution=resolution)